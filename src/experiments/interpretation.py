"""Full-feature representation analysis and deterministic fake-logit CAM panels."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data.dataset import FaceDataset
from src.data.transforms import build_transform,transform_options,denormalize_image
from src.experiments.common import record_run,save_csv,save_json
from src.experiments.frequency_analysis import sample_spread
from src.interpret.tsne import extract_embeddings,project_embeddings
from src.interpret.gradcam import SignedBinaryTarget,find_last_conv
from src.utils import load_checkpoint,resolve_device


def representation(config,checkpoint,ff_manifest,external_manifest):
    root = Path(config['output_dir'])/'09_representation_analysis'
    device = resolve_device(config['device'])
    model,saved = load_checkpoint(checkpoint,device)
    root = root/saved['model_name']
    record_run(root,config,[checkpoint,ff_manifest,external_manifest])
    arrays,metadata = [],[]
    transform = build_transform(**transform_options(saved['config']))
    for manifest,domain in [(ff_manifest,'FF++'),(external_manifest,'External')]:
        dataset = FaceDataset(manifest,config['data']['root'],transform,return_metadata=True)
        indices = []
        for _,frame in dataset.frame.groupby(['label','method']):
            indices.extend(sample_spread(frame,config['representation']['samples_per_group'],config['seed']).index.tolist())
        features,labels,probabilities,row_indices = extract_embeddings(model,dataset,indices,device,config['batch_size'],0)
        frame = dataset.frame.iloc[row_indices].copy().reset_index(drop=True)
        frame['label'],frame['prob_fake'],frame['domain'] = labels,probabilities,domain
        frame['group'] = domain+' '+frame.method.astype(str)
        arrays.append(features);metadata.append(frame)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    features = np.concatenate(arrays)
    rows = pd.concat(metadata,ignore_index=True)
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'features.npz',features=features)
    pca = PCA(svd_solver='full').fit(features)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    eigenweights = pca.explained_variance_ratio_
    rank = float(np.exp(-np.sum(eigenweights[eigenweights>0]*np.log(eigenweights[eigenweights>0]))))
    results = {'samples':len(rows),'dimensions':features.shape[1],
               'pca_scaling':'center only; original classifier-input units','effective_rank':rank,
               'effective_rank_definition':'exp(entropy(normalized covariance eigenvalues))',
               **{f'pcs_{int(q*100)}':int(np.searchsorted(cumulative,q)+1) for q in (.9,.95,.99)}}
    for label in ['label','group','domain']:
        results[f'silhouette_{label}'] = float(silhouette_score(features,rows[label],metric='euclidean'))
    # Within-domain real-vs-fake separation, also in the ORIGINAL feature space (never on t-SNE coordinates).
    for domain in ('FF++','External'):
        mask = (rows.domain==domain).to_numpy()
        results[f'silhouette_label_within_{domain.replace("+","p").lower()}'] = float(silhouette_score(features[mask],rows.label[mask],metric='euclidean'))
    results['silhouette_space'] = 'original classifier-input feature space, euclidean, unstandardized'
    save_json(root/'representation_metrics.json',results)
    save_csv(root/'pca_variance.csv',pd.DataFrame({'component':np.arange(1,len(cumulative)+1),'cumulative':cumulative}))
    coordinates = project_embeddings(features,config['representation']['perplexity'],config['seed'])
    rows['tsne_x'],rows['tsne_y'] = coordinates[:,0],coordinates[:,1]
    save_csv(root/'joint_coordinates.csv',rows)
    fig,axes = plt.subplots(1,3,figsize=(17,5))
    for axis,column in zip(axes,['label','group','domain']):
        for value,frame in rows.groupby(column):
            axis.scatter(frame.tsne_x,frame.tsne_y,s=8,alpha=.65,label=str(value))
        axis.set_title(column+' (same joint coordinates)')
        axis.set(xticks=[],yticks=[])
        axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'joint_tsne.png',dpi=250);plt.close(fig)
    fig,axis = plt.subplots(figsize=(7,4))
    axis.plot(np.arange(1,len(cumulative)+1),cumulative)
    for level in (.9,.95,.99):
        axis.axhline(level,ls='--',alpha=.5)
    axis.set(xlabel='Principal components',ylabel='Cumulative explained variance',ylim=(0,1))
    fig.tight_layout();fig.savefig(root/'pca_cumulative_variance.png',dpi=250);plt.close(fig)
    return results


def cam_panels(config,checkpoint,manifest,predictions,condition):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    device = resolve_device(config['device'])
    model,saved = load_checkpoint(checkpoint,device)
    root = Path(config['output_dir'])/'10_gradcam_failure_analysis'/saved['model_name']/condition
    record_run(root,config,[checkpoint,manifest])
    options = transform_options(saved['config'])
    dataset = FaceDataset(manifest,config['data']['root'],build_transform(**options),return_metadata=True)
    if len(dataset) != len(predictions):
        raise ValueError('CAM manifest/prediction count mismatch')
    # Predictions preserve manifest order; verify labels AND paths before indexing.
    if not np.array_equal(dataset.frame.label,predictions.label) or not np.array_equal(dataset.frame.image_path.astype(str),predictions.image_path.astype(str)):
        raise ValueError('CAM sample order mismatch')
    selected = []
    for name,y,p in [('TN',0,0),('TP',1,1),('FP',0,1),('FN',1,0)]:
        eligible = predictions[(predictions.label==y)&(predictions.prediction==p)]
        key = 'actual_video_id' if 'actual_video_id' in eligible else 'video_id'
        eligible = eligible.drop_duplicates(key).head(config['gradcam']['per_category'])
        selected.extend((index,name) for index in eligible.index)
    if not selected:
        raise ValueError('No CAM examples')
    fig,axes = plt.subplots((len(selected)+3)//4,4,figsize=(13,3.5*((len(selected)+3)//4)),squeeze=False)
    case_rows = []
    layer = find_last_conv(model)
    with GradCAM(model=model,target_layers=[layer]) as cam:
        for axis,(index,category) in zip(axes.flat,selected):
            tensor,label,_ = dataset[index]
            heatmap = cam(input_tensor=tensor.unsqueeze(0).to(device),targets=[SignedBinaryTarget(1)])[0]
            rgb = denormalize_image(tensor,options['mean'],options['std'])
            axis.imshow(show_cam_on_image(rgb,heatmap,use_rgb=True))
            row = predictions.iloc[index]
            axis.set_title(f'{category}: p(fake)={row.prob_fake:.3f}\nthreshold={row.threshold:.3f}; fake-logit CAM',fontsize=9)
            axis.axis('off')
            case_rows.append({**row.to_dict(),'category':category,'explanation_target':'fake_logit','target_layer':str(layer)})
    for axis in list(axes.flat)[len(selected):]:
        axis.axis('off')
    fig.tight_layout();fig.savefig(root/'gradcam.png',dpi=250);plt.close(fig)
    save_csv(root/'selected_cases.csv',pd.DataFrame(case_rows))
    save_json(root/'categories.json',{name:sum(category==name for _,category in selected) for name in ['TN','TP','FP','FN']})
