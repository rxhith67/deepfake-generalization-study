"""Evaluate approved controls/LOMO checkpoints and identical corruption grids."""
from pathlib import Path
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments.common import save_csv,save_json,record_run,update_master
from src.experiments.evaluation import infer,measurement,strict_ensemble,probability_figures
from src.experiments.calibration import select_threshold
from src.experiments.manifests import METHODS,PROTOCOL
from src.experiments.corruptions import jpeg,frequency_filter,working_image
from src.experiments.interpretation import representation,cam_panels


def run(config,training_root,protocol_root,external_original,external_normalized):
    root = Path(config['output_dir'])
    training_root,protocol_root = Path(training_root),Path(protocol_root)
    destination = root/'01_leave_one_manipulation_out'
    record_run(destination,config,[protocol_root/'control/test.csv',external_normalized])
    checkpoints = {m:training_root/'control'/m/'checkpoints'/f'{m}_best.pt' for m in ['hybrid','xception','freq_cnn']}
    predictions,results,controls = {},[],{}
    datasets = {'val':(protocol_root/'control/val.csv','ffpp_source_safe','val'),
                'test':(protocol_root/'control/test.csv','ffpp_source_safe','test'),
                'external_original':(external_original,'external_original','external_test'),
                'external_normalized':(external_normalized,'external_normalized','external_test')}
    for model,checkpoint in checkpoints.items():
        if not (checkpoint.parent.parent/'complete.json').exists():
            raise ValueError(f'Incomplete control: {model}')
        for name,(manifest,dataset,split) in datasets.items():
            rows = infer(checkpoint,manifest,config,destination/f'control_{model}_{name}.csv',dataset,split,PROTOCOL)
            predictions[(model,name)] = rows
            if name != 'val':
                results.append(measurement(rows,'matched_control',config,ci=True))
            if name == 'test':
                controls[model] = rows
                results.append(measurement(rows,'matched_control',config,unit='video',ci=True))
                for method in METHODS:
                    metric = measurement(rows[(rows.label==0)|(rows.method==method)],'control_per_method',config)
                    metric['test_manipulation'] = method
                    results.append(metric)
    candidates = []
    for weight in np.linspace(0,1,101):
        frame = strict_ensemble([predictions[('hybrid','val')],predictions[('xception','val')]],[weight,1-weight],.5)
        candidates.append((roc_auc_score(frame.label,frame.prob_fake),weight))
    _,weight = max(candidates)
    val = strict_ensemble([predictions[('hybrid','val')],predictions[('xception','val')]],[weight,1-weight],.5)
    threshold = select_threshold(val)
    save_json(destination/'ensemble_validation.json',{'weights':[weight,1-weight],'threshold':threshold,
                                                      'selection':'FF++ source-safe validation AUC; threshold by validation BA'})
    for name in ['test','external_original','external_normalized']:
        rows = strict_ensemble([predictions[('hybrid',name)],predictions[('xception',name)]],[weight,1-weight],threshold)
        save_csv(destination/f'control_ensemble_{name}.csv',rows)
        results.append(measurement(rows,'matched_control',config,ci=True))
    lomo_metrics,heatmap = [],[]
    for method in METHODS:
        for model in ('hybrid','xception'):
            checkpoint = training_root/f'lomo_{method}'/model/'checkpoints'/f'{model}_best.pt'
            if not (checkpoint.parent.parent/'complete.json').exists():
                raise ValueError(f'Incomplete LOMO: {model}/{method}')
            # Supplementary known-method test columns are evaluated only after checkpoint selection.
            all_test = infer(checkpoint,protocol_root/'control/test.csv',config,destination/f'lomo_{method}_{model}_all_test.csv',
                             'ffpp_source_safe','test',PROTOCOL,condition=f'lomo_{method}')
            for column in METHODS:
                subset = all_test[(all_test.label==0)|(all_test.method==column)]
                metric = measurement(subset,'lomo',config,ci=column==method)
                metric.update(test_manipulation=column,held_out=method,unseen=column==method)
                heatmap.append(metric)
                if column == method:
                    known = controls[model]
                    known = known[(known.label==0)|(known.method==method)]
                    if set(known.sample_id) != set(subset.sample_id):
                        raise ValueError('Known/LOMO cohort mismatch')
                    metric['known_auc'] = roc_auc_score(known.label,known.prob_fake)
                    metric['generalization_gap'] = metric['known_auc']-metric['auc']
                    lomo_metrics.append(metric)
            # Explain every fold using the same complete test manifest and explicit condition.
            if model == 'xception':
                heldout_rows = all_test[(all_test.label==0)|(all_test.method==method)].reset_index(drop=True)
                cam_panels(config,checkpoint,protocol_root/f'lomo_{method}/test.csv',heldout_rows,f'lomo_{method}')
    results += lomo_metrics
    save_csv(destination/'leave_one_out_summary.csv',pd.DataFrame(lomo_metrics))
    save_csv(destination/'cross_manipulation.csv',pd.DataFrame(heatmap))
    save_csv(destination/'matched_control_metrics.csv',pd.DataFrame([r for r in results if r['experiment']!='lomo']))
    save_csv(destination/'macro_lomo_summary.csv',pd.DataFrame(lomo_metrics).groupby('model')[['auc','known_auc','generalization_gap']].mean().reset_index())
    for model in ('hybrid','xception'):
        data = pd.DataFrame(heatmap)
        matrix = data[data.model==model].pivot(index='held_out',columns='test_manipulation',values='auc').reindex(index=METHODS,columns=METHODS)
        fig,axis = plt.subplots(figsize=(6,5))
        image = axis.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
        for (i,j),value in np.ndenumerate(matrix.to_numpy()):
            axis.text(j,i,f'{value:.3f}'+(' *' if i==j else ''),ha='center',va='center',color='white')
        axis.set(xticks=range(4),yticks=range(4),xticklabels=METHODS,yticklabels=METHODS,title=f'{model}: * unseen method',ylabel='Held out during train/val')
        axis.tick_params(axis='x',rotation=30)
        fig.colorbar(image,ax=axis,label='AUC');fig.tight_layout();fig.savefig(destination/f'{model}_heatmap.png',dpi=250);plt.close(fig)
        gaps = pd.DataFrame(lomo_metrics)
        gaps = gaps[gaps.model==model].set_index('held_out')
        axis = gaps[['known_auc','auc']].plot.bar(ylim=(0,1),rot=30,ylabel='AUC',title=f'{model}: identical per-method test cohorts')
        axis.figure.tight_layout();axis.figure.savefig(destination/f'{model}_known_vs_unseen.png',dpi=250);plt.close(axis.figure)
    update_master(root,results)
    # Shared clean references, explicit Q100 re-encoding, then signed-frequency interventions.
    for experiment,folder,conditions in [
        ('jpeg','04_jpeg_robustness',[(f'Q{q}',partial(jpeg,quality=q),{'jpeg_quality':q}) for q in config['jpeg_qualities']]),
        ('frequency_sensitivity','07_frequency_sensitivity',[(f'{kind}_{cutoff}',partial(frequency_filter,kind=kind,cutoff=cutoff),
           {'filter_type':kind,'filter_cutoff':cutoff}) for kind in ['lowpass','highpass'] for cutoff in config['frequency']['cutoffs']])]:
        grid_results = []
        for model,checkpoint in checkpoints.items():
            for name in ['test','external_normalized']:
                manifest,dataset,split = datasets[name]
                clean = predictions[(model,name)]
                clean_auc = roc_auc_score(clean.label,clean.prob_fake)
                reference = clean_auc
                for condition,transform,parameters in conditions:
                    rows = infer(checkpoint,manifest,config,root/folder/f'{model}_{name}_{condition}.csv',dataset,split,PROTOCOL,transform,condition)
                    if set(rows.sample_id) != set(clean.sample_id):
                        raise ValueError('Corruption cohort changed')
                    metric = measurement(rows,experiment,config)
                    if condition == 'Q100':
                        reference = metric['auc']
                    metric.update(parameters,clean_auc=clean_auc,reference_auc=reference,
                                  absolute_auc_drop=reference-metric['auc'],relative_degradation=(reference-metric['auc'])/reference)
                    grid_results.append(metric)
        save_csv(root/folder/'metrics.csv',pd.DataFrame(grid_results))
        update_master(root,grid_results)
        for dataset in ['ffpp_source_safe','external_normalized']:
            data = pd.DataFrame(grid_results)
            data = data[data.test_dataset==dataset]
            fig,axes = plt.subplots(1,2,figsize=(12,4))
            for model,frame in data.groupby('model'):
                for axis,column in zip(axes,['auc','relative_degradation']):
                    axis.plot(frame.condition,frame[column],marker='o',label=model)
                    axis.set(ylabel=column,title=dataset)
                    axis.tick_params(axis='x',rotation=35)
                    axis.legend()
            axes[0].set_ylim(0,1)
            fig.tight_layout();fig.savefig(root/folder/f'{dataset}_curves.png',dpi=250);plt.close(fig)
    representation(config,checkpoints['hybrid'],protocol_root/'control/test.csv',external_normalized)
    for name in ['test','external_normalized']:
        cam_panels(config,checkpoints['xception'],datasets[name][0],predictions[('xception',name)],name)
    return results
