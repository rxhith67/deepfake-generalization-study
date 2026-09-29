"""Measure raw external bias before fixed five-landmark normalization."""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import SimilarityTransform

from src.data.detect_faces import create_mtcnn
from src.experiments.common import configuration, record_run, save_csv, save_json, atomic_text, update_master
from src.experiments.evaluation import infer, measurement, strict_ensemble, probability_figures
from src.experiments.manifests import canonical_id

# Fixed 5-point face template in a 112px reference coordinate system.
TEMPLATE = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                     [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float64)


def align_face(rgb, landmarks, size=224):
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.shape != (5,2) or not np.isfinite(landmarks).all():
        raise ValueError("Five finite landmark coordinates required")
    transform = SimilarityTransform()
    if not transform.estimate(landmarks, TEMPLATE * size / 112):
        raise ValueError("Degenerate landmarks")
    matrix = transform.params[:2].astype(np.float32)
    aligned = cv2.warpAffine(rgb, matrix, (size,size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
    return aligned, matrix


def image_statistics(rgb):
    unit = rgb.astype(np.float32) / 255
    gray = cv2.cvtColor(unit, cv2.COLOR_RGB2GRAY)
    work = cv2.resize(gray, (224,224), interpolation=cv2.INTER_AREA)
    power = np.abs(np.fft.fft2(work - work.mean()))**2
    fy, fx = np.meshgrid(np.fft.fftfreq(224), np.fft.fftfreq(224), indexing="ij")
    radius = np.hypot(fx,fy) / .5
    result = {"brightness": float(gray.mean()), "contrast": float(gray.std()),
              "sharpness": float(cv2.Laplacian(work,cv2.CV_32F).var()),
              "high_frequency_energy": float(power[radius > .3].sum() / max(power.sum(),1e-12))}
    for i, name in enumerate("rgb"):
        result[f"mean_{name}"] = float(unit[...,i].mean())
        result[f"std_{name}"] = float(unit[...,i].std())
    return result


def prepare(config):
    root = Path(config["output_dir"]) / "02_external_bias_audit"
    if (root / "normalization_complete.json").exists():
        return root / "normalized.csv"
    run = record_run(root, config, [config["data"]["external_csv"]])
    original = pd.read_csv(config["data"]["external_csv"])
    original["sample_id"] = original.image_path.map(canonical_id)
    key_lookup = {(int(r.label), "/".join(r.sample_id.split("/")[2:])): r for r in original.itertuples()}
    detector = create_mtcnn("cuda" if config["device"] in ("auto","cuda") and __import__('torch').cuda.is_available() else "cpu")
    metadata, candidates = [], []
    for label, name in [(0,"real"),(1,"fake")]:
        raw_root = Path(config["data"]["external_raw"]) / name
        for path in sorted(raw_root.rglob("*.jpg")):
            relative = path.relative_to(raw_root).as_posix()
            entry = {"raw_path": str(path), "relative_key": relative, "label": label,
                     "in_original_cohort": (label, relative) in key_lookup, "bytes": path.stat().st_size,
                     "extension": path.suffix.lower()}
            with Image.open(path) as image:
                entry.update(width=image.width, height=image.height, aspect_ratio=image.width/image.height,
                             color_mode=image.mode, image_format=image.format,
                             jpeg_quantization_present=hasattr(image,"quantization"))
                tables = getattr(image,"quantization", {})
                entry['jpeg_quantization_mean'] = float(np.mean([v for table in tables.values() for v in table])) if tables else None
                rgb = np.asarray(image.convert("RGB"))
            entry.update(image_statistics(rgb))
            boxes, probabilities, landmarks = detector.detect(rgb, landmarks=True)
            if boxes is None:
                entry['status'] = 'no_face'
            else:
                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                index = int(np.argmax(areas))
                box, points = boxes[index], landmarks[index]
                entry.update(face_area_ratio=float(areas[index]/(rgb.shape[0]*rgb.shape[1])),
                             face_crop_width=float(box[2]-box[0]), face_crop_height=float(box[3]-box[1]),
                             detection_confidence=float(probabilities[index]), status="detected")
                entry.update({f"box_{i}":float(v) for i,v in enumerate(box)})
                entry.update({f"landmark_{i}_{axis}":float(points[i,j]) for i in range(5) for j,axis in enumerate('xy')})
                if entry['in_original_cohort']:
                    candidates.append((path, label, relative, points))
            metadata.append(entry)
        print(f"External raw audit: {name} complete", flush=True)
    # Commit raw measurements BEFORE generating/evaluating normalized images.
    meta = pd.DataFrame(metadata)
    save_csv(root / "external_metadata.csv", meta)
    fig, axes = plt.subplots(2,4,figsize=(15,7))
    for axis,column in zip(axes.flat, ['width','height','aspect_ratio','brightness','contrast','sharpness','high_frequency_energy','face_area_ratio']):
        for label,name in [(0,'real'),(1,'fake')]:
            axis.hist(meta.loc[meta.label == label,column].dropna(), bins=25, histtype='step', label=name)
        axis.set_title(column)
        axis.legend()
    fig.tight_layout()
    (root / "source_distribution_plots").mkdir(exist_ok=True)
    fig.savefig(root / "source_distribution_plots" / "raw_distributions.png",dpi=250)
    plt.close(fig)
    normalized, failures = [], []
    for path,label,relative,points in candidates:
        try:
            with Image.open(path) as image:
                aligned,matrix = align_face(np.asarray(image.convert('RGB')),points,config['external_normalization']['image_size'])
            destination = root / "normalized_external" / ("fake" if label else "real") / relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            ok = cv2.imwrite(str(destination),cv2.cvtColor(aligned,cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY,config['external_normalization']['jpeg_quality']])
            if not ok:
                raise OSError(f"Could not save {destination}")
            row = key_lookup[(label,relative)]._asdict()
            row['image_path'] = str(destination.resolve())
            normalized.append(row)
        except (ValueError,cv2.error) as error:
            failures.append({'raw_path':str(path),'label':label,'reason':str(error)})
    norm = pd.DataFrame(normalized)
    save_csv(root / "normalized.csv", norm)
    save_csv(root / "original_matched.csv", original[original.sample_id.isin(norm.sample_id)])
    save_csv(root / "alignment_failures.csv", pd.DataFrame(failures,columns=['raw_path','label','reason']))
    save_json(root / "normalization_complete.json", {'run_id':run['run_id'],'raw_count':len(meta),'original_count':len(original),
              'retained_count':len(norm),'retained_by_label':norm.label.value_counts().to_dict(),
              'template_112':TEMPLATE.tolist(),'border_rgb':[127,127,127],'config':config['external_normalization']})
    return root / "normalized.csv"


def evaluate(config):
    import json
    root = Path(config['output_dir']) / '02_external_bias_audit'
    metrics, conditions = [], {}
    for condition, manifest in [('original_matched',root/'original_matched.csv'),('normalized',root/'normalized.csv')]:
        frames = []
        for model, checkpoint in config['checkpoints'].items():
            rows = infer(checkpoint,manifest,config,root/f'{model}_{condition}_predictions.csv',f'external_{condition}',
                         'external_test',config['historical_protocol_id'],condition=condition)
            frames.append(rows)
        saved = json.loads(Path('outputs/tables/ensemble_200video_cloud_frame.json').read_text())
        ensemble = strict_ensemble(frames,saved['weights'],saved['calibration_metrics']['threshold'])
        frames.append(ensemble)
        save_csv(root/f'ensemble_{condition}_predictions.csv',ensemble)
        for rows in frames:
            metrics.append(measurement(rows,'external_bias_audit',config,ci=True))
        conditions[condition] = {r.model.iloc[0]:r for r in frames}
        probability_figures(conditions[condition], root / condition)
    save_csv(root/'original_vs_normalized_metrics.csv',pd.DataFrame(metrics))
    update_master(config['output_dir'],metrics)
    table = pd.DataFrame(metrics).pivot(index='model',columns='condition',values='auc')
    axis = table.plot.bar(ylim=(0,1),ylabel='ROC-AUC',rot=0)
    axis.axhline(.5,color='black',linestyle='--')
    axis.figure.tight_layout()
    axis.figure.savefig(root/'original_vs_normalized_auc.png',dpi=250)
    plt.close(axis.figure)
    atomic_text(root/'bias_audit.md','# External source/preprocessing audit\n\n'
                'Original and normalized comparisons use identical retained sample IDs. '
                'Both classes use the same five-landmark similarity alignment and explicit JPEG Q90. '
                'This differs from historical bounding-box-only crops. Source/content confounding remains; '
                'a change in AUC is not proof of a single causal explanation. Raw metadata precedes normalization. '
                'See normalization_complete.json for retention and external_metadata.csv for detection failures.\n\n'
                +table.to_string()+'\n')


if __name__ == '__main__':
    config = configuration()
    prepare(config)
    evaluate(config)
