"""Shared, cached inference and probability reporting for the upgrade."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import FaceDataset
from src.data.transforms import build_transform, transform_options
from src.evaluate import predict, aggregate_predictions
from src.eval.metrics import report_all
from src.eval.calibration import calibration_metrics, bootstrap_auc, reliability
from src.experiments.common import fingerprint, sha256, save_csv, save_json
from src.experiments.manifests import canonical_id
from src.utils import load_checkpoint, resolve_device


def infer(checkpoint, manifest, config, output, dataset_name, split, protocol, corruption=None, condition="original"):
    output = Path(output)
    manifest = Path(manifest)
    identity = {"checkpoint": sha256(checkpoint), "manifest": sha256(manifest), "condition": condition,
                "dataset": dataset_name, "split": split, "protocol": protocol, "seed": config["seed"],
                "config": config}
    key = fingerprint(identity)
    meta_path = output.with_suffix(".meta.json")
    if output.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_key") == key and meta.get("prediction_sha256") == sha256(output):
            print(f"Cache verified: {output}", flush=True)
            return pd.read_csv(output)
        raise ValueError(f"Cache mismatch; use a new run directory: {output}")
    print(f"Inference: {Path(checkpoint).name} / {dataset_name} / {split} / {condition}", flush=True)
    device = resolve_device(config["device"])
    model, saved = load_checkpoint(checkpoint, device)
    transform = build_transform(**transform_options(saved.get("config", {})))
    if corruption is not None:
        base_transform = transform
        transform = lambda image: base_transform(image=corruption(image))
    dataset = FaceDataset(manifest, config["data"]["root"], transform, return_metadata=True)
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=False)
    labels, probabilities, indices = predict(model, loader, device)
    rows = dataset.frame.iloc[indices].reset_index(drop=True).copy()
    if "sample_id" not in rows:
        rows["sample_id"] = rows.image_path.map(canonical_id)
    if rows.sample_id.duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    if "target_group" not in rows and dataset_name.startswith("ffpp"):
        rows["target_group"] = rows.video_id.astype(str)
    rows["label"] = labels
    rows["prob_fake"] = probabilities
    rows["threshold"] = float(saved["threshold"])
    rows["prediction"] = (rows.prob_fake >= saved["threshold"]).astype(int)
    rows["model"] = saved["model_name"]
    rows["dataset"] = dataset_name
    rows["split"] = split
    rows["protocol_id"] = protocol
    rows["condition"] = condition
    rows["threshold_source"] = "ffpp_validation_checkpoint"
    rows["checkpoint_sha256"] = identity["checkpoint"]
    save_csv(output, rows)
    save_json(meta_path, {**identity, "cache_key": key, "prediction_sha256": sha256(output),
                         "threshold": saved["threshold"], "transform": transform_options(saved.get("config", {})),
                         "architecture_class": type(model).__name__})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def strict_ensemble(frames, weights, threshold, name="ensemble"):
    if len(frames) != len(weights) or not frames:
        raise ValueError("Matching frames/weights required")
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid ensemble weights")
    reference = frames[0].set_index("sample_id", verify_integrity=True)
    matrices = []
    for frame in frames:
        frame = frame.set_index("sample_id", verify_integrity=True)
        if set(reference.index) != set(frame.index):
            raise ValueError("Ensemble sample coverage mismatch")
        frame = frame.loc[reference.index]
        for column in ("label", "split", "protocol_id", "condition", "dataset"):
            if column in reference and not reference[column].equals(frame[column]):
                raise ValueError(f"Ensemble {column} mismatch")
        matrices.append(frame.prob_fake.to_numpy())
    result = reference.reset_index().copy()
    result["prob_fake"] = np.stack(matrices, axis=1) @ (weights / weights.sum())
    result["threshold"] = threshold
    result["prediction"] = (result.prob_fake >= threshold).astype(int)
    result["model"] = name
    result["checkpoint_sha256"] = "ensemble:see_run_config"
    return result


def measurement(rows, experiment, config, unit="frame", ci=False):
    threshold = float(rows.threshold.iloc[0])
    if rows.threshold.nunique() != 1:
        raise ValueError("Mixed thresholds in evaluation")
    data = rows
    if unit == "video":
        if not str(rows.dataset.iloc[0]).startswith("ffpp"):
            raise ValueError("External images are not videos")
        data = aggregate_predictions(rows, ["dataset", "method", "video_id"])
    result = report_all(data.label, data.prob_fake, threshold)
    result.pop("confusion")
    result.update(calibration_metrics(data.label, data.prob_fake, config["calibration_bins"]))
    result.update({"experiment": experiment, "model": rows.model.iloc[0], "test_dataset": rows.dataset.iloc[0],
                   "protocol_id": rows.protocol_id.iloc[0], "evaluation_unit": unit, "condition": rows.condition.iloc[0],
                   "threshold_source": rows.threshold_source.iloc[0], "seed": config["seed"],
                   "num_samples": len(data), "score_type": "probability", "split": rows.split.iloc[0]})
    if ci:
        # Target-video clusters keep all frames and related manipulation rows together.
        groups = data.video_id.astype(str) if str(rows.dataset.iloc[0]).startswith("ffpp") else None
        result.update(bootstrap_auc(data.label, data.prob_fake, groups, config["bootstrap_replicates"], config["seed"]))
        result["bootstrap_unit"] = "target_video" if groups is not None else "image"
    return result


def probability_figures(frames, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for title, rows in frames.items():
        fpr, tpr, _ = roc_curve(rows.label, rows.prob_fake)
        axes[0].plot(fpr, tpr, label=title)
        bins = [r for r in reliability(rows.label, rows.prob_fake) if r["count"]]
        axes[1].plot([r["mean_probability"] for r in bins], [r["fraction_fake"] for r in bins], marker=".", label=title)
        for label in (0, 1):
            axes[2].hist(rows.loc[rows.label == label, "prob_fake"], bins=np.linspace(0,1,26),
                         density=True, histtype="step", label=f"{title} {'fake' if label else 'real'}")
    for axis in axes[:2]:
        axis.plot([0,1], [0,1], "k--", alpha=.4)
        axis.set(xlim=(0,1), ylim=(0,1))
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed fake fraction", title="Reliability (15 bins)")
    axes[2].set(xlabel="P(fake)", ylabel="Density", title="Score distributions", xlim=(0,1))
    for axis in axes:
        axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(directory / "roc_reliability_distributions.png", dpi=250)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(frames), figsize=(4*len(frames), 4), squeeze=False)
    for axis, (title, rows) in zip(axes.flat, frames.items()):
        matrix = confusion_matrix(rows.label, rows.prediction, labels=[0,1])
        axis.imshow(matrix, cmap="Blues")
        for (i,j), value in np.ndenumerate(matrix):
            axis.text(j,i,str(value),ha="center",va="center")
        axis.set(title=title, xlabel="Predicted", ylabel="True", xticks=[0,1], yticks=[0,1])
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=250)
    plt.close(fig)
