"""Local inference + calibration for one completed source-video-disjoint control checkpoint.

Thresholds are re-selected from FF++ source-disjoint validation predictions only and then frozen.
External data are evaluated with that frozen threshold and never used for any selection.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from src.eval.calibration import calibration_metrics, reliability
from src.experiments.calibration import select_threshold
from src.experiments.common import configuration, record_run, save_csv, save_json, sha256, atomic_text, update_master
from src.experiments.evaluation import infer, measurement, probability_figures
from src.experiments.manifests import METHODS, PROTOCOL

FF = "ffpp_source_safe"
DISPLAY = {"hybrid": "CoAtNet", "xception": "Xception", "freq_cnn": "RGB+FFT dual-domain CNN"}
METRICS = ("auc", "balanced_accuracy", "ece", "nll", "brier")


def components(source_ids):
    """Connected components over participant IDs: videos sharing any participant stay in one cluster."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for value in source_ids.unique():
        ids = value.split("|")
        for other in ids[1:]:
            parent[find(other)] = find(ids[0])
        find(ids[0])
    return source_ids.map(lambda v: "c" + find(v.split("|")[0]))


def add_clusters(rows):
    rows = rows.copy()
    rows["source_video_ids"] = rows.source_video_ids.astype(str)
    rows["cluster_component"] = components(rows.source_video_ids)
    rows["cluster_video"] = rows.method.astype(str) + "/" + rows.actual_video_id.astype(str)
    return rows


def video_level(rows):
    grouped = rows.groupby("cluster_video", sort=False).agg(
        label=("label", "first"), prob_fake=("prob_fake", "mean"), method=("method", "first"),
        target_group=("target_group", "first"), cluster_component=("cluster_component", "first"),
        cluster_video=("cluster_video", "first"), frames=("label", "size")).reset_index(drop=True)
    grouped["threshold"] = rows.threshold.iloc[0]
    return grouped


def point(rows, bins):
    y, p, t = rows.label.to_numpy(), rows.prob_fake.to_numpy(), float(rows.threshold.iloc[0])
    out = {"auc": roc_auc_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, (p >= t).astype(int))}
    out.update(calibration_metrics(y, p, bins))
    return out


def cluster_ci(rows, cluster_column, config, metrics=METRICS):
    """95% percentile CI from resampling whole clusters (all frames of a cluster move together)."""
    labels, probs = rows.label.to_numpy(), rows.prob_fake.to_numpy()
    threshold = float(rows.threshold.iloc[0])
    _, inverse = np.unique(rows[cluster_column].to_numpy(), return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
    rng = np.random.default_rng(config["seed"])
    draws, valid = [], 0
    for _ in range(config["bootstrap_replicates"]):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(labels[idx])) < 2:
            continue
        valid += 1
        sample = pd.DataFrame({"label": labels[idx], "prob_fake": probs[idx], "threshold": threshold})
        draws.append(point(sample, config["calibration_bins"]))
    frame = pd.DataFrame(draws)
    out = {"bootstrap_valid": valid, "n_clusters": len(members)}
    for m in metrics:
        low, high = np.quantile(frame[m], [.025, .975])
        out[f"{m}_ci_lower"], out[f"{m}_ci_upper"] = float(low), float(high)
    return out


def rescore(rows, threshold, source):
    rows = rows.copy()
    rows["checkpoint_saved_threshold"] = rows.threshold
    rows["threshold"] = threshold
    rows["prediction"] = (rows.prob_fake >= threshold).astype(int)
    rows["threshold_source"] = source
    return rows


def evaluate(model, config, training_note="source_disjoint_control"):
    root = Path(config["output_dir"])
    dest = root / "01_matched_controls" / model
    checkpoint = dest / f"{model}_best.pt"
    complete = json.loads((dest / "complete.json").read_text())
    if sha256(checkpoint) != complete["checkpoint_sha256"]:
        raise ValueError("Local checkpoint hash differs from Modal complete.json")
    protocol = root / "protocol" / "control"
    ext = root / "02_external_bias_audit"
    sets = {"val": (protocol / "val.csv", FF, "val"), "test": (protocol / "test.csv", FF, "test"),
            "external_original_matched": (ext / "original_matched.csv", "external_original_matched", "external_test"),
            "external_normalized": (ext / "normalized.csv", "external_normalized", "external_test")}
    record_run(dest / "evaluation", config, [checkpoint, *[m for m, _, _ in sets.values()]])
    raw = {}
    for name, (manifest, dataset, split) in sets.items():
        raw[name] = infer(checkpoint, manifest, config, dest / "evaluation" / f"{model}_{name}_raw.csv",
                          dataset, split, PROTOCOL, condition=("original" if name in ("val", "test") else name))
    # Validation-only threshold; recorded next to the checkpoint's own epoch-selected threshold.
    threshold = select_threshold(raw["val"])
    saved_threshold = float(raw["val"].threshold.iloc[0])
    scored = {n: rescore(r, threshold, "source_disjoint_validation_frozen") for n, r in raw.items()}
    scored["test"] = add_clusters(scored["test"])
    for n, r in scored.items():
        save_csv(dest / "predictions" / f"{model}_{n}_predictions.csv", r)
    save_json(dest / "evaluation" / "threshold.json", {
        "validation_selected_threshold": threshold, "checkpoint_saved_threshold": saved_threshold,
        "rule": "maximize balanced accuracy on source-disjoint FF++ validation (nearest 0.5 on ties)",
        "external_used_for_selection": False})

    master, table = [], []
    for name, rows in scored.items():
        if name == "val":
            metric = measurement(rows, "matched_control_validation", config)
            table.append(metric)
            continue
        units = [("frame", rows)] + ([("video", video_level(rows))] if name == "test" else [])
        for unit, data in units:
            if unit == "video":
                data = data.assign(dataset=rows.dataset.iloc[0], model=rows.model.iloc[0], split="test",
                                   protocol_id=PROTOCOL, condition="original",
                                   threshold_source=rows.threshold_source.iloc[0])
            metric = measurement(data, "matched_control", config)  # rows are already at the requested unit
            metric["evaluation_unit"] = unit
            if name == "test":
                tg = cluster_ci(data, "target_group", config)
                cc = cluster_ci(data, "cluster_component", config)
                metric.update({f"{k}_target_group": v for k, v in tg.items()})
                metric.update({f"{k}_source_component": v for k, v in cc.items()})
                for m in METRICS:  # headline CI = the wider, source-component cluster interval
                    metric[f"{m}_ci_lower"], metric[f"{m}_ci_upper"] = cc[f"{m}_ci_lower"], cc[f"{m}_ci_upper"]
                metric["auc_ci_lower"], metric["auc_ci_upper"] = cc["auc_ci_lower"], cc["auc_ci_upper"]
                metric["bootstrap_unit"] = "source_video_component"
            else:
                ci = cluster_ci(data.assign(g=np.arange(len(data))), "g", config)
                metric.update(ci)
                metric["bootstrap_unit"] = "image (external images are not grouped by source)"
            table.append(metric)
        if name == "test":
            for method in METHODS:
                subset = rows[(rows.label == 0) | (rows.method == method)]
                metric = measurement(subset, "control_per_method", config)
                metric["test_manipulation"] = method
                metric.update({k: v for k, v in cluster_ci(subset, "cluster_component", config, ("auc",)).items()})
                metric["bootstrap_unit"] = "source_video_component"
                table.append(metric)
    for metric in table:
        metric["model_display"] = DISPLAY.get(model, model)
        metric["training_run"] = training_note
        metric["checkpoint_sha256"] = complete["checkpoint_sha256"]
    metrics = pd.DataFrame(table)
    save_csv(dest / "evaluation" / "metrics.csv", metrics)
    update_master(root, [m for m in table if m["experiment"] != "matched_control_validation"])

    # Calibration artifacts (validation shown for reference only; never used for scoring claims).
    cal = dest / "calibration"
    reliab = []
    for name, rows in scored.items():
        for r in reliability(rows.label, rows.prob_fake, config["calibration_bins"]):
            reliab.append({"model": model, "dataset": name, **r})
    save_csv(cal / "reliability_bins.csv", pd.DataFrame(reliab))
    save_csv(cal / "confidence_distributions.csv", pd.concat(scored.values(), ignore_index=True)[
        ["sample_id", "model", "dataset", "split", "label", "method", "prob_fake", "threshold", "prediction"]])
    transfer = []
    for name, rows in scored.items():
        m = measurement(rows, "threshold_transfer", config)
        transfer.append({k: m[k] for k in ["model", "test_dataset", "threshold", "threshold_source", "auc", "accuracy",
                         "balanced_accuracy", "recall", "specificity", "false_positive_rate", "f1", "ece", "nll", "brier"]}
                        | {"role": "selection" if name == "val" else "transfer", "n": len(rows)})
    save_csv(cal / "threshold_transfer.csv", pd.DataFrame(transfer))
    probability_figures({n: r for n, r in scored.items() if n != "val"}, cal / "figures")
    atomic_text(cal / "calibration.md", f"# {model} source-video-disjoint control: calibration\n\n"
                f"Threshold {threshold:.6f} selected on source-disjoint FF++ validation only "
                f"(checkpoint's own saved threshold: {saved_threshold:.6f}), frozen for all other sets. "
                "15 equal-width bins. External sets are transfer evaluations, not calibration/selection sets. "
                "Test CIs resample whole source-video connected components (frames are not independent).\n")
    return metrics


if __name__ == "__main__":
    print(evaluate(sys.argv[1], configuration())[["experiment", "test_dataset", "evaluation_unit", "auc",
          "auc_ci_lower", "auc_ci_upper", "balanced_accuracy", "ece", "nll", "brier"]].to_string())
