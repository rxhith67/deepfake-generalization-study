"""Standardized JPEG and low/high-pass sensitivity for the three matched source-video-disjoint controls.

No retraining and no per-condition threshold tuning: every threshold-dependent metric uses the
model's frozen clean, source-disjoint-validation-derived threshold. Reference (no extra
compression / no filter) and explicit Q100 re-encoding are separate conditions.
"""
import json
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from src.experiments.common import configuration, record_run, save_csv, sha256, atomic_text, update_master
from src.experiments.corruptions import jpeg, frequency_filter
from src.experiments.evaluation import infer, measurement
from src.experiments.manifests import PROTOCOL
from src.experiments.matched_control import DISPLAY, FF, rescore, add_clusters

MODELS = ("hybrid", "xception", "freq_cnn")


def controls(config):
    root = Path(config["output_dir"]) / "01_matched_controls"
    out = {}
    for model in MODELS:
        d = root / model
        complete = json.loads((d / "complete.json").read_text())
        if sha256(d / f"{model}_best.pt") != complete["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {model}")
        thr = json.loads((d / "evaluation" / "threshold.json").read_text())
        if thr["external_used_for_selection"]:
            raise ValueError("External data used for selection")
        out[model] = {"dir": d, "checkpoint": d / f"{model}_best.pt", "threshold": thr["validation_selected_threshold"],
                      "sha": complete["checkpoint_sha256"]}
    return out


def paired_bootstrap(frames, groups, threshold, config):
    """One shared cluster resample per replicate across all conditions -> paired CIs for changes vs reference."""
    labels = frames["reference"].label.to_numpy()
    _, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
    probs = {c: f.prob_fake.to_numpy() for c, f in frames.items()}
    rng = np.random.default_rng(config["seed"])
    draws = {c: {"auc": [], "ba": []} for c in frames}
    for _ in range(config["bootstrap_replicates"]):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(labels[idx])) < 2:
            continue
        for c, p in probs.items():
            draws[c]["auc"].append(roc_auc_score(labels[idx], p[idx]))
            draws[c]["ba"].append(balanced_accuracy_score(labels[idx], (p[idx] >= threshold).astype(int)))
    ref = {k: np.asarray(v) for k, v in draws["reference"].items()}
    out = {}
    for c, d in draws.items():
        r = {}
        for k in ("auc", "ba"):
            x = np.asarray(d[k])
            r[f"{k}_ci_lower"], r[f"{k}_ci_upper"] = np.quantile(x, [.025, .975])
            delta = x - ref[k]
            r[f"{k}_change_ci_lower"], r[f"{k}_change_ci_upper"] = np.quantile(delta, [.025, .975])
        r["bootstrap_valid"], r["n_clusters"] = len(draws[c]["auc"]), len(members)
        out[c] = {k: float(v) for k, v in r.items()}
    return out


def run_grid(config, info, experiment, folder, conditions, datasets):
    root = Path(config["output_dir"])
    results = []
    for model, item in info.items():
        for name, (manifest, dataset, split) in datasets.items():
            clean_raw = pd.read_csv(item["dir"] / "predictions" / f"{model}_{name}_predictions.csv")
            clean = rescore(clean_raw, item["threshold"], "source_disjoint_validation_frozen")
            frames = {"reference": clean}
            for condition, transform, parameters in conditions:
                raw = infer(item["checkpoint"], manifest, config,
                            root / folder / "predictions" / f"{model}_{name}_{condition}.csv",
                            dataset, split, PROTOCOL, transform, condition)
                rows = rescore(raw, item["threshold"], "source_disjoint_validation_frozen")
                if set(rows.sample_id) != set(clean.sample_id):
                    raise ValueError("Cohort changed between conditions")
                rows = rows.set_index("sample_id").loc[clean.sample_id].reset_index()
                if not np.array_equal(rows.label, clean.label):
                    raise ValueError("Label order mismatch")
                frames[condition] = rows
            if name == "test":
                groups = add_clusters(clean).cluster_component.to_numpy()
                unit = "source_video_component"
            else:
                groups, unit = np.arange(len(clean)), "image"
            ci = paired_bootstrap(frames, groups, item["threshold"], config)
            reference_auc = roc_auc_score(clean.label, clean.prob_fake)
            q100_auc = None
            params = {"reference": {}} | {c: p for c, _, p in conditions}
            for condition, rows in frames.items():
                metric = measurement(rows, experiment, config)
                metric.update({k: v for k, v in ci[condition].items()})
                metric["auc_ci_lower"], metric["auc_ci_upper"] = ci[condition]["auc_ci_lower"], ci[condition]["auc_ci_upper"]
                metric.update(params[condition])
                metric.update(reference_auc=reference_auc, auc_change=metric["auc"] - reference_auc,
                              relative_auc_change=(metric["auc"] - reference_auc) / reference_auc,
                              ba_change=metric["balanced_accuracy"] - roc_ba(clean),
                              model_display=DISPLAY[model], training_run="source_disjoint_control",
                              checkpoint_sha256=item["sha"], frozen_threshold=item["threshold"],
                              bootstrap_unit=unit)
                metric["condition"] = condition
                results.append(metric)
            if experiment == "jpeg_robustness":
                q100 = next(r for r in results if r["model"] == results[-1]["model"] and r["condition"] == "Q100"
                            and r["test_dataset"] == results[-1]["test_dataset"])
                for r in results[-len(frames):]:
                    if r["condition"].startswith("Q") and r["condition"] != "Q100":
                        r["auc_change_vs_q100"] = r["auc"] - q100["auc"]
    frame = pd.DataFrame(results)
    save_csv(root / folder / "metrics.csv", frame)
    update_master(root, results)
    return frame


def roc_ba(rows):
    return balanced_accuracy_score(rows.label, rows.prediction)


def plot(frame, folder, order, title):
    for dataset, data in frame.groupby("test_dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for model, rows in data.groupby("model"):
            rows = rows.set_index("condition").loc[order]
            x = np.arange(len(order))
            axes[0].errorbar(x, rows.auc, yerr=[rows.auc - rows.auc_ci_lower, rows.auc_ci_upper - rows.auc], marker="o",
                             capsize=2, label=DISPLAY[model])
            axes[1].errorbar(x, rows.auc_change, yerr=[rows.auc_change - rows.auc_change_ci_lower,
                             rows.auc_change_ci_upper - rows.auc_change], marker="o", capsize=2, label=DISPLAY[model])
        for axis, label in zip(axes, ("ROC-AUC (95% CI)", "AUC change vs reference (paired 95% CI)")):
            axis.set_xticks(range(len(order)), order, rotation=35)
            axis.set(ylabel=label, title=f"{title}: {dataset}")
            axis.axhline(0.5 if axis is axes[0] else 0, color="k", ls="--", alpha=.4)
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / f"{dataset}_curves.png", dpi=220)
        plt.close(fig)


def main(which):
    config = configuration()
    root = Path(config["output_dir"])
    info = controls(config)
    protocol = root / "protocol" / "control"
    ext = root / "02_external_bias_audit"
    test = {"test": (protocol / "test.csv", FF, "test")}
    both = test | {"external_normalized": (ext / "normalized.csv", "external_normalized", "external_test")}
    if which in ("jpeg", "all"):
        folder = "04_jpeg_robustness"
        record_run(root / folder, config, [protocol / "test.csv"])
        qualities = config["jpeg_qualities"]
        conds = [(f"Q{q}", partial(jpeg, quality=q), {"jpeg_quality": q}) for q in qualities]
        frame = run_grid(config, info, "jpeg_robustness", folder, conds, test)
        plot(frame, root / folder, ["reference"] + [f"Q{q}" for q in qualities], "JPEG")
    if which in ("frequency", "all"):
        folder = "07_frequency_sensitivity"
        record_run(root / folder, config, [protocol / "test.csv", ext / "normalized.csv"])
        cutoffs = config["frequency"]["cutoffs"]
        conds = [(f"{k}_{c}", partial(frequency_filter, kind=k, cutoff=c), {"filter_type": k, "filter_cutoff": c})
                 for k in ("lowpass", "highpass") for c in cutoffs]
        frame = run_grid(config, info, "frequency_sensitivity", folder, conds, both)
        plot(frame, root / folder, ["reference"] + [c[0] for c in conds], "Frequency filter")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
