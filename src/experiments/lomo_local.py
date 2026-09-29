"""Local evaluation of completed leave-one-manipulation-out (LOMO) checkpoints.

Each run: verify hash + manifest hashes + held-out absence from train AND val, verify validation AUC
against the training log, freeze a threshold from LOMO validation only, then score the shared
source-disjoint test manifest (real + all four manipulations). The held-out cohort is real + held-out method.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments.calibration import select_threshold
from src.experiments.common import configuration, record_run, save_csv, save_json, sha256, update_master
from src.experiments.evaluation import infer, measurement
from src.experiments.manifests import METHODS, PROTOCOL
from src.experiments.matched_control import DISPLAY, FF, add_clusters, cluster_ci, rescore


def verify(run_dir, root, method, model):
    complete = json.loads((run_dir / "complete.json").read_text())
    checkpoint = run_dir / f"{model}_best.pt"
    if sha256(checkpoint) != complete["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    protocol = root / "protocol" / f"lomo_{method}"
    for split in ("train", "val", "test"):
        if sha256(protocol / f"{split}.csv") != complete["config"]["experiment"]["manifest_hashes"][split]:
            raise ValueError(f"Manifest hash mismatch: {split}")
    frames = {s: pd.read_csv(protocol / f"{s}.csv") for s in ("train", "val")}
    for split, frame in frames.items():
        if method in set(frame.method):
            raise ValueError(f"Held-out {method} present in {split}")
    log = pd.read_csv(run_dir / f"{model}_epochs.csv")
    return complete, checkpoint, protocol, log, {s: sorted(set(f.method)) for s, f in frames.items()}


def paired_gap_ci(control, lomo, config):
    """Control-minus-LOMO AUC on the identical held-out cohort, shared cluster resamples."""
    labels = control.label.to_numpy()
    _, inverse = np.unique(control.cluster_component.to_numpy(), return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
    rng = np.random.default_rng(config["seed"])
    gaps = []
    for _ in range(config["bootstrap_replicates"]):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(labels[idx])) == 2:
            gaps.append(roc_auc_score(labels[idx], control.prob_fake.to_numpy()[idx])
                        - roc_auc_score(labels[idx], lomo.prob_fake.to_numpy()[idx]))
    low, high = np.quantile(gaps, [.025, .975])
    return float(low), float(high)


def evaluate(method, model, config):
    root = Path(config["output_dir"])
    run_dir = root / "05_lomo" / f"lomo_{method}" / model
    complete, checkpoint, protocol, log, methods = verify(run_dir, root, method, model)
    record_run(run_dir / "evaluation", config, [checkpoint, protocol / "val.csv", protocol / "test.csv"])
    control_test = root / "protocol" / "control" / "test.csv"
    raw_val = infer(checkpoint, protocol / "val.csv", config, run_dir / "evaluation" / "val_raw.csv", FF, "val",
                    PROTOCOL, condition=f"lomo_{method}")
    val_auc = float(roc_auc_score(raw_val.label, raw_val.prob_fake))
    # The saved checkpoint is the best-validation-AUC epoch, so compare with that epoch only. Tolerance 1e-3
    # covers Modal-A10 vs local-GPU float differences (observed ~4e-5) yet is 10x below epoch-to-epoch spread.
    logged_best = float(log.val_auc.max())
    auc_difference = abs(val_auc - logged_best)
    if auc_difference > 1e-3:
        raise ValueError(f"Validation AUC {val_auc:.6f} differs from best logged epoch {logged_best:.6f}")
    threshold = select_threshold(raw_val)
    raw_test = infer(checkpoint, control_test, config, run_dir / "evaluation" / "test_all_raw.csv", FF, "test",
                     PROTOCOL, condition=f"lomo_{method}")
    test = add_clusters(rescore(raw_test, threshold, "lomo_validation_frozen"))
    save_csv(run_dir / "predictions_test_all.csv", test)
    control = pd.read_csv(root / "01_matched_controls" / model / "predictions" / f"{model}_test_predictions.csv")
    control = control.set_index("sample_id").loc[test.sample_id].reset_index()
    control = add_clusters(control)
    rows, table = [], []
    for column in METHODS:
        mask = ((test.label == 0) | (test.method == column)).to_numpy()
        subset, ctrl = test[mask].reset_index(drop=True), control[mask].reset_index(drop=True)
        unseen = column == method
        metric = measurement(subset, "lomo", config)
        metric.update(cluster_ci(subset, "cluster_component", config, ("auc", "balanced_accuracy")))
        metric.update({"auc_ci_lower": metric.pop("auc_ci_lower"), "auc_ci_upper": metric.pop("auc_ci_upper")})
        metric.update(condition=f"lomo_{method}", test_manipulation=column, held_out=method, unseen=unseen,
                      control_auc_same_cohort=float(roc_auc_score(ctrl.label, ctrl.prob_fake)),
                      model_display=DISPLAY[model], training_run="source_disjoint_lomo",
                      checkpoint_sha256=complete["checkpoint_sha256"], frozen_threshold=threshold,
                      lomo_val_auc=val_auc, bootstrap_unit="source_video_component")
        metric["gap_control_minus_lomo"] = metric["control_auc_same_cohort"] - metric["auc"]
        if unseen:
            metric["gap_ci_lower"], metric["gap_ci_upper"] = paired_gap_ci(ctrl, subset, config)
        table.append(metric)
    save_csv(run_dir / "evaluation" / "metrics.csv", pd.DataFrame(table))
    update_master(root, table)
    fig, axis = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 26)
    axis.hist(test[test.label == 0].prob_fake, bins, density=True, histtype="step", label="real")
    for column in METHODS:
        axis.hist(test[test.method == column].prob_fake, bins, density=True, histtype="step",
                  label=column + (" (held out)" if column == method else ""), ls="--" if column == method else "-")
    axis.axvline(threshold, color="k", ls=":")
    axis.set(xlabel="P(fake)", ylabel="density", title=f"{DISPLAY[model]} / hold out {method}")
    axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(run_dir / "confidence_distribution.png", dpi=200)
    plt.close(fig)
    save_json(run_dir / "evaluation" / "verification.json", {
        "checkpoint_sha256_verified": True, "manifest_hashes_verified": True, "train_methods": methods["train"],
        "val_methods": methods["val"], "heldout_absent_from_train_and_val": True, "recomputed_val_auc": val_auc,
        "training_log_best_val_auc": logged_best, "abs_difference_vs_log": auc_difference,
        "tolerance": 1e-3, "frozen_threshold": threshold})
    return pd.DataFrame(table)


if __name__ == "__main__":
    cfg = configuration()
    out = evaluate(sys.argv[1], sys.argv[2], cfg)
    print(out[["test_manipulation", "unseen", "auc", "auc_ci_lower", "auc_ci_upper", "balanced_accuracy",
               "control_auc_same_cohort", "gap_control_minus_lomo"]].to_string())
