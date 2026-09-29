"""Aggregate the eight verified LOMO evaluations: matrix, heatmaps, known-vs-unseen plot, macro summary.

Macro CIs resample the 26 source-video connected components ONCE per replicate and apply the same
resample to all four held-out cohorts and to the matched control, so the macro gap is paired.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments.common import configuration, save_csv, update_master, atomic_text
from src.experiments.manifests import METHODS, PROTOCOL
from src.experiments.matched_control import DISPLAY, add_clusters

MODELS = ("hybrid", "xception")


def load(root):
    parts, checks = [], []
    for method in METHODS:
        for model in MODELS:
            d = root / "05_lomo" / f"lomo_{method}" / model
            metrics = pd.read_csv(d / "evaluation" / "metrics.csv")
            verification = json.loads((d / "evaluation" / "verification.json").read_text())
            if not (verification["checkpoint_sha256_verified"] and verification["manifest_hashes_verified"]
                    and verification["heldout_absent_from_train_and_val"]
                    and verification.get("abs_difference_vs_log", 0) <= 1e-3):
                raise ValueError(f"Unverified LOMO run: {method}/{model}")
            if len(metrics) != 4 or metrics.unseen.sum() != 1 or metrics.isna()[["auc", "balanced_accuracy"]].any().any():
                raise ValueError(f"Incomplete metrics: {method}/{model}")
            parts.append(metrics)
            checks.append({"held_out": method, "model": model, "abs_val_auc_diff_vs_log": verification.get("abs_difference_vs_log"),
                           "frozen_threshold": verification["frozen_threshold"], "lomo_val_auc": verification["recomputed_val_auc"]})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(checks)


def macro_bootstrap(root, config):
    rng_seed, reps = config["seed"], config["bootstrap_replicates"]
    out = []
    for model in MODELS:
        control = add_clusters(pd.read_csv(root / "01_matched_controls" / model / "predictions" / f"{model}_test_predictions.csv"))
        sid = control.sample_id.to_numpy()
        _, inverse = np.unique(control.cluster_component.to_numpy(), return_inverse=True)
        members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
        labels = control.label.to_numpy()
        methods = control.method.to_numpy()
        control_p = control.prob_fake.to_numpy()
        lomo_p = {}
        for method in METHODS:
            frame = pd.read_csv(root / "05_lomo" / f"lomo_{method}" / model / "predictions_test_all.csv").set_index("sample_id").loc[sid]
            lomo_p[method] = frame.prob_fake.to_numpy()
        rng = np.random.default_rng(rng_seed)
        draws = []
        for _ in range(reps):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            row = {}
            try:
                for method in METHODS:
                    keep = idx[(labels[idx] == 0) | (methods[idx] == method)]
                    if len(np.unique(labels[keep])) < 2:
                        raise ValueError
                    row[method] = (roc_auc_score(labels[keep], lomo_p[method][keep]), roc_auc_score(labels[keep], control_p[keep]))
            except ValueError:
                continue
            unseen = np.mean([v[0] for v in row.values()])
            ctrl = np.mean([v[1] for v in row.values()])
            draws.append((unseen, ctrl, ctrl - unseen))
        draws = np.asarray(draws)
        point_unseen, point_ctrl = [], []
        for method in METHODS:
            keep = (labels == 0) | (methods == method)
            point_unseen.append(roc_auc_score(labels[keep], lomo_p[method][keep]))
            point_ctrl.append(roc_auc_score(labels[keep], control_p[keep]))
        for k, name in enumerate(("macro_unseen_auc", "macro_control_auc_same_cohorts", "macro_gap_control_minus_lomo")):
            point = [np.mean(point_unseen), np.mean(point_ctrl), np.mean(point_ctrl) - np.mean(point_unseen)][k]
            low, high = np.quantile(draws[:, k], [.025, .975])
            out.append({"model": model, "statistic": name, "estimate": float(point), "ci_lower": float(low), "ci_upper": float(high),
                        "valid_replicates": len(draws), "n_clusters": len(members)})
    return pd.DataFrame(out)


def figures(dest, frame, unseen, macro):
    # 1) unseen AUC heatmap: held-out manipulation x model
    matrix = unseen.pivot(index="held_out", columns="model", values="auc").reindex(index=METHODS, columns=list(MODELS))
    fig, axis = plt.subplots(figsize=(5, 4.2))
    image = axis.imshow(matrix, vmin=0.3, vmax=1, cmap="viridis")
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        axis.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < .75 else "black")
    axis.set(xticks=range(2), yticks=range(4), xticklabels=[DISPLAY[m] for m in MODELS], yticklabels=METHODS,
             ylabel="Held out during train and val", title="Unseen-manipulation AUC")
    fig.colorbar(image, label="AUC"); fig.tight_layout(); fig.savefig(dest / "unseen_auc_heatmap.png", dpi=220); plt.close(fig)
    # 2) full cross-manipulation matrices per model (diagonal = unseen)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, model in zip(axes, MODELS):
        data = frame[frame.model == model].pivot(index="held_out", columns="test_manipulation", values="auc").reindex(index=METHODS, columns=METHODS)
        image = axis.imshow(data, vmin=0.3, vmax=1, cmap="viridis")
        for (i, j), value in np.ndenumerate(data.to_numpy()):
            axis.text(j, i, f"{value:.2f}" + ("*" if i == j else ""), ha="center", va="center", color="white" if value < .75 else "black")
        axis.set(xticks=range(4), yticks=range(4), xticklabels=METHODS, yticklabels=METHODS, title=f"{DISPLAY[model]} (* = unseen)",
                 ylabel="Held out", xlabel="Test manipulation")
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(dest / "cross_manipulation_heatmaps.png", dpi=220); plt.close(fig)
    # 3) known-vs-unseen with per-cohort CIs: control (known) vs LOMO model on identical held-out cohort
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for axis, model in zip(axes, MODELS):
        rows = unseen[unseen.model == model].set_index("held_out").loc[list(METHODS)]
        x = np.arange(4)
        axis.bar(x - .2, rows.control_auc_same_cohort, .4, label="control (method seen in training)")
        axis.bar(x + .2, rows.auc, .4, label="LOMO model (method unseen)",
                 yerr=[rows.auc - rows.auc_ci_lower, rows.auc_ci_upper - rows.auc], capsize=3)
        axis.axhline(.5, color="k", ls="--", alpha=.4)
        axis.set(xticks=x, xticklabels=METHODS, title=DISPLAY[model], ylim=(0, 1), ylabel="AUC on identical cohort")
        axis.tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Known vs unseen (8 fake videos per method; error bars: source-component bootstrap 95% CI)", fontsize=9)
    fig.tight_layout(); fig.savefig(dest / "known_vs_unseen.png", dpi=220); plt.close(fig)
    # 4) paired gaps
    fig, axis = plt.subplots(figsize=(7.5, 4))
    for k, model in enumerate(MODELS):
        rows = unseen[unseen.model == model].set_index("held_out").loc[list(METHODS)]
        x = np.arange(4) + (k - .5) * .25
        axis.errorbar(x, rows.gap_control_minus_lomo, yerr=[rows.gap_control_minus_lomo - rows.gap_ci_lower,
                      rows.gap_ci_upper - rows.gap_control_minus_lomo], fmt="o", capsize=3, label=DISPLAY[model])
    axis.axhline(0, color="k", ls="--", alpha=.4)
    axis.set(xticks=range(4), xticklabels=METHODS, ylabel="AUC gap: control minus LOMO (paired 95% CI)")
    axis.legend(); fig.tight_layout(); fig.savefig(dest / "paired_gaps.png", dpi=220); plt.close(fig)


def run(config):
    root = Path(config["output_dir"])
    dest = root / "05_lomo" / "summary"
    dest.mkdir(parents=True, exist_ok=True)
    frame, checks = load(root)
    unseen = frame[frame.unseen].copy()
    known = frame[~frame.unseen].groupby(["held_out", "model"]).agg(known_methods_mean_auc=("auc", "mean")).reset_index()
    unseen = unseen.merge(known, on=["held_out", "model"])
    macro = macro_bootstrap(root, config)
    save_csv(dest / "cross_manipulation.csv", frame)
    save_csv(dest / "leave_one_out_summary.csv", unseen)
    save_csv(dest / "macro_summary.csv", macro)
    save_csv(dest / "verification_summary.csv", checks)
    figures(dest, frame, unseen, macro)
    rows = []
    for r in macro[macro.statistic == "macro_unseen_auc"].itertuples():
        rows.append({"experiment": "lomo_macro", "model": r.model, "test_dataset": "ffpp_source_safe", "condition": "lomo_macro_unseen",
                     "protocol_id": PROTOCOL, "evaluation_unit": "frame", "auc": r.estimate, "auc_ci_lower": r.ci_lower,
                     "auc_ci_upper": r.ci_upper, "n_clusters": r.n_clusters, "bootstrap_unit": "source_video_component",
                     "training_run": "source_disjoint_lomo"})
    update_master(root, rows)
    return unseen, macro


if __name__ == "__main__":
    u, m = run(configuration())
    print(u[["held_out", "model", "auc", "auc_ci_lower", "auc_ci_upper", "balanced_accuracy", "control_auc_same_cohort",
             "gap_control_minus_lomo", "gap_ci_lower", "gap_ci_upper", "known_methods_mean_auc"]].round(3).to_string(index=False))
    print(m.round(3).to_string(index=False))
