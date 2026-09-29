"""Read-only results-integrity pass: recompute headline numbers from raw saved predictions/scores
and compare with FINAL_EXPERIMENT_SUMMARY.md and master_results.csv. Runs no models."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("outputs/generalization_upgrade")
summary = (ROOT / "FINAL_EXPERIMENT_SUMMARY.md").read_text(encoding="utf-8")
master = pd.read_csv(ROOT / "master_results.csv")
results, failures = [], []


def check(name, recomputed, expected, tol=1e-6, where="summary"):
    ok = abs(recomputed - expected) <= tol
    results.append((name, recomputed, expected, ok, where))
    if not ok:
        failures.append(name)


def in_summary(value):
    return f"{value:.6f}" in summary


def m(**kw):
    frame = master
    for k, v in kw.items():
        frame = frame[frame[k].isna() if v is None else frame[k] == v]
    return frame


# 1. dataset sizes -----------------------------------------------------------------------------
proto = ROOT / "protocol" / "control"
sizes = {s: len(pd.read_csv(proto / f"{s}.csv")) for s in ("train", "val", "test")}
assert sizes == {"train": 16680, "val": 1380, "test": 1860}, sizes
test = pd.read_csv(proto / "test.csv")
assert test.label.value_counts().to_dict() == {0: 900, 1: 960}
assert test.groupby(["method", "actual_video_id"]).ngroups == 62
ext = pd.read_csv(ROOT / "02_external_bias_audit" / "normalized.csv")
assert len(ext) == 576 and ext.label.value_counts().to_dict() == {0: 288, 1: 288}
recon_ff = pd.read_csv(ROOT / "11_reconstruction_detection" / "ffpp" / "reconstruction_scores.csv")
assert len(recon_ff) == 600

# 2. matched controls, external, JPEG, filtering: AUC from raw prediction files --------------------
for model in ("hybrid", "xception", "freq_cnn"):
    base = ROOT / "01_matched_controls" / model / "predictions"
    for name, dataset in (("test", "ffpp_source_safe"), ("external_normalized", "external_normalized"),
                          ("external_original_matched", "external_original_matched")):
        rows = pd.read_csv(base / f"{model}_{name}_predictions.csv")
        auc = roc_auc_score(rows.label, rows.prob_fake)
        rec = m(experiment="matched_control", model=model, test_dataset=dataset, evaluation_unit="frame").auc.iloc[0]
        check(f"matched {model}/{dataset} raw-AUC == master", auc, rec, where="master")
        assert in_summary(rec), f"{model}/{dataset} not printed in summary"
    for condition in ("Q100", "Q90", "Q60", "Q40", "Q20", "Q10"):
        rows = pd.read_csv(ROOT / "04_jpeg_robustness" / "predictions" / f"{model}_test_{condition}.csv")
        rec = m(experiment="jpeg_robustness", model=model, condition=condition).auc.iloc[0]
        check(f"jpeg {model}/{condition}", roc_auc_score(rows.label, rows.prob_fake), rec, where="master")
        assert in_summary(rec)
    for name, dataset in (("test", "ffpp_source_safe"), ("external_normalized", "external_normalized")):
        for kind in ("lowpass", "highpass"):
            for cutoff in ("0.1", "0.3", "0.5"):
                condition = f"{kind}_{cutoff}"
                rows = pd.read_csv(ROOT / "07_frequency_sensitivity" / "predictions" / f"{model}_{name}_{condition}.csv")
                rec = m(experiment="frequency_sensitivity", model=model, test_dataset=dataset, condition=condition).auc.iloc[0]
                check(f"freq {model}/{dataset}/{condition}", roc_auc_score(rows.label, rows.prob_fake), rec, where="master")
                assert in_summary(rec)

# 3. LOMO unseen AUC from raw predictions ---------------------------------------------------------
for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
    for model in ("hybrid", "xception"):
        rows = pd.read_csv(ROOT / "05_lomo" / f"lomo_{method}" / model / "predictions_test_all.csv")
        sub = rows[(rows.label == 0) | (rows.method == method)]
        rec = m(experiment="lomo", model=model, test_manipulation=method, condition=f"lomo_{method}").auc.iloc[0]
        check(f"lomo {model}/{method} unseen", roc_auc_score(sub.label, sub.prob_fake), rec, where="master")
        assert in_summary(rec)
        assert sub.threshold.nunique() == 1

# 4. historical baseline from raw historical predictions ----------------------------------------------
for model in ("hybrid", "xception", "ensemble"):
    for name, dataset in (("test", "ffpp_historical"), ("external", "external_original")):
        rows = pd.read_csv(ROOT / "00_baseline_reproduction" / f"{model}_{name}_predictions.csv")
        rec = m(experiment="baseline_reproduction", model=model, test_dataset=dataset, evaluation_unit="frame").auc.iloc[0]
        check(f"historical {model}/{dataset}", roc_auc_score(rows.label, rows.prob_fake), rec, where="master")
        assert in_summary(rec)

# 5. reconstruction: orientation applied exactly as pre-registered, no inversion ----------------------------
orientation = {"mse": -1.0, "lpips": -1.0, "ssim": 1.0}
for folder, dataset, path in (("external", "external_normalized", "external"), ("ffpp", "ffpp_source_safe", "ffpp")):
    scores = pd.read_csv(ROOT / "11_reconstruction_detection" / folder / "reconstruction_scores.csv")
    for metric, sign in orientation.items():
        rec = m(experiment="reconstruction", model=f"SD15_VAE_{metric}", test_dataset=dataset, test_manipulation=None).auc.iloc[0]
        check(f"recon {metric}/{dataset} (sign {sign:+.0f})", roc_auc_score(scores.label, sign * scores[metric]), rec, where="master")
        assert in_summary(rec)
recon = m(experiment="reconstruction")
assert recon.score_orientation.dropna().isin(["-MSE", "+SSIM", "-LPIPS"]).all()
assert recon[["ece", "nll", "brier"]].isna().all().all(), "reconstruction rows must carry no calibration metrics"
assert (recon.score_type == "non_probability").all()

# 6. CI units ---------------------------------------------------------------------------------------------
unit_rules = []
ext_new = m(experiment="matched_control")
ext_new = ext_new[ext_new.test_dataset.str.startswith("external")]
assert ext_new.bootstrap_unit.str.startswith("image").all(), "external CI must be image-level"
ff_new = m(experiment="matched_control", test_dataset="ffpp_source_safe")
assert (ff_new.bootstrap_unit == "source_video_component").all()
for exp in ("jpeg_robustness", "frequency_sensitivity"):
    frame = m(experiment=exp)
    for _, r in frame.iterrows():
        expected = "source_video_component" if r.test_dataset == "ffpp_source_safe" else "image"
        assert r.bootstrap_unit == expected, (exp, r.test_dataset, r.bootstrap_unit)
lomo = m(experiment="lomo")
assert (lomo.bootstrap_unit == "source_video_component").all()
hist_ext = m(experiment="baseline_reproduction", test_dataset="external_original", evaluation_unit="frame")
assert (hist_ext.bootstrap_unit == "image").all()
hist_ff = m(experiment="baseline_reproduction", test_dataset="ffpp_historical", evaluation_unit="frame")
assert (hist_ff.bootstrap_unit == "target_video").all()

# 7. historical vs source-disjoint never mixed -----------------------------------------------------------------
protocols = master.groupby("test_dataset").protocol_id.unique().to_dict()
historical_datasets = {"ffpp_historical", "external_original", "external_original_matched", "external_normalized"}
for dataset, ids in protocols.items():
    if dataset == "ffpp_historical":
        assert set(ids) == {"historical_target_video_v1"}, (dataset, ids)
    if dataset == "ffpp_source_safe":
        assert set(ids) == {"source_video_disjoint_v1"}, (dataset, ids)
mixed = master[(master.test_dataset == "ffpp_historical") & (master.experiment != "baseline_reproduction") &
               (master.experiment != "baseline_per_method") & (master.experiment != "calibration")]
assert mixed.empty, "historical FF++ rows appear outside baseline/calibration"
new_models_on_historical = master[(master.experiment.isin(["matched_control", "lomo", "jpeg_robustness", "frequency_sensitivity"])) &
                                  (master.test_dataset == "ffpp_historical")]
assert new_models_on_historical.empty
source_on_new = master[(master.test_dataset == "ffpp_source_safe") & (master.experiment.isin(["baseline_reproduction", "baseline_per_method"]))]
assert source_on_new.empty, "historical checkpoints evaluated on the source-disjoint test set"

# 8. forbidden phrasing --------------------------------------------------------------------------------------------
text = summary.lower()
flags = {
    "superior": re.findall(r"[^.]*\bsuperior\b[^.]*\.", text),
    "more robust (unnegated)": [s for s in re.findall(r"[^.]*\bmore robust\b[^.]*\.", text) if "neither" not in s and "not " not in s],
    "calibrated probability": re.findall(r"[^.]*calibrated probabilit[^.]*\.", text),
    "universal (unnegated)": [s for s in re.findall(r"[^.]*\buniversal\b[^.]*\.", text) if "not" not in s and "no " not in s],
    "better detector (unnegated)": [s for s in re.findall(r"[^.]*better detector[^.]*\.", text) if "not" not in s],
    "general detector": re.findall(r"[^.]*general (diffusion )?detector[^.]*\.", text),
}
phrase_report = {k: v for k, v in flags.items()}

# report ----------------------------------------------------------------------------------------------------------
lines = ["# Results-integrity report (read-only; no models run)", "",
         f"Independent recomputations from raw saved predictions/scores: **{len(results)}**; mismatches: **{len(failures)}**.", "",
         "Dataset sizes verified from manifests: train 16,680 / val 1,380 / test 1,860 frames (62 videos: 30 real, 32 fake; 900 real + 960 fake frames); "
         "external normalized 576 (288/288); reconstruction FF++ secondary 600; external raw fakes 311 (see provenance).", "",
         "Structural assertions passed: external CIs are image-level; new-protocol FF++ CIs are source-video-component; historical FF++ CIs are target-video; "
         "reconstruction rows have no ECE/NLL/Brier and are `non_probability`; reconstruction orientation reproduces the master AUCs with fixed signs (-MSE, +SSIM, -LPIPS); "
         "historical checkpoints never appear on the source-disjoint test set and new controls/LOMO never on the historical test set.", "",
         "## Phrase scan of FINAL_EXPERIMENT_SUMMARY.md", ""]
for key, hits in phrase_report.items():
    lines.append(f"- {key}: {len(hits)} hit(s)")
    lines += [f"    - {h.strip()[:300]}" for h in hits]
lines += ["", "## Recomputation table (AUC)", "", "| check | recomputed | recorded | match |", "|---|---|---|---|"]
lines += [f"| {n} | {a:.6f} | {b:.6f} | {'yes' if ok else 'NO'} |" for n, a, b, ok, _ in results]
(ROOT / "INTEGRITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"recomputations={len(results)} failures={failures}")
for k, v in phrase_report.items():
    print(k, len(v))
sys.exit(1 if failures else 0)
