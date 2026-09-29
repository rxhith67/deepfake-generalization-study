# RESUME_STATUS (updated 2026-09-30, verified from artifacts)

**STATUS: Tier 1 and Tier 2 are FROZEN as the primary results. Tier 3/4 not run; no further experiments pending.** Final integrity pass done: `INTEGRITY_REPORT.md` (83 headline AUCs recomputed from raw predictions/scores, 0 mismatches; CI units, protocol separation, reconstruction-score handling and phrase scan checked), `FINAL_EXPERIMENT_SUMMARY.md` (regenerated), `FINAL_EXPERIMENT_INVENTORY.md` (question -> dataset/model/metric/result/artifact/limitation map). Summary fixes in the pass: added the missing spectrum-analysis section; moved the RGB+FFT paragraph next to the controls; stated CI units in every section; added the "not a general diffusion detector / uncalibrated anomaly score" caveat; softened the RGB+FFT low-rank wording to non-causal.

Modal profile rxhith67, volume `deepfake-detection`. Suite app `ap-JlvVtTo0YswgxV83wXOOrG` is STOPPED after finishing all 11 jobs (3 controls + 8 LOMO); nothing was launched/stopped/modified by the recovery sessions. Local: ~7 GB disk, RAM tight (~1-2 GB free) -> one heavy local process at a time.
Naming: "hybrid" = CoAtNet; "freq_cnn" = RGB+FFT dual-domain CNN. New source-video-disjoint results are kept separate from historical target-video-disjoint results (different experiment/test_dataset/protocol_id in master_results.csv; historical rows untouched).

| Experiment | Status | Evidence | Output | Required Action |
|---|---|---|---|---|
| Protocol, baseline reproduction, external bias audit, historical calibration, spectrum analysis | COMPLETE | verified earlier | 00_, 02_, 03_, 06_, protocol/ | none |
| Matched controls x3 (train + test inference + calibration + CIs) | COMPLETE | hashes verified; val AUCs reproduce logs; test AUC 0.846 / 0.866 / 0.569 (RGB+FFT CI 0.421-0.736 includes 0.5) | 01_matched_controls | none |
| Standardized JPEG | COMPLETE | 21 metric rows, 18 new prediction files (3 reference rows reuse clean control predictions), no NaN/dup | 04_jpeg_robustness | none |
| Low/high-pass sensitivity | COMPLETE | 42 rows, 36 prediction files, hashes/counts verified | 07_frequency_sensitivity | none |
| LOMO x8 (both models x 4 held-out methods) | COMPLETE | each run: SHA-256, manifest hashes, held-out absent from train+val, val AUC within 1e-3 of best logged epoch (observed <=3.7e-5); 32 lomo rows + 2 lomo_macro rows in master | 05_lomo, 05_lomo/summary | none |
| LOMO matrix, heatmaps, known-vs-unseen plot, paired gaps, macro summary | COMPLETE | macro unseen AUC CoAtNet 0.535 (0.394-0.643), Xception 0.578 (0.454-0.679); 26 clusters; 8 fake videos per method | 05_lomo/summary | none |
| Representation analysis (PCA, t-SNE, silhouette in original feature space, effective rank) | COMPLETE for the 3 controls (exploratory) | representation_metrics.json per model | 09_representation_analysis | optional: LOMO-model embeddings |
| Grad-CAM failure analysis | COMPLETE for Xception (normal FF++, 4 LOMO folds with fold checkpoints, normalized external); qualitative only | 6 panels, 2 TP/TN/FP/FN each; LOMO checkpoint hash verified in run.json | 10_gradcam_failure_analysis/xception | optional: CoAtNet/RGB+FFT CAMs |
| Tier 2 provenance/compatibility check | COMPLETE | dataset card + paper confirm "Stable Diffusion V1.5" (512x512); exact checkpoint and VAE NOT established; SD1.5 `vae/` pinned at 451f4fe1... used as an explicit compatibility ASSUMPTION | 11_reconstruction_detection/PROVENANCE_AND_PROTOCOL.md, model_revision.json | none |
| Tier 2 smoke test (16 images) | COMPLETE (passed) | shape/range, finite, bit-identical repeat, class-blind input | 11_reconstruction_detection/smoke | none |
| Tier 2 external normalized (576) | COMPLETE | 170 master rows (152 unchanged + 18 reconstruction); AUC (pre-registered orientation, uninverted): -MSE 0.352 (0.306-0.396), +SSIM 0.442 (0.394-0.489), -LPIPS 0.681 (0.636-0.723) | 11_reconstruction_detection/external | none |
| Tier 2 FF++ secondary (600 frames) | COMPLETE | pooled AUCs 0.634 / 0.607 / 0.550, all CIs include 0.5 | 11_reconstruction_detection/ffpp | none |
| Tier 3 / Tier 4 | DISABLED | - | - | do not run |
| Tests | COMPLETE | 53 passed | tests/ | re-run after code changes |

## Tier 2 environment note
`diffusers==0.40.0` and `lpips==0.1.4` were installed with `--no-deps --target .tier2_pkgs/` (untracked, 57 MB; do not commit). The main environment's `pip freeze` was verified unchanged. `src/experiments/reconstruction.py` appends that folder to `sys.path`. LPIPS is the standard AlexNet variant, not the AEROBLADE paper's VGG layer-2 variant. Confounds recorded before results: resolution/upsampling history, alignment padding bands, an MTCNN non-face "real" crop, JPEG history.

## Incidents (execution only; no completed output invalidated)
- Filtering run crashed once (numpy MemoryError) when overlapped with LOMO inference (which hit CUDA OOM); rerun alone reused cached predictions.
- LOMO validation-AUC check initially too strict (1e-6); now best-epoch match within 1e-3 with the difference recorded (observed max 3.7e-5).
- Grad-CAM: first driver used the control checkpoint for LOMO panels (bug); those four panels were rebuilt using each fold's LOMO checkpoint and provenance verified.
- Two helper scripts (report.py patches) had escaping errors during editing and were fixed; outputs regenerated from source.
