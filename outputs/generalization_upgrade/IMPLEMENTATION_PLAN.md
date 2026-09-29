# Generalization upgrade: implementation plan

Prepared 2026-09-30 from the actual repository audit. Status: **source-video-disjoint derived protocol approved; Tier 1 and Tier 2 implementation in progress**. User explicitly approved matched CoAtNet/Xception controls, fresh RGB+FFT training, and preserving historical artifacts unchanged. Tier 3 and Tier 4 are NOT authorized for this implementation stage.

Research question: How well do deepfake detectors generalize beyond seen manipulation methods and image conditions, and which forensic representations or training strategies remain reliable under distribution shift?

## 1. Decisions and safety gates

### A. Training protocol approved

The existing target-video split has zero target overlap but shared source/donor identifiers. Preserve it for historical reproduction, with its limitation stated. Do not silently declare it completely identity-safe.

**Recommended new-training protocol:** derive source-video-disjoint subsets while preserving target assignments: exclude a manipulated video when any filename participant belongs to another partition. Keep original files untouched. Feasibility counts are train **16,680**, validation **1,380**, test **1,860** images. This includes only four fake validation videos and eight fake test videos per method, so report wide uncertainty and class imbalance. Person-level disjointness remains unverified.

The user approved this material scientific trade-off. Use the derived filtering protocol, not a source-component-grouped repartition. Merely repeating the original target protocol is useful historically but does not resolve the source-video overlap requirement.

Under the recommended stricter subset, train **two new known-method control models**, one CoAtNet and one Xception, as well as the eight LOMO models. Otherwise comparing a new source-safe LOMO model to an old source-overlapping baseline would mix two experimental changes. Compute each known-versus-unseen gap on the same real + held-out-method test samples. Keep historical-baseline results separate.

### B. Checkpoint eligibility

The old Frequency CNN trained on 1,800 images in the expanded test set. Train a **fresh** Frequency CNN on the selected new training partition; never initialize it with that checkpoint. Keep legacy Frequency CNN and Meso results in a clearly labelled historical appendix. Existing full-data CoAtNet/Xception checkpoints are reusable for historical reproduction and descriptive external/corruption analysis, not as LOMO initialization.

### C. Storage and environment

Only 3.45 GiB was free locally. Small configs, CSVs, plots and feature caches can remain local. Keep large training/VAE caches on the existing Modal volume or another explicitly selected location. No automatic deletions. No broad dependency changes to the working environment; use a tested optional environment for reconstruction. Log environment versions and resolved architecture names.

## 2. Shared architecture and experiment contract

Extend `src/config.py` with a separate `config/generalization_upgrade.yaml`. Add a small `src/experiments/` package for orchestration/provenance, reusing existing train/evaluate/model/data helpers instead of replacing them. Use `notebooks/deepfake_detection_complete.ipynb` as the runnable user-facing entry point, with expensive actions opt-in and cached review available.

Every run saves:

- Run ID/date, seed, exact merged YAML, code revision plus dirty-tree state, environment/device information, resolved model name and checkpoint hash.
- Dataset/manifest hash, split/protocol ID, source/target IDs where justified, sample IDs, preprocessing and feature-layer definitions.
- Device/batch/workers; input resolution; normalization/interpolation; JPEG/blur/resize/filter/patch parameters; augmentation/optimizer/LR/scheduler/epochs and selected checkpoint.
- Threshold and validation source manifest/hash; ensemble weights with their validation provenance. New LOMO models select their own threshold on their **allowed-method validation set**, never held-out-method labels.
- Raw predictions with `sample_id`, actual video stem, target group, optional identity ID only when known, label, manipulation, model, probability, predicted label, threshold, dataset, split and protocol.

The existing `video_id` is often a target group rather than a complete manipulated-video stem; store those as separate fields. Normalize local/cloud paths to data-relative sample keys, not basename alone. Assert unique keys, exact coverage and matching labels before joins.

Cache key: checkpoint hash + input/manifest hash + complete transform config + seed + feature layer. Atomic writes and completion metadata prevent incomplete files being reused. Do not silently trust cached predictions from a different protocol. Keep transformed images on demand or bounded in cache; persist manifests/seeds so they can be regenerated.

`master_results.csv` will have the requested metric columns plus `run_id`, `protocol_id`, `evaluation_unit`, `threshold_source`, `score_type` and status/provenance fields. Register **only real completed measurements**; pending jobs belong in a run-status file, not fabricated metric rows. Summaries/figures must distinguish historical results, reproduced results and new results.

## 3. Phased implementation map

Cost ranges below are planning estimates, not measured future runtimes or monetary quotes. CPU work is separate from GPU time. All paths in the output column are beneath `outputs/generalization_upgrade/`.

| Phase / priority | Task | Files to create or extend | Existing code to reuse | Retraining? | Expected output | Estimated cost | Dependencies / risks |
|---|---|---|---|---|---|---|---|
| 0 / Tier 1, done | Audit, test baseline, inventory and provenance eligibility | `outputs/audit/current_project_audit.md`, this plan, `test_report.txt` | Existing manifests, checkpoints, saved predictions, tests | No | Durable audit and plan | Completed local checks; tests 67.91s | Source overlap, old-model contamination, low disk identified |
| 1 / Tier 1 | Central config, manifest/protocol validator, run registry and canonical IDs | `config/generalization_upgrade.yaml`; `src/experiments/{config,provenance,manifests,run}.py`; tests; targeted `.gitignore` additions | `src/config.py`, dataset/split/portable-manifest utilities | No | Config snapshots, eligibility reports, run metadata | CPU, minutes | Resolve split gate before derived training manifests; ignore restricted derived images/caches |
| 2 / Tier 1 | Fresh checkpoint reproduction and strict saved-prediction comparison | `src/experiments/baseline.py`; notebook baseline cells | `load_checkpoint`, `evaluate_checkpoint`, `binary_method_reports`, video aggregation, ensemble utilities | No | `00_baseline_reproduction/{metrics.csv,predictions.csv,roc_curves.png,confusion_matrices.png,baseline_reproduction.md}` | Roughly 10-30 local GPU minutes, measure first | Version/resize differences; frozen validation thresholds; historical protocol only |
| 3 / Tier 1 | External metadata audit BEFORE normalization, then identical normalization and evaluation | `src/experiments/{external_audit,normalize_external}.py`; extend face detection to expose boxes/landmarks without changing old defaults | Existing raw images, MTCNN, crop helpers, paired manifest, evaluators | No | `02_external_bias_audit/`: metadata, distributions, derived normalized images/manifest, original-vs-normalized table/plot/report | Roughly 10-40 local GPU minutes plus CPU analysis | Source confounding persists; detection attrition; alignment changes crop geometry |
| 4 / Tier 1 | ECE/NLL/Brier, reliability/confidence distributions and threshold transfer | `src/eval/calibration.py`, `src/experiments/calibration.py`, tests | Saved/reproduced probabilities, validation threshold helper | No | `03_calibration/`: metrics, reliability diagrams, confidence distributions, threshold transfer | CPU minutes; bootstrap 5-30 min depending on size | Calibration is not threshold optimization; no external tuning |
| 5 / Tier 1 | Approved source-safe controls, LOMO suite and fresh Frequency CNN | `src/experiments/lomo.py`; derived split YAMLs/manifests; `scripts/modal_generalization.py`; safe run/resume extensions in `src/train.py` | Model registry, train loop, augmentation, Modal volume workflow | **Yes**: 8 LOMO + 2 matched controls + 1 Frequency CNN under recommended protocol | `01_leave_one_manipulation_out/`: predictions/checkpoints/logs, summary, heatmap, gap bars and distributions; matched controls under baseline output | Approximately **2-4 A10 GPU-hours** including overhead contingency | No exposed FF++ initialization; small source-safe validation/test; one initial seed; retain every negative result |
| 6 / Tier 1 | Standardized JPEG for CoAtNet/Xception/fresh Frequency CNN | `src/experiments/{corruptions,jpeg}.py`; transform tests | Legacy robustness reporting/plot conventions, prediction helpers | No beyond phase 5 | `04_jpeg_robustness/`: six-quality predictions/metrics, AUC and degradation plots | Roughly 10-40 GPU minutes | Identical sample IDs and common degradation resolution; historical Q100 is not actual re-encoding |
| 7 / Tier 1 | Group FFT, mean log spectra, differences, radial power and energy ratios | `src/experiments/frequency_analysis.py` | RGB image loading and NumPy FFT; not normalized classifier FFT inputs | No | `06_frequency_analysis/`: sampling manifest, spectra/differences, radial curves, energy CSV | CPU 5-20 min | DC handling, spectrum normalization, matched real references, correlated frames |
| 8 / Tier 1 | Low/high-pass cutoff sensitivity for three models | Extend `corruptions.py`; `src/experiments/frequency_sensitivity.py` | Evaluators, shared selected sample IDs | No beyond phase 5 | `07_frequency_sensitivity/`: predictions, heatmaps, cutoff curves and relative drops | Roughly 15-60 GPU minutes | Filtering/clipping changes signal distribution; document signed high-pass visualization vs classifier inputs |
| 9 / Tier 1 | Cached embeddings, joint t-SNE, PCA variance/effective rank and original-space silhouette | Extend `src/interpret/tsne.py` compatibly; `src/experiments/representation.py` | Existing classifier-input hook and deterministic sampling | No | `09_representation_analysis/`: full feature cache, joint coordinate CSVs, recoloured figures, PCA/rank metrics | 5-15 GPU min plus 5-30 CPU min | Balanced sample construction, high-dimensional distance convention, no causal t-SNE claims |
| 10 / Tier 1 | Systematic known/LOMO/external success/failure CAM panels | Extend `src/interpret/gradcam.py`; `src/experiments/gradcam_analysis.py` | Signed target, layer discovery, saved predictions | No | `10_gradcam_failure_analysis/`: sample list, annotated overlays, failure notes | 5-20 GPU min for bounded panels | Explicit explanation target; missing categories reported; semantic measurements only with masks |
| 11 / Tier 2 | AEROBLADE-style deterministic VAE reconstruction | Optional pinned requirements/cloud image; `src/experiments/reconstruction.py`; orientation/range tests | Existing images/manifests/metrics/provenance helpers | No classifier training | `11_reconstruction_detection/`: MSE/SSIM/LPIPS scores, ROC/distributions, reconstruction/error panels | Roughly 15-45 GPU min after downloads; small batches | Missing diffusers/LPIPS, model access/revision, SD1.5 provenance, local disk; method is VAE analysis, not full paper reproduction |
| 12 / Tier 3 | Controlled BASIC/JPEG/ROBUST CoAtNet ablation | `config/augmentation_*.yaml`; augmentation additions; `src/experiments/augmentation.py` | Same architecture, selected protocol and training loop | **Yes**, up to 3 matched runs | `05_augmentation_ablation/`: clean/JPEG/external table and grouped figure | Approximately 0.5-1.5 A10 GPU-hours | Reuse a phase-5 control only if initialization, split, epochs and all settings match exactly; do not add 12 LOMO runs automatically |
| 13 / Tier 3 | Deterministic 50% patch mask and shuffle | Extend corruption helpers; `src/experiments/spatial_coherence.py` | Existing checkpoints, selected samples and evaluators | No | `08_spatial_coherence/`: examples, predictions, AUC drops | 5-20 GPU min | Per-sample seeds independent of batch order; global/local interpretations remain qualified |
| 14 / Tier 3 | Conventional external deepfake set if genuinely available | Dataset adapter and manifest only as needed | Existing preparation/evaluation pipeline | No | Separate named external output and summary rows | Unknown until data access/size established | No ready conventional dataset verified; do not count GenImage ADM as face swapping |
| 15 / Tier 3 | Optional spatial/frequency feature fusion | `src/experiments/feature_fusion.py`; small head config | CoAtNet features and **spectral branch** of fresh dual-domain model | **Yes**, small MLP only with eligible frozen encoders | `12_optional_feature_fusion/` | Feature extraction + 5-20 GPU min, or CPU head | Existing Frequency CNN is already RGB+FFT; label branch ablation accurately; train/val-only fitting |
| 16 / Tier 4 | Frozen CLIP/DINO linear baseline | Optional adapter/config and `src/experiments/foundation.py` | Dataset, cached feature interface, metrics | **Yes**, linear probe only | `13_optional_foundation_baseline/` | Estimate after available backbone/cache check | No large fine-tune; separate allowed-method probes needed for genuine LOMO |
| 17 / Tier 4 | Optional faithful NPR representation | `src/experiments/npr.py`, method tests, lightweight-head config | Shared training/evaluation infrastructure | **Yes**, lightweight classifier | `14_optional_npr/` | Estimate after official method review | Must verify original research implementation, not label an invented pixel transform NPR |
| 18 / Tier 4 | Optional SBI training intervention | Separate augmentation module/config and tests | Same CoAtNet training protocol | **Yes**, matched control/intervention | `15_optional_sbi/` | At least one extra training run; re-estimate first | Mask/blending correctness; do not expect face blending to solve full-image diffusion |
| Every completed phase / final | Automated tables A-F, summary, notebook and ladder | `src/experiments/report.py`; notebook; `master_results.csv`, `FINAL_EXPERIMENT_SUMMARY.md`, `test_report.txt` | Existing tables/plots, proven new run outputs | No | Report-ready CSV/Markdown and 16 requested figures | CPU/report generation minutes; review time separate | Never replace missing results with hypothetical values or mix protocols |

Tier 4 does not start while Tier 1 is incomplete. Tier 3 is conditional, not an implicit unlimited GPU budget. The plan's core scope is Tier 1 plus reconstruction.

## 4. Experimental definitions to freeze before runs

### Historical reproduction and new controls

Run each full-data checkpoint on original validation, test and external manifests with its own stored preprocessing. Compare sample-level predictions and metrics to saved files. Investigate material differences before using new values; pin the old cloud environment if necessary. Do not force scores to match. Reuse existing validation-derived frame/video ensemble weights for reproduction; log their calibration level separately.

Matched source-safe controls, if approved, are separate runs and separate result rows. New three-model JPEG/filter comparisons use these matched controls plus the fresh Frequency CNN; historical checkpoint curves can also be shown, clearly labelled, without pretending their training protocols match.

### External source audit and normalization

Audit all raw downloaded images, flag membership in the original 576-image cohort, and record dimensions/aspect/format/mode/bytes, JPEG quantization metadata if present, measured brightness/contrast/sharpness/high-frequency energy/RGB statistics. Re-detect to record original box/crop dimensions and face-area ratio. Do not pretend JPEG quality is exactly recoverable from arbitrary files.

Create derived output only. Use one fixed landmark-detection/alignment, crop, RGB, resize/interpolation and **explicit JPEG Q90** pipeline for both classes. This adds genuine alignment absent from the legacy cropper; save its template/warp and state this difference. Test identity alignment and failure handling. Keep the normalization choice fixed before examining its effect on AUC.

Use the original 576-image cohort as the primary source of paired original/normalized comparisons. If detection/alignment fails, show counts by class, save failure reasons, evaluate the original and normalized variants on the **same surviving IDs**, and separately retain the original full-cohort result. Do not change cohort silently or drop failures based on predictions. Source and semantic differences cannot be removed simply by re-encoding.

### Calibration, uncertainty and score behavior

Use fixed equal-width probability bins (e.g. 15) for binary probability ECE; state the exact convention. NLL uses documented clipping for numerical stability; Brier is mean squared probability error. Reliability plots and score histograms cover real/fake by domain; inspect whether fake scores fall, real scores rise, ordering reverses or distributions collapse.

Threshold selection is only from the model's eligible FF++ validation data. Freeze it across clean/JPEG/external conditions. Each LOMO checkpoint uses its own allowed-method validation threshold. No post-hoc external sign flips, temperature selection or test-tuned thresholds.

Bootstrap 95% AUC CIs with fixed seed and recorded resampling count (initially 2,000). Preserve related FF++ frames within video clusters; preferably use source/target components when feasible. External uses sample bootstrap, with pair-group sensitivity analysis if appropriate. Use paired resampling for model differences. LOMO summary is a macro-average over four held-out-method AUCs, not pooled uncalibrated scores from four different models. State how shared real videos are handled in the aggregate bootstrap. No tiny-AUC-difference significance claims without uncertainty.

### LOMO

For each of Deepfakes, Face2Face, FaceSwap and NeuralTextures: exclude that manipulation from **both train and validation**; retain real data; evaluate real + held-out fake in test. Train CoAtNet and Xception from generic pretrained weights only. Same seed and schedule across matched conditions; no checkpoint exposed to excluded-method training data.

For the requested full heatmap, also evaluate each LOMO checkpoint on the other test methods as supplementary columns; clearly mark only the held-out diagonal as unseen. Compare known control versus LOMO on identical per-method test samples. Do not use any test column to select models. Source-safe subset counts and low effective sample sizes remain visible.

### Standardized corruptions and frequency analysis

Use common 224 x 224 RGB working images for degradation, then model-specific resizing/normalization. Explicitly re-encode Q100/Q90/Q60/Q40/Q20/Q10 with a fixed implementation; keep a separate no-additional-JPEG reference. The old Q100 curves remain historical and are not relabelled lossless. Same sample IDs at each quality/model; thresholds frozen. Absolute AUC drop is reference minus condition; relative drop divides by reference AUC.

FFT analysis uses consistent unnormalized grayscale input and fixed sampling counts per group/video. Save the sampling manifest. Define mean subtraction/windowing and DC treatment once; default quantitative non-DC energy excludes DC. Show average log magnitude, power-derived radial curves and low/mid/high energy ratios with fixed band boundaries. FF++ fake references use FF++ real; diffusion uses normalized external real. Colour scales are comparable and difference maps use a zero-centred scale.

For sensitivity, define radius using `fftfreq`: radial frequency divided by axis Nyquist (0.5 cycles/pixel), so corner radius can exceed 1. Ideal low-pass masks retain radius <= cutoff; high-pass is the complementary mask, cutoffs 0.1/0.3/0.5. Inverse FFT is taken per RGB channel. Preserve signed high-pass residuals for measurement; explicitly define the fixed display/classifier offset and clipping, never per-image min-max normalize. Test constant images and sinusoidal frequencies. Spectral sensitivity shows dependence under intervention, not a universal artifact or causal mechanism.

Patch masking/shuffling uses a fixed grid divisible into the working resolution, approximately 50% neutral-valued patches, and deterministic sample-ID-derived seeds. Preserve shuffled patch contents exactly; do not add invisible resizing differences.

### Representations and Grad-CAM

Save classifier-input features before projection, sample metadata, layer name and shape. Fit one joint t-SNE and reuse coordinates for binary, manipulation and domain colourings. Record seed/perplexity/iterations and any standardization/PCA preprocessing. Compute variance thresholds 90/95/99% from a separately specified full PCA, not the already-truncated 50-component t-SNE input. Define effective rank explicitly (e.g. exponential entropy of normalized covariance eigenvalues). Silhouette uses original high-dimensional embeddings with the scaling/distance convention documented, not 2D coordinates.

Grad-CAM cases are deterministically selected across distinct videos where possible from TP/TN/FP/FN for known, LOMO and normalized external data. Explain fake-logit evidence consistently, or show predicted/true targets in separately labelled panels. Missing categories are reported, not synthesized. Qualitative rough-region descriptions must not masquerade as measured landmark-mask statistics.

### Reconstruction

Verify official AEROBLADE research/code and the chosen VAE model documentation before implementation. Use a compatible, revision-pinned Stable Diffusion VAE if provenance can be established; otherwise label compatibility as an assumption. No DDIM inversion or image generation is needed.

Resize with a declared policy to VAE-compatible dimensions, convert to the model's documented range, use deterministic posterior **mode**, decode correctly, and compare in a common RGB range. Be careful about model-specific latent scaling. Use float32 initially and small batches; optimize only after correctness checks.

For MSE and LPIPS, define fake score = **negative reconstruction error** in advance. For SSIM similarity, fake score = SSIM (equivalently negative `1 - SSIM` error). Never choose a sign to improve external AUC. A lower-error-generated hypothesis can fail.

Evaluate external normalized and a fixed small method-balanced FF++ test subset. Save scores, model revision, reconstructions, error maps, distributions, ROC and CIs. These are anomaly scores, **not probabilities**: ECE/NLL and threshold-dependent metrics are N/A unless a separate defensible FF++-validation-only score calibration is explicitly implemented. Do not fit external labels to manufacture calibration. Describe this as an AEROBLADE-style VAE experiment, not full reproduction of every paper detail.

## 5. Reporting deliverables and completion criteria

Tables generated from `master_results.csv`, each as CSV and Markdown:

- A: model baseline frame/video AUC, accuracy, balanced accuracy, precision/recall/F1/ECE/NLL, with protocol and checkpoint eligibility.
- B: per-manipulation real-versus-fake performance.
- C: model x held-out-manipulation LOMO results, uncertainty and known-to-unseen gaps.
- D: model x six JPEG qualities, absolute/relative degradation.
- E: FF++ / external original / external normalized AUC, gap, external ECE/NLL, matched cohort size.
- F: spatial, RGB+FFT, ensemble, reconstruction and completed optional methods; distinguish true spectral-only ablations, and use N/A where metrics do not apply.

Figures: (1) pipeline/generalization ladder; (2) baseline ROC; (3) per-method performance; (4) LOMO heatmap; (5) JPEG curves; (6) original/normalized external comparison; (7) score distributions; (8) reliability diagrams; (9) spectra/differences; (10) filtering sensitivity; (11) binary t-SNE; (12) method/domain t-SNE; (13) PCA variance; (14) CAM success/failures; (15) reconstruction/error examples; (16) final four-level ladder.

Use high-resolution images and vector output where practical, consistent model labels and honest axes. The ladder connects known FF++, unseen manipulation, prespecified JPEG condition (e.g. Q20) and normalized external AUC. These categorical shifts are not guaranteed to be monotonically harder; do not force a decreasing line or hide counterexamples. Use N/A for unrun Frequency CNN LOMO, and do not force reconstruction into inapplicable conditions.

`FINAL_EXPERIMENT_SUMMARY.md` is updated after each completed phase with research question, method/data/models, actual metrics, figure paths, evidence-supported interpretation, limitations, unexpected results and hypothesis support/contradiction. Pending phases stay explicitly pending. The notebook loads verified outputs and exposes reproducible run cells instead of hard-coded headline numbers.

## 6. Verification strategy

Baseline: 37 tests pass before edits. Add small synthetic/unit tests, without downloading models in normal CI, for:

- Target/source overlap and checkpoint training/evaluation eligibility; canonical local/cloud IDs; exact ensemble coverage.
- LOMO train/val exclusion and fixed split assignments; generic pretrained initialization provenance.
- Validation-only threshold selection, immutable transfer, frame/video calibration separation.
- ECE bins/edge probabilities, NLL/Brier known cases; bootstrap grouping and deterministic seeds.
- JPEG shape/type/range and actual Q100 encoding; sinusoidal low/high-pass behavior; deterministic patch transforms independent of loader batching.
- Identical normalization for both classes, geometric alignment, detection failures and matched-cohort accounting.
- Reconstruction preprocessing/range/latent handling and score orientation with a stub VAE; probabilities versus non-probability scores.
- Prediction serialization/round-trip, cache invalidation, atomic/resumable output handling; feature dimensionality and joint projection reuse.

After each phase run targeted tests, then the full suite when shared behavior changes. Save commands, versions, counts and warnings in `test_report.txt`. Run a small GPU smoke test before full experiments. Never select scientific hyperparameters from test smoke-test performance. Save changes cleanly; if committing, stage only task-owned files and do not capture the pre-existing dirty worktree indiscriminately.

## 7. Compute and storage budget

Observed old A10 training+validation totals: CoAtNet 8.16 minutes / eight epochs; Xception 17.76 minutes / twelve epochs. Four full-protocol LOMO runs per architecture, at approximately 80% of original data, suggest about **1.38 A10-hours** of epoch work before setup/evaluation. That is only an extrapolation. Two matched source-safe controls and a fresh Frequency CNN add work; source filtering also reduces row counts. Reserve **2-4 A10 GPU-hours** for this core training package, then adjust from one measured run per architecture.

Core inference grids, CAM/feature extraction and VAE reconstruction: provisionally **1-3 GPU-hours**, plus CPU plotting/bootstrapping. Thus Tier 1 + Tier 2 is provisionally **3-7 GPU-hours** overall, not a completion-time promise. Cloud image build, downloads, queueing, debugging and human review add elapsed time. Optional augmentation adds roughly 0.5-1.5 A10 GPU-hours; other optional methods require separate estimates before launch.

No dollar estimate or credit balance is claimed: current pricing/account balance has not been checked. Before paid runs, inspect the already-authorized Modal account, confirm the intended workspace/volume, report an updated estimate and use bounded timeouts/run counts. Do not start a broad hyperparameter search.

Do not download a full diffusion pipeline when only VAE weights are needed. Cache only the selected revision and bounded reconstructions; expect hundreds of MB for a VAE and potentially GB-scale dependency/cache overhead. Eleven new model checkpoints plus resumable optimizer states can exhaust the remaining local disk. Keep those remotely; download final small reports first. New restricted image outputs under `outputs/generalization_upgrade` need explicit ignore rules before creation.

## 8. Immediate next steps

1. Return this plan and the audit, as requested, before implementation/training.
2. Confirm the source-video-safe protocol trade-off; this blocks **new training only**, not safe baseline/configuration work.
3. Implement shared provenance/configuration and fresh historical baseline reproduction.
4. Execute external bias audit, controlled normalization and calibration.
5. Proceed through LOMO and remaining Tier 1 experiments, then reconstruction, with measured resource updates and tests after every phase.

No result-quality improvement is promised. Completion means a reproducible, defensible investigation with genuine numerical results, including failures and uncertainty—not merely a larger reported accuracy.
