# Current project audit

Audit date: 2026-09-30. Scope: existing repository, local data, split metadata, checkpoint metadata, saved results, notebook, tests, dependencies and hardware. No training, downloads, installations or remote jobs were started. Existing source, datasets, checkpoints and results were not changed.

## 1. Executive finding

This is an implemented FF++ deepfake detection project, not a blank starting point. The full dataset, two strong trained checkpoints, validation/test predictions, external predictions, JPEG evaluations and interpretability utilities are reusable. The requested generalization study is **not yet implemented**.

The important findings before extension are:

1. All **37 existing tests pass**, with one non-fatal CPU-core detection warning.
2. All **30,000 FF++ manifest images and 576 external manifest images exist** locally. There are 1,000 raw FF++ c23 videos.
3. Recalculating AUC from saved predictions agrees with the reported results. This is a **cached-prediction consistency check**, not fresh checkpoint inference.
4. Existing splits are target-video-disjoint, but **not disjoint across all source/donor video identifiers**. Do not call them comprehensively person-identity-safe.
5. The old Frequency CNN training split overlaps the current test split by **1,800 exact images / 12 target groups**. Reusing that checkpoint on the expanded test is invalid as a clean held-out comparison.
6. Face preprocessing performs detection, bounding-box cropping and resizing, **not landmark alignment**.
7. The current Frequency CNN is **RGB + FFT fusion**, not frequency-only.
8. External real images have variable original dimensions; all 311 downloaded fake images are 512 x 512. This confirms a measurable source/preprocessing difference, not its causal effect on predictions.
9. Local dependencies differ considerably from the training environment, and only **3.45 GiB** of local disk space was free at inspection.

## 2. Repository map and reusable implementation

| Area | Existing files | Finding / reuse |
|---|---|---|
| Configuration | `src/config.py`, `config/base.yaml`, model YAML files | Recursive YAML defaults and deep merge can underpin the new configuration. |
| Downloads | `face_forensics_Script.py`, `src/data/download.py`, `scripts/download_deepfakeface_subset.py`, `scripts/download_genimage_adm_subset.py` | Preserve existing download/access workflow. No additional FF++ download is required. |
| Video processing | `src/data/extract_frames.py`, `src/data/prepare_videos.py` | Uniform frame indices, sequential decoding, resume-by-existing-crops support. |
| Face processing | `src/data/detect_faces.py` | MTCNN largest-face selection, expanded box, resize; reuse detector and crop helpers. |
| Manifests | `src/data/combine_manifests.py`, `pair_manifests.py`, `portable_manifest.py`, `subsample_manifest.py` | Existing CSV schema and local/cloud path conversion. |
| Splits | `src/data/splits.py` | Seeded group splits; preserve original assignments and files. Add explicit source-video audits around them. |
| Data loading | `src/data/dataset.py`, `transforms.py` | Manifest dataset, balanced sampler, model-specific normalization and resizing. |
| Models | `src/models/registry.py`, `meso.py`, `xception.py`, `hybrid.py`, `freq_cnn.py`, `ensemble.py` | Four architectures and probability combination support. No new model zoo needed. |
| Training | `src/train.py`, `scripts/modal_train.py` | BCE, AdamW, cosine schedule, AMP, initial freezing, validation selection and cloud A10 workflow. |
| Evaluation | `src/evaluate.py`, `src/eval/metrics.py`, `ensemble.py` | Predictions, binary metrics, real-versus-each-method reports, frame/video aggregation, validation-tuned ensembles. |
| Distribution shift | `src/eval/robustness.py`, `cross_generation.py`, `plot_compression_summary.py` | Legacy JPEG and external comparisons; useful scaffolding, not all requested analyses. |
| Interpretation | `src/interpret/gradcam.py`, `tsne.py` | Signed binary CAM target and classifier-input embedding extraction. |
| Notebook | `notebooks/deepfake_detection_complete.ipynb` | 43 cells, 17 code cells; end-to-end controller and results narrative, dependent on `src`. |
| Documentation | `README.md`, `DATA.md`, `PROJECT_STATUS.md`, `GROUP_PROJECT_SUMMARY.md`, original specification and assessment material | Preserve existing work; correct overbroad identity/alignment/interpretation claims during implementation. |
| Tests | `tests/test_*.py` | 20 test modules, 37 collected passing tests. |

The complete notebook should remain the user-facing entry point. Some sections review saved outputs rather than execute every advertised experiment; new experiment cells need explicit run/review controls and provenance. Earlier exploration/results notebooks also remain available.

## 3. Data inventory

### Main FF++ dataset

Manifest: `data/faceforensics_manifest_200video.csv`.

Processed root: `data/processed/faceforensics/`; categories `real`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`.

Each category contributes 200 videos and 6,000 images. The total is 1,000 videos / 30,000 crops. Processed JPEG files occupy approximately 432.7 MB in aggregate. Existence was checked for every split path; full decoding of every crop was not repeated during this audit.

| Split | File | Images | Target groups | Images per category |
|---|---|---:|---:|---:|
| Train | `data/splits_200video/train.csv` | 21,000 | 140 | 4,200 |
| Validation | `data/splits_200video/val.csv` | 4,500 | 30 | 900 |
| Test | `data/splits_200video/test.csv` | 4,500 | 30 | 900 |

There are four fake categories to one real category: binary classes are **not** balanced, despite equal category sizes. This makes balanced accuracy and AUC particularly important; compare F1 and ordinary accuracy with the class prevalence stated.

Other manifests: `data/faceforensics_manifest.csv`, `faceforensics_manifest_30video.csv`, `faceforensics_manifest_200video_15frames.csv`; older splits in `data/splits/`, half-frame splits in `data/splits_200video_15frames/`, cloud-portable splits in `data/modal_splits_200video/`. The historical 100-video-per-category split has 10,500 / 2,250 / 2,250 images.

**Configuration pitfall:** `config/hybrid_200video.yaml` and `config/xception_200video.yaml` default to the **15-frame** manifests. `scripts/modal_train.py` overrides those paths to the full 30-frame splits. Loading the YAML alone does not reproduce the final cloud training dataset.

### External faces

Evaluation manifest: `data/deepfakeface_sd15_paired.csv`; 576 rows, 288 real and 288 fake, all paths present.

Raw root: `data/raw/deepfakeface_subset/{real,fake}`; **311 JPEG images per class**. All 622 raw images could be opened for metadata inspection. Processed root: `data/processed/deepfakeface_sd15/`.

The downloader identifies `desingh/DeepFakeFace`, selecting common archive-relative filenames from `wiki.zip` and `text2img.zip`. The project attributes these to IMDB-WIKI real faces and Stable Diffusion v1.5 generated faces. Exact external generator/VAE provenance still needs dataset documentation verification before choosing a reconstruction model.

All downloaded fake images are RGB JPEG, 512 x 512. Real images are RGB JPEG with varying dimensions, including 440 x 660, 440 x 587 and many other sizes. The retained paired evaluation set is smaller than the raw download; record detection and pairing attrition explicitly in the new audit.

Filename pairing establishes a bookkeeping relationship, **not proof of identical identity/content**. External `video_id` is actually an image/pair key, not a video identity. Do not present external image aggregation as video evaluation.

Other external material: `data/genimage_adm_faces.csv` contains 131 real + 131 ADM face crops. This is another generative-image dataset, **not** a conventional external face-swap benchmark. No ready conventional Celeb-DF dataset was established by this audit; configuration entries alone are not data availability.

## 4. Preprocessing actually implemented

1. Uniformly sample up to 30 frame indices per video, sequentially decode and convert BGR to RGB.
2. MTCNN detects face boxes; choose the largest box.
3. Add a margin of 0.30 times the larger box side **on each side**, clipped to image boundaries.
4. Crop and resize to 224 x 224 with `cv2.INTER_AREA`.
5. Save JPEG using OpenCV defaults; quality is not explicitly recorded.
6. Model transforms resize again as configured, augment during training, then normalize.

There is no landmark-based warp/alignment and no persisted face boxes, landmarks, original crop sizes or face-area ratios. These measurements require new detection on raw external images. Historical names/docstrings describing the crops as aligned are inaccurate.

Hybrid uses 224 inputs with ImageNet normalization. Corrected Xception uses 299 inputs, bicubic interpolation and mean/std `(0.5, 0.5, 0.5)`.

Legacy JPEG evaluation resizes to the model's input resolution **before** JPEG corruption. `jpeg_quality=100` does **not** re-encode; it means no additional compression of existing JPEG crops from c23 videos. Preserve that historical result, but distinguish it from explicit Q100 re-encoding in the standardized study.

## 5. Split integrity and checkpoint eligibility

`GroupShuffleSplit` uses `video_id`, seed 42 for train/remainder and 43 for validation/test. In video preparation, manipulated stem `AAA_BBB` is assigned group `AAA`. This preserves the original target-video protocol.

Actual intersections:

| Split pair | Target IDs shared | IDs shared when both filename participants are included |
|---|---:|---:|
| Train / validation | 0 | 40 |
| Train / test | 0 | 32 |
| Validation / test | 0 | 12 |

Counts in the final column are unique video identifiers, not image counts or proven person identities. They were computed by splitting each crop-parent video stem on `_` and considering both participants. No person-level identity annotation proves that different original video IDs contain different people.

The source/donor overlap must be resolved or explicitly qualified before new training. **Do not silently replace the historical split or retrospectively call old checkpoints source-safe.**

### Non-destructive stricter-subset feasibility check

A possible derived protocol keeps the existing target assignment and retains a manipulated video only when **all participant video IDs belong to that same partition**. This changes no original CSV and reassigns no target. Read-only filtering produced:

| Partition | Images retained | Real images | Images per fake method | Distinct videos |
|---|---:|---:|---:|---:|
| Train | 16,680 | 4,200 | 3,120 | 556 |
| Validation | 1,380 | 900 | 120 | 46 |
| Test | 1,860 | 900 | 240 | 62 |

This would separate known participant video IDs but leaves only **4 fake validation videos and 8 fake test videos per method**, so uncertainty and class-prevalence changes are serious limitations. It still cannot certify person-level disjointness. These are feasibility counts only: **no derived split has been written or selected for training**. The implementation plan identifies this as a decision gate.

### Old checkpoints on the expanded dataset

`outputs/checkpoints/freq_cnn_100video_best.pt` embeds `data/splits/train.csv` as its training manifest. Comparing this manifest against `data/splits_200video/test.csv` yields **1,800 matching crop paths and 12 shared target groups**. It cannot be used as a fair current-test baseline or fine-tuning initialization for that comparison. Train a fresh Frequency CNN on the selected upgrade train partition.

Other 100-video checkpoints require the same eligibility check; do not mix old test results into the final full-dataset leaderboard. Existing full-data Hybrid/Xception checkpoints remain valid for reproducing the **historical target-video protocol**, with its stated source-overlap limitation.

## 6. Models, checkpoints and training

| Model | Implementation | Best reusable checkpoint | Notes |
|---|---|---|---|
| CoAtNet-0 / `hybrid` | `src/models/hybrid.py`, timm `coatnet_0_rw_224` with fallback | `outputs/modal_downloads/hybrid_200video_full_best.pt` | 106,814,054 bytes; best epoch 8; seed 42. Record resolved timm architecture, not only fallback alias. |
| Xception | `src/models/xception.py`, timm Xception/legacy alias | `outputs/modal_downloads/xception_200video_full_best.pt` | 83,548,890 bytes; best epoch 10 of 12; seed 42; corrected preprocessing. |
| Frequency CNN | `src/models/freq_cnn.py` | `outputs/checkpoints/freq_cnn_100video_best.pt` | 969,501 bytes; legacy-only eligibility. RGB and FFT branches each produce 128 features, concatenated into an MLP. |
| MesoInception | `src/models/meso.py` | `outputs/checkpoints/meso_100video_best.pt` | Historical lightweight baseline; no unnecessary new Meso study planned. |
| Probability ensemble | `src/eval/ensemble.py` | Uses two independent checkpoints | Frame- and video-calibrated ensembles are distinct configurations, not one interchangeable score. |

Additional legacy checkpoints are preserved in `outputs/checkpoints/`; partial/local training artifacts exist in `outputs/200video/`; nested cloud-download results also exist. Filenames alone must not determine which run is authoritative.

Verified training code: binary logit output, BCEWithLogitsLoss, AdamW, weight decay 1e-4, head/base learning rate 1e-4 and backbone 1e-5, cosine schedule, balanced sampler, AMP, initial 60% leaf-module freezing for three epochs, validation-AUC checkpoint selection and early stopping. Hybrid cap is eight epochs; Xception cap is twelve. Augmentation includes flip, rotation, colour jitter and random JPEG Q30-100.

Validation thresholds are selected by maximum balanced accuracy. Saved thresholds: Hybrid **0.9622429609298706**, Xception **0.9074631929397583**. They are not arbitrary 0.5 thresholds, which explains some high-probability false negatives in old figures.

Nuances to retain/document: frozen parameter modules still run in training mode (BatchNorm buffers can update); learning-rate grouping uses `head`/`classifier`/`fc` name substrings, potentially including non-head `fc` modules. `init_checkpoint` restores weights, not optimizer/scheduler/scaler state: it is not full interrupted-run resumption. All LOMO runs must start from eligible generic pretrained weights, **never a detector already trained on the held-out manipulation**.

## 7. Saved results verified against predictions

Recomputed using `roc_auc_score(label, prob_fake)` from saved CSV files. Video scores below average frame probabilities within dataset/method/video groups. Fresh inference remains a planned separate phase.

| Historical model/configuration | FF++ frame AUC | FF++ video AUC | External original AUC |
|---|---:|---:|---:|
| CoAtNet | 0.8832822531 | 0.9038888889 | 0.3856939622 |
| Xception | 0.8616807099 | 0.8958333333 | 0.3898775077 |
| Frame-calibrated ensemble | 0.8805185185 | 0.9038888889 | 0.3823181906 |
| Separately video-calibrated ensemble | N/A here | 0.9047222222 | N/A here |

Interpretation: useful discrimination on the historical FF++ test; poor transfer and below-chance score ranking on this external set. The ensemble is not better on all metrics. None of these results is evidence of universal deepfake detection, and the external drop cannot be attributed solely to diffusion generation.

Primary existing artifacts:

- `outputs/modal_downloads/{hybrid,xception}_{val,test}.json` and corresponding `_predictions.csv` (4,500 rows each).
- `outputs/modal_downloads/{hybrid,xception}_epochs.csv`, `_compression.csv` and `_compression.json`.
- `outputs/tables/model_summary_200video_cloud.csv`, `per_method_summary_200video_cloud.csv`, `compression_200video_cloud.csv`.
- `outputs/tables/{hybrid,xception}_200video_cloud_deepfakeface.json` and `_predictions.csv`.
- `outputs/tables/ensemble_200video_cloud_{frame,video,deepfakeface}.{csv,json}`.
- `outputs/figures/compression_200video_cloud.png` and existing model Grad-CAM panels.

Q10 historical frame AUCs: Hybrid 0.7124654321; Xception 0.6607648148. Preserve all intermediate qualities and severe-degradation results.

The ensemble merge checks one-to-one IDs and label consistency, but an inner join can silently omit missing samples. Add complete coverage and disjoint calibration/evaluation validation. Cloud-relative paths and Windows absolute paths require a canonical sample ID before joins.

## 8. Features and interpretation artifacts

Existing t-SNE figures and coordinate CSVs:

- `outputs/figures/hybrid_200video_tsne_test.{csv,png}`.
- `outputs/figures/hybrid_200video_tsne_domain.{csv,png}`.

`src/interpret/tsne.py` hooks the final linear layer's input; scales features, applies PCA up to 50 components, then fits seeded t-SNE. The existing CSV contains metadata/probabilities and **two-dimensional coordinates**, not full embeddings. No `.npy`, `.npz` or `.parquet` feature cache was found below `outputs`.

Re-extract and persist full penultimate embeddings (expected 768 for the Hybrid; assert actual shape) for PCA variance/rank and high-dimensional silhouette analysis. Silhouette on t-SNE coordinates is not a substitute. Joint embeddings/projection can be reused across colouring schemes; geometry cannot establish causal shortcuts.

Existing Grad-CAM chooses the last convolution and explains the **true class sign**, even for mistakes. It takes the first qualifying cases, potentially repeated frames from one video. New panels need explicit explained class, threshold, diverse-video deterministic selection and consistent layer naming. Do not infer exact eye/mouth/hair attribution without actual region masks or landmarks.

No standalone group-average FFT/radial-spectrum analysis, filtering sensitivity, controlled source audit, calibration suite, LOMO suite or VAE reconstruction implementation was found. Having an FFT branch is not equivalent to those analyses.

## 9. Environment, dependencies and resources

Python: 3.11.9, invoked with `.venv/Scripts/python.exe`. `.venv/pyvenv.cfg` has `include-system-site-packages = true`; the environment is not isolated from user/system packages.

| Package | Observed locally | Existing pinned/cloud expectation where relevant |
|---|---|---|
| torch | 2.8.0+cu128 | 2.2.x; cloud 2.2.2 |
| torchvision | 0.23.0+cu128 | 0.17.x; cloud 0.17.2 |
| timm | 1.0.27 | 1.0.7 |
| numpy | 2.4.6 | Cloud 1.26.4 |
| scikit-learn | 1.7.1 | 1.5.1 |
| facenet-pytorch | 2.6.0 | 2.6.0 |
| albumentations | 1.3.1 | 1.3.1 |
| modal | 1.5.3 | Available locally |
| transformers | 4.55.1 | Available, compatibility still needs testing |
| safetensors | 0.6.2 | Available |
| scikit-image | 0.25.2 | Available for SSIM |
| diffusers | Missing | Needed for planned VAE implementation |
| accelerate | Missing | Optional/helpful depending on selected loading strategy |
| lpips | Missing | Needed for perceptual reconstruction metric |

Metadata presence is not proof every optional package is runtime-compatible. Do not reinstall or downgrade the working environment blindly. Use a separately pinned optional reconstruction environment/cloud image and smoke tests. Pin the chosen VAE repository and revision; verify access/licence and external generator compatibility before downloading.

GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB total, 7,956 MiB free at the read-only check. CUDA is available. Local GPU load/free memory is a snapshot.

Disk: 3,700,760,576 bytes free, approximately **3.45 GiB**. Do not create all corruptions as full image duplicates locally. Store compact predictions/features and use bounded caches; keep bulky reconstruction/training artifacts on Modal or a user-selected drive. Do not delete any old dataset/checkpoint to make space without explicit authorization.

Existing Modal code uses an A10, 8 CPUs, 32 GiB host memory, app `deepfake-detection-training`, volume `deepfake-detection`. Account access, current balance, remote volume contents and current prices were **not** queried during this local audit. Prior account authorization does not establish present credit balance.

Measured historical epoch-log totals (training + per-epoch validation, excluding setup/download/final evaluation): Hybrid 489.76 seconds across 8 epochs; Xception 1,065.76 seconds across 12 epochs. These support order-of-magnitude planning, not guaranteed runtime. See the implementation plan for conservative GPU-hour ranges.

## 10. Baseline test report

Command before any task edits:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Result: **37 passed, 1 warning in 67.91 seconds**; exit code 0.

Warning: joblib/loky could not determine physical CPU cores on Windows and fell back to logical cores during a t-SNE test. This is not a failed test.

These tests establish existing software behavior, not scientific validity of splits or unimplemented experiments. New leakage, provenance, calibration, transform and reconstruction tests are required.

## 11. Preservation and remaining work

Pre-existing dirty worktree: modified `GROUP_PROJECT_SUMMARY.md`, `PROJECT_STATUS.md`, `README.md`, `notebooks/deepfake_detection_complete.ipynb`; untracked t-SNE source/test and four t-SNE plot/CSV artifacts. These were preserved. No automatic commit should bundle them as new audit work.

This audit creates only new documentation/test-record files. It does not mark baseline fresh inference, normalized external evaluation, new training or any requested experiment complete. The complete experiment-to-code mapping, output specification, costs, risks and split decision gate are in `outputs/generalization_upgrade/IMPLEMENTATION_PLAN.md`.
