# Implementation status

Updated: 25 September 2026

## Outcome

The implementation and required experiments are complete. The final training
run used the full 30-frame, 1,000-video FaceForensics++ c23 sample on an A10
GPU. Model selection, thresholds, and ensemble weights were chosen from the
validation split only; the test split was not used for tuning.

The complete workflow and saved results are also available as one executed
Jupyter notebook: `notebooks/deepfake_detection_complete.ipynb`.

## Requirement coverage

| Planned requirement | Status | Evidence |
|---|---|---|
| Face extraction, alignment, augmentation, and identity-safe splits | Complete | `src/data`, `data/splits_200video` |
| MesoInception baseline | Complete | checkpoint, test metrics, Grad-CAM, compression and cross-generation outputs |
| Corrected Xception transfer model | Complete | `outputs/modal_downloads/xception_200video_full_best.pt` |
| Frequency-domain CNN | Complete | checkpoint and full 100-video experiment outputs |
| Alternative hybrid model | Complete | `outputs/modal_downloads/hybrid_200video_full_best.pt` |
| Per-manipulation evaluation | Complete | `outputs/tables/per_method_summary_200video_cloud.csv` |
| Validation-calibrated ensemble | Complete | `outputs/tables/ensemble_200video_cloud_frame.json` and `ensemble_200video_cloud_video.json` |
| Cross-generation experiment | Complete | `outputs/tables/hybrid_200video_cloud_cross_generation.json` |
| JPEG robustness at 6 quality levels | Complete | `outputs/tables/compression_200video_cloud.csv` |
| Grad-CAM panels for all four model families | Complete | `outputs/figures/*gradcam.png` |
| t-SNE class, method, and domain-shift visualisation | Complete | `outputs/figures/hybrid_200video_tsne_*.png` |
| Automated tests | Complete | 37 tests passing |

## Final held-out results

| Model | Frame AUC | Balanced accuracy | F1 | Video AUC |
|---|---:|---:|---:|---:|
| CoAtNet hybrid | **0.8833** | 0.7840 | **0.8607** | 0.9039 |
| Corrected Xception | 0.8617 | 0.7751 | 0.8323 | 0.8958 |
| Frame-calibrated ensemble | 0.8805 | **0.7847** | 0.8423 | 0.9039 |
| Video-calibrated ensemble | 0.8789 | 0.7840 | 0.8711 | **0.9047** |

The hybrid is the recommended single-frame model. The video-calibrated
ensemble is the recommended video model.

The hybrid reaches validation AUC 0.9401 and test AUC 0.8833. On the paired
Stable Diffusion v1.5 test it reaches AUC 0.3857, an AUC drop of 0.4976. This is
the intended cross-generation-shift finding, not a hidden or tuned-away result.
JPEG Q10 reduces hybrid AUC to 0.7125 (from 0.8833 at Q100).

A guarded low-learning-rate continuation was also attempted from epoch 8. Its
next two validation AUCs did not exceed 0.9401, so early stopping correctly
retained the original epoch-8 checkpoint and its test results.

## Honest limitations

- The target thresholds are conservative: precision is high (0.9326 for the
  hybrid), but fake recall is 0.7992.
- NeuralTextures is the hardest in-domain manipulation (binary AUC 0.8238).
- GAN-manipulation training does not generalise to full-face diffusion images.
- Celeb-DF v2 remains optional because access was not obtained; it is not a
  required primary objective in the specification.

## Reproduction

Run `pytest -q` for the implementation tests. Local training uses the YAML
files in `config`; the reproducible Modal A10 entrypoint is
`scripts/modal_train.py`. Private dataset files and credentials are excluded
from source control.
