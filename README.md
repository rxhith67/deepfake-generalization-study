# Deepfake Detection in the Wild

PyTorch implementation of a binary face-image detector (`real=0`, `fake=1`) for
UTS 42177. The project compares a compact MesoInception network, Xception
transfer learning, a spatial/frequency dual-domain CNN, and a CoAtNet hybrid.
Its two principal experiments measure generalisation from GAN-era
FaceForensics++ manipulations to diffusion-generated faces and robustness to
JPEG compression.

## Quick start

```bash
conda env create -f environment.yml
conda activate deepfake
jupyter lab notebooks/deepfake_detection_complete.ipynb
```

The recommended entry point is the fully executed master notebook above. It
contains the complete workflow, final tables and figures, with expensive
download/training flags disabled by default. The equivalent command-line
workflow remains available for automated and cloud runs:

```bash
python -m src.data.extract_frames --input data/raw/videos --output data/frames
python -m src.data.detect_faces --input data/frames --output data/processed/faceforensics --label 0
python -m src.data.splits --manifest data/manifest.csv --output-dir data/splits
python -m src.train --config config/meso.yaml
```

The PyTorch 2.2/torchvision 0.17 pins are intentional: they are the supported
pair for the specified `facenet-pytorch==2.6.0` face detector.

FaceForensics++ and Celeb-DF v2 require their own access approval. GenImage is
public but remains non-commercial/CC BY-NC-SA. The repository includes only
download helpers, not dataset content. See [`DATA.md`](DATA.md) for the expected
manifest, licence notes, and preparation workflow.

## Completed experiment snapshot

The final run uses 30,000 FaceForensics++ c23 aligned face crops from 1,000
videos (200 pristine and 200 per manipulation), split 70/15/15 by target
identity into 21,000/4,500/4,500 images. The cross-generation set contains 576
paired DeepFakeFace crops (288 IMDB-WIKI photographs and 288 Stable Diffusion
v1.5 outputs). Model selection, ensemble weights, and operating thresholds are
learned on validation data only.

| Model | FF++ frame AUC | Balanced accuracy | FF++ video AUC | SD1.5-face AUC | JPEG Q10 AUC |
|---|---:|---:|---:|---:|---:|
| CoAtNet hybrid | **0.8833** | 0.7840 | 0.9039 | 0.3857 | **0.7125** |
| Corrected Xception | 0.8617 | 0.7751 | 0.8958 | **0.3899** | 0.6608 |
| Validation-weighted ensemble | 0.8805 | **0.7847** | **0.9047**¹ | 0.3823 | - |

¹ Video AUC uses the separately validation-calibrated video ensemble (hybrid
weight 0.62); the frame ensemble uses hybrid weight 0.91. The hybrid's
validation AUC is 0.9401. Its test precision is 0.9326, recall 0.7992, and F1
0.8607. Per-method binary AUC ranges from 0.8238 (NeuralTextures) to 0.9139
(Deepfakes).

The complete current results are in
`outputs/tables/model_summary_200video_cloud.csv`, with the compression curve
in `outputs/figures/compression_200video_cloud.png`. The primary scientific
finding is the 0.4976 hybrid AUC drop on unseen diffusion faces: expanding the
training set improves in-domain detection, but does not solve cross-generation
generalisation. Earlier four-model baseline results remain in
`outputs/tables/model_summary_100video.csv`.

## Commands

Train one model:

```bash
python -m src.train --config config/xception.yaml
```

Evaluate a checkpoint and write JSON/CSV results:

```bash
python -m src.evaluate --checkpoint outputs/checkpoints/xception_best.pt \
  --csv data/splits/test.csv --output outputs/tables/xception_test.json
```

Run the two headline experiments:

```bash
python -m src.eval.cross_generation --checkpoint outputs/checkpoints/xception_best.pt \
  --in-domain-csv data/splits/test.csv --cross-csv data/splits/diffusion_test.csv
python -m src.eval.robustness --checkpoint outputs/checkpoints/xception_best.pt \
  --csv data/splits/test.csv --qualities 100 90 60 40 20 10
```

Download and prepare the paired cross-generation subset:

```bash
python scripts/download_deepfakeface_subset.py --count 300
python -m src.data.detect_faces --input data/raw/deepfakeface_subset/real \
  --output data/processed/deepfakeface_sd15/real --label 0 \
  --method real --dataset deepfakeface --device cuda
python -m src.data.detect_faces --input data/raw/deepfakeface_subset/fake \
  --output data/processed/deepfakeface_sd15/fake --label 1 \
  --method StableDiffusion_v1_5 --dataset deepfakeface --device cuda
python -m src.data.pair_manifests \
  --first data/manifests/deepfakeface_real.csv \
  --second data/manifests/deepfakeface_sd15.csv \
  --output data/deepfakeface_sd15_paired.csv
```

Generate a Grad-CAM grid:

```bash
python -m src.interpret.gradcam --checkpoint outputs/checkpoints/xception_best.pt \
  --csv data/splits/test.csv --output outputs/figures/xception_gradcam.png
```

Generate a t-SNE projection of the final hybrid feature space:

```bash
python -m src.interpret.tsne \
  --checkpoint outputs/modal_downloads/hybrid_200video_full_best.pt \
  --csv data/splits_200video/test.csv data/deepfakeface_sd15_paired.csv \
  --source-names FaceForensics++ DeepFakeFace-SD1.5 \
  --data-root data --device cuda --max-samples 1200 \
  --output outputs/figures/hybrid_200video_tsne_domain.png
```

Every CLI supports `--help`. Paths in CSV manifests are resolved relative to
`data.root` from the merged YAML configuration unless an explicit `--data-root`
is provided.

## Testing

```bash
pytest -q
```

The tests use generated videos/images and do not require any research dataset.
Pretrained model downloads are disabled in tests.

## Reproducibility notes

- Splits are made by `video_id`; the splitter rejects leakage between splits.
- The random seed, merged configuration, model name, validation metrics, and
  decision threshold are stored in each checkpoint.
- Training uses balanced sampling, AdamW, cosine decay, early stopping on
  validation AUC, and CUDA mixed precision when available.
- Evaluation produces probabilities, predictions, global metrics, and optional
  per-manipulation metrics suitable for the implementation report.
- t-SNE uses validation-independent penultimate-layer features, deterministic
  sampling, PCA pre-reduction, and a fixed random seed.
