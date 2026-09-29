# Dataset preparation

Obtain each dataset from its official source and comply with its licence. Raw
and processed data are ignored by Git.

The completed primary sample contains 1,000 FaceForensics++ c23 videos (200
pristine plus 200 for each of Deepfakes, Face2Face, FaceSwap, and
NeuralTextures) and 30 uniformly sampled, aligned crops per video. The
identity-safe 70/15/15 manifests are in `data/splits_200video`; their counts are
21,000 train, 4,500 validation, and 4,500 test images. The lighter
`data/splits_200video_15frames` manifests are retained for machines with less
memory, but are not the source of the final cloud results.

## Manifest schema

All training and evaluation commands consume UTF-8 CSV files with these fields:

| Column | Required | Meaning |
|---|---:|---|
| `image_path` | yes | Absolute path, or path relative to `data.root` |
| `label` | yes | `0` for real, `1` for fake |
| `video_id` | yes for splitting | Source video identifier; frames from one ID stay together |
| `method` | recommended | `real`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, generator name, etc. |
| `dataset` | recommended | `faceforensics`, `celeb_df`, `ai_genbench`, etc. |

Example:

```csv
image_path,label,video_id,method,dataset
processed/faceforensics/real/000/000001.jpg,0,real_000,real,faceforensics
processed/faceforensics/fake/Deepfakes/000/000001.jpg,1,deepfakes_000,Deepfakes,faceforensics
```

Use the same identity-group `video_id` for a pristine source and every
manipulation targeting that source (for example, target `033` for both `033`
and `033_097`). Run `python -m src.data.splits --help` to create deterministic
70/15/15 splits. The command checks that no identity group leaks across outputs.

For diffusion tests, build a separate balanced manifest containing face-only
real and generated samples. Dataset-specific filtering is intentionally not
guessed by this repository because AI GenBench/GenImage releases and metadata
layouts can change; export them to the stable schema above.

## Reproducing the paired Stable Diffusion face test

The primary cross-generation result uses DeepFakeFace, the dataset released
with *Robustness and Generalizability of Deepfake Detection: A Study with
Diffusion Models*. Its `text2img` partition contains Stable Diffusion v1.5
faces and its `wiki` partition contains corresponding IMDB-WIKI photographs.
The helper performs HTTP range reads, so selecting 300 matched pairs does not
require downloading both complete multi-gigabyte ZIP files.

```bash
python scripts/download_deepfakeface_subset.py --count 300 --seed 42
python -m src.data.detect_faces --input data/raw/deepfakeface_subset/real \
  --output data/processed/deepfakeface_sd15/real \
  --manifest data/manifests/deepfakeface_real.csv \
  --label 0 --method real --dataset deepfakeface --device cuda
python -m src.data.detect_faces --input data/raw/deepfakeface_subset/fake \
  --output data/processed/deepfakeface_sd15/fake \
  --manifest data/manifests/deepfakeface_sd15.csv \
  --label 1 --method StableDiffusion_v1_5 --dataset deepfakeface --device cuda
python -m src.data.pair_manifests \
  --first data/manifests/deepfakeface_real.csv \
  --second data/manifests/deepfakeface_sd15.csv \
  --output data/deepfakeface_sd15_paired.csv --key-depth 2
```

After intersecting filenames that passed MTCNN in both partitions, the current
evaluation manifest contains 288 matched identities (576 rows). This paired
design controls identity/content better than independently sampling each class.

## Exploratory GenImage route

The full GenImage release is hundreds of gigabytes. The project also includes
an exploratory route using the
middle ADM test shard from the community `nebula/GenImage-arrow` repackaging of
the official GenImage content. It is about 508 MB and contains 2,153 real and
1,694 ADM-generated images before face filtering. The upstream GenImage
CC BY-NC-SA/non-commercial terms still apply.

```bash
python scripts/download_genimage_adm_subset.py
python -m src.data.prepare_genimage_arrow \
  --arrow data/raw/genimage_arrow/data/test/ADM/data-00001-of-00003.arrow \
  --output data/processed/genimage_adm \
  --manifest data/genimage_adm_faces.csv \
  --max-per-class 250 --device cuda
```

The converter shuffles rows, runs MTCNN, aligns the largest detection, and
automatically trims the manifest to equal class counts. Do not call this result
"face-only" without manual/semantic validation: ImageNet contains many animal
classes and face detectors can accept animal faces. For that reason, the ADM
result is archived as exploratory and is not used in the primary result table.
