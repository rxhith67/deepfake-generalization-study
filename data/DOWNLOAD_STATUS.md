# Dataset download status

| Dataset | Purpose | Access requested | Downloaded | Processed |
|---|---|---:|---:|---:|
| FaceForensics++ c23 sample (200 videos/class) | Primary train/validation/test | Yes | Yes | Yes |
| DeepFakeFace SD1.5/wiki paired subset | Cross-generation test | Public | Yes | Yes |
| GenImage ADM exploratory subset | Data-pipeline exploration only | Public | Yes | Excluded after visual QA |
| Celeb-DF v2 | Optional cross-dataset test | No | No | No |

## Current local sample

- 1,000 videos: 200 pristine originals and 200 from each of Deepfakes,
  Face2Face, FaceSwap, and NeuralTextures.
- 30,000 aligned FaceForensics++ face crops: 6,000 per class.
- Leakage-safe target-identity split: 21,000 train, 4,500 validation, 4,500 test.
  Manipulated pair names are grouped by their target video ID with the matching
  pristine video.
- 576 aligned DeepFakeFace crops: 288 matched IMDB-WIKI real/Stable Diffusion
  v1.5 pairs.
- All four baseline models, corrected Xception preprocessing, validation-only
  calibration, JPEG tests, cross-generation tests, ensembles, and Grad-CAM
  panels are complete. The final CoAtNet hybrid and Xception models were trained
  on the full 30-frame split using an A10 GPU. Consolidated final metrics are in
  `outputs/tables/model_summary_200video_cloud.csv`.

## Next external action

1. Optional: request Celeb-DF v2 access for the secondary cross-dataset test.
2. Optional: add a diffusion-face training partition only as a separately
   documented domain-generalisation experiment; do not tune on the held-out
   DeepFakeFace test pairs.

Never commit dataset archives, extracted videos, face crops, private download URLs,
or access tokens. The project `.gitignore` already excludes raw and processed data.
