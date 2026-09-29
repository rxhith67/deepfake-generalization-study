# Deepfake Detection in the Wild

## Complete Group Project Summary

**Course:** UTS 42177 Image Processing and Pattern Recognition  
**Task:** Binary classification of facial images as real (`0`) or fake (`1`)  
**Implementation:** Python 3.11, PyTorch, OpenCV, Albumentations, MTCNN and `timm`  
**Current status:** Core implementation and required experiments complete  
**Last updated:** 25 September 2026

---

## 1. Project overview

This project develops a deepfake-detection system that receives a face image,
or a face extracted from a video frame, and predicts the probability that it
is fake.

The system was trained primarily on **FaceForensics++**, which contains
traditional GAN-era face manipulations. We did not restrict the project to a
single in-domain accuracy test. Our main research question was whether a model
trained on these established manipulation techniques would continue to work
on newer diffusion-generated faces and on images degraded by strong JPEG
compression.

The project therefore investigates two practical problems:

1. **Cross-generation generalisation:** can a detector trained on Deepfakes,
   Face2Face, FaceSwap and NeuralTextures detect Stable Diffusion faces?
2. **Compression robustness:** how much performance is lost when face images
   are recompressed at qualities similar to online and social-media content?

Four model families were implemented, trained and compared. The strongest two
were then combined using an ensemble calibrated only on validation data.

---

## 2. Why this is an image-processing project

Deepfake detection is not only a neural-network classification problem. The
quality of the image-processing pipeline determines what visual evidence is
presented to the model.

Our pipeline includes:

1. **Uniform frame sampling:** 30 frames are sampled across the duration of
   each video instead of using only neighbouring frames.
2. **Face detection:** the dominant face is detected in each sampled frame.
3. **Face cropping and alignment:** background content is removed so the model
   focuses on the manipulated facial area.
4. **Spatial normalisation:** crops are resized and normalised according to the
   requirements of each pretrained architecture.
5. **Image augmentation:** training images receive horizontal flips, rotations,
   colour variation and simulated JPEG degradation.
6. **Frequency analysis:** the frequency-domain CNN uses spectral information
   to search for generation and resampling artefacts that may not be obvious in
   RGB pixels.
7. **Controlled degradation:** test images are recompressed at six JPEG quality
   levels to quantify robustness.
8. **Visual explanation:** Grad-CAM heatmaps show which facial regions
   influenced model decisions.
9. **Temporal aggregation:** frame probabilities are averaged to generate a
   video-level probability.

The project consequently covers spatial processing, frequency-domain analysis,
image degradation, feature learning and visual interpretation.

---

## 3. Datasets

### 3.1 FaceForensics++ training and in-domain testing

The final experiment uses the lightly compressed **c23** version of
FaceForensics++.

| Category | Videos | Crops per video | Total crops |
|---|---:|---:|---:|
| Real/pristine | 200 | 30 | 6,000 |
| Deepfakes | 200 | 30 | 6,000 |
| Face2Face | 200 | 30 | 6,000 |
| FaceSwap | 200 | 30 | 6,000 |
| NeuralTextures | 200 | 30 | 6,000 |
| **Total** | **1,000** |  | **30,000** |

The data was split by target identity rather than by individual image. The
pristine source and all manipulations associated with an identity remain in
the same partition, preventing frames of the same person/video pair from
leaking between training and testing.

| Split | Identity groups | Images |
|---|---:|---:|
| Training | 140 | 21,000 |
| Validation | 30 | 4,500 |
| Test | 30 | 4,500 |

Because there are four fake methods for every real category, the final test
set contains 900 real and 3,600 fake frames. For this reason we report balanced
accuracy and AUC in addition to ordinary accuracy.

### 3.2 Cross-generation dataset

The cross-generation experiment uses a paired subset of **DeepFakeFace**:

- 288 real IMDB-WIKI face photographs;
- 288 Stable Diffusion v1.5 generated faces;
- 576 aligned images in total.

Pairing the filenames gives a more controlled comparison than independently
sampling unrelated real and generated faces.

### 3.3 Exploratory and optional datasets

- A GenImage/ADM subset was processed experimentally but excluded from the
  primary claims after visual quality checks showed that its real ImageNet
  partition was not consistently human-face-only.
- Celeb-DF v2 remains an optional cross-dataset experiment because access was
  not obtained. It was not required for completion of the primary objectives.

---

## 4. Processing pipeline

```text
FaceForensics++ videos
        |
        v
Uniformly sample 30 frames per video
        |
        v
Detect the largest face with MTCNN
        |
        v
Crop, align and resize the face
        |
        +-----------------------------+
        |                             |
        v                             v
Spatial RGB transforms        Frequency representation
        |                             |
        +--------------+--------------+
                       v
              Deepfake detector
                       |
                       v
              P(fake) for each frame
                       |
                       v
       Frame metrics and video aggregation
```

Training augmentation consists of:

- horizontal flipping with probability 0.5;
- rotation up to approximately 10 degrees;
- brightness, contrast, saturation and hue variation;
- random JPEG compression between quality 30 and 100;
- architecture-specific resizing and normalisation.

CoAtNet and the custom CNNs use 224×224 inputs with ImageNet normalisation.
Corrected Xception preprocessing uses 299×299 inputs, bicubic interpolation and
the `[-1, 1]`-style normalisation expected by its pretrained weights.

---

## 5. Models implemented

| Model | Purpose | Main idea |
|---|---|---|
| MesoInception | Shallow baseline | Compact convolutional model intended for mesoscopic facial artefacts |
| Xception | Strong transfer baseline | ImageNet-pretrained depthwise-separable convolutional network |
| Frequency CNN | Image-processing alternative | Learns from spectral/frequency evidence rather than RGB evidence alone |
| CoAtNet hybrid | Main advanced model | Combines convolutional inductive bias with attention-based global modelling |
| Hybrid–Xception ensemble | Final alternative | Combines complementary model probabilities using validation-selected weights |

The MesoInception, frequency CNN and early model comparison were completed on
the initial 500-video/15,000-crop experiment. After correcting Xception's input
preprocessing, the two strongest architectures—Xception and CoAtNet—were
retrained on the expanded 1,000-video/30,000-crop dataset.

---

## 6. Training methodology

The final models were trained on an NVIDIA A10 GPU using the following
procedure:

- binary cross-entropy with logits;
- AdamW optimisation and weight decay;
- cosine learning-rate decay;
- balanced sampling to prevent the four fake methods from overwhelming the
  real class during optimisation;
- mixed-precision CUDA training;
- pretrained backbones with the first 60% frozen for the first three epochs;
- lower learning rate for backbone parameters than classifier parameters;
- early stopping and checkpoint selection using **validation AUC**;
- operating-threshold selection using validation balanced accuracy;
- deterministic seed and configuration stored inside each checkpoint.

Final cloud settings:

| Model | Batch size | Scheduled epochs | Selected epoch | Best validation AUC |
|---|---:|---:|---:|---:|
| CoAtNet hybrid | 64 | 8 | 8 | **0.9401** |
| Corrected Xception | 32 | 12 | 10 | 0.9167 |

The hybrid validation AUC was still rising at epoch 8, so a guarded
low-learning-rate continuation was attempted. The next two validation results
did not exceed 0.9401. Early stopping correctly retained the original epoch-8
checkpoint, demonstrating that additional training would have increased
overfitting rather than improved model selection performance.

### Leakage controls

The test set was not used to select epochs, ensemble weights or classification
thresholds. Frame-ensemble weights, video-ensemble weights and operating
thresholds were determined from validation predictions and applied unchanged
to the test data.

---

## 7. Final in-domain results

### 7.1 Main held-out test results

| Model | Frame AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | Video AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CoAtNet hybrid** | **0.8833** | **0.7931** | 0.7840 | 0.9326 | **0.7992** | **0.8607** | 0.9039 |
| Corrected Xception | 0.8617 | 0.7589 | 0.7751 | 0.9380 | 0.7481 | 0.8323 | 0.8958 |
| Frame-calibrated ensemble | 0.8805 | 0.7716 | **0.7847** | **0.9404** | 0.7628 | 0.8423 | 0.9039 |

The **CoAtNet hybrid is the recommended single-frame detector** because it has
the strongest frame AUC and F1. Its high precision means that positive
predictions are usually correct, although its conservative threshold misses
some manipulated frames.

The separately calibrated video ensemble uses a hybrid/Xception weighting of
0.62/0.38 and achieves:

- video AUC: **0.9047**;
- video balanced accuracy: **0.8250**;
- video F1: **0.8947**.

It is therefore the recommended configuration when an entire video, rather
than an isolated frame, is available. The frame-calibrated ensemble uses a
0.91/0.09 hybrid/Xception weighting. Ensembling did not exceed the hybrid's
frame AUC, showing that the two models make sufficiently correlated errors at
frame level.

### 7.2 Hybrid performance by manipulation method

Each fake method is compared against the real test frames.

| Manipulation | Binary AUC | Balanced accuracy | Fake recall | F1 |
|---|---:|---:|---:|---:|
| Deepfakes | **0.9139** | **0.8078** | **0.8467** | **0.8150** |
| Face2Face | 0.9022 | 0.8039 | 0.8389 | 0.8105 |
| FaceSwap | 0.8932 | 0.8022 | 0.8356 | 0.8086 |
| NeuralTextures | 0.8238 | 0.7222 | 0.6756 | 0.7086 |

NeuralTextures is the most difficult in-domain method. This suggests that its
rendering artefacts differ from the stronger signals learned for swapping and
reenactment methods.

### 7.3 Earlier four-model comparison

The original 500-video experiment established the model baseline before the
expanded GPU run.

| Model | Test accuracy | Test AUC |
|---|---:|---:|
| MesoInception | 0.5813 | 0.6709 |
| Frequency CNN | 0.5951 | 0.5899 |
| CoAtNet hybrid | **0.8013** | **0.8179** |
| Original Xception run | 0.7204 | 0.7550 |
| Corrected Xception preprocessing | 0.7209 | 0.8020 |

Correcting Xception's resize and normalisation increased its AUC from 0.7550
to 0.8020 on the same smaller experimental setup. Expanding to 1,000 videos
then raised the final Xception AUC to 0.8617.

---

## 8. Compression-robustness experiment

The test images were JPEG-encoded in memory at six controlled quality levels.
The stored source images were not overwritten.

| JPEG quality | Hybrid AUC | Xception AUC |
|---:|---:|---:|
| 100 | **0.8833** | 0.8617 |
| 90 | **0.8636** | 0.8505 |
| 60 | **0.8444** | 0.8261 |
| 40 | **0.8268** | 0.8010 |
| 20 | **0.7763** | 0.7592 |
| 10 | **0.7125** | 0.6608 |

Both models degrade steadily as compression becomes more severe. The hybrid
is more robust at every tested quality level, but its Q10 AUC still falls by
approximately 0.171 from the uncompressed evaluation. This confirms that
compression can remove or distort the subtle forensic traces used by the
detectors.

---

## 9. Cross-generation experiment

| Model | FaceForensics++ AUC | Stable Diffusion face AUC | AUC drop |
|---|---:|---:|---:|
| CoAtNet hybrid | 0.8833 | 0.3857 | **0.4976** |
| Corrected Xception | 0.8617 | **0.3899** | 0.4718 |
| Frame ensemble | 0.8805 | 0.3823 | 0.4982 |

This is the project's main scientific result. Stronger performance on known
GAN-era manipulations does **not** transfer to full-face Stable Diffusion
generation. The below-0.5 AUC indicates a substantial domain mismatch: some
visual properties associated with real and fake samples in FaceForensics++ are
reversed or absent in the diffusion dataset.

This result should not be described as a successful diffusion detector.
Instead, it demonstrates the exact generalisation problem the project was
designed to measure. We deliberately did not tune on the held-out diffusion
test set, because doing so would invalidate the cross-generation experiment.

---

## 10. Interpretability

Grad-CAM panels were produced for all four model families, including correct
and incorrect examples. These heatmaps help determine whether predictions are
based on relevant areas such as eyes, mouth, skin boundaries and blending
regions, or on irrelevant background and compression artefacts.

The visualisations are stored in `outputs/figures/*gradcam.png`.

Two t-SNE projections were also generated from the final hybrid model's
768-dimensional penultimate-layer features. The FaceForensics++ plot shows
partial real/fake overlap and method-specific clusters. The combined-domain
plot shows FaceForensics++ and DeepFakeFace/Stable Diffusion occupying strongly
different regions, while real and fake Stable Diffusion-domain samples remain
mixed. This provides qualitative support for the measured cross-generation
failure. The plots are stored in `outputs/figures/hybrid_200video_tsne_*.png`.

---

## 11. What has been completed

| Requirement | Status |
|---|---|
| Video frame extraction and uniform sampling | Complete |
| Face detection, cropping and alignment | Complete |
| Leakage-safe identity grouping | Complete |
| MesoInception baseline | Complete |
| Xception transfer learning | Complete |
| Frequency-domain CNN | Complete |
| CoAtNet hybrid alternative | Complete |
| Validation-calibrated ensemble | Complete |
| Frame, video and per-method metrics | Complete |
| Stable Diffusion cross-generation test | Complete |
| Six-level JPEG robustness test | Complete |
| Grad-CAM interpretation | Complete |
| t-SNE feature and domain-shift visualisation | Complete |
| Configuration and reproducibility support | Complete |
| Automated implementation tests | **37 passing** |

The implementation satisfies the main functional objectives. However, the
proposal's aspirational numerical success criteria of over 75% baseline
accuracy and over 90% Xception accuracy were not achieved on the strict
identity-held-out test set. These values must be reported honestly rather than
replaced with validation results or a leakage-prone frame split.

---

## 12. Strengths and limitations

### Strengths

- Complete end-to-end video-to-prediction pipeline.
- Strict identity-safe split and validation-only model selection.
- Four substantially different model approaches.
- Expanded final dataset and GPU training.
- Frame-level, video-level and manipulation-level analysis.
- Controlled compression experiment at six quality levels.
- Paired cross-generation evaluation.
- Explainability outputs and automated tests.
- Honest negative result demonstrating a real domain-generalisation problem.

### Limitations

- The 1,000-video sample is still a subset of the complete FaceForensics++
  release.
- Four fake categories versus one real category creates a naturally imbalanced
  evaluation set, although balanced sampling and balanced metrics address this.
- Cross-generation performance is poor because diffusion synthesis differs
  substantially from the training manipulations.
- The system evaluates cropped faces rather than complete audiovisual videos;
  audio, motion and temporal inconsistency features are not modelled directly.
- Celeb-DF v2 was not evaluated because access was not available.
- NeuralTextures remains noticeably harder than the other in-domain methods.

---

## 13. Recommended final project message

The strongest way to present the project is not to claim that deepfake
detection has been solved. The evidence supports the following conclusion:

> A CoAtNet-based detector can learn useful forensic evidence from
> FaceForensics++ and reaches 0.883 frame AUC and 0.904 video AUC on held-out
> identities. However, performance falls under severe JPEG compression and
> collapses on Stable Diffusion faces, demonstrating that in-domain accuracy
> alone is not evidence of real-world deepfake generalisation.

This combines a functioning detector with a meaningful image-processing and
domain-shift investigation.

---

## 14. Important project files

- `notebooks/deepfake_detection_complete.ipynb` — complete executed project notebook and recommended group entry point
- `README.md` — setup, commands and concise project overview
- `PROJECT_STATUS.md` — implementation checklist and final status
- `DATA.md` — dataset preparation and manifest format
- `src/data/` — extraction, detection, preparation and split tools
- `src/models/` — implemented model architectures
- `src/train.py` — training and checkpoint-resume pipeline
- `src/evaluate.py` — detailed frame/video evaluation
- `src/eval/` — ensemble, robustness and cross-generation experiments
- `scripts/modal_train.py` — reproducible A10 cloud training and evaluation
- `outputs/tables/model_summary_200video_cloud.csv` — final consolidated results
- `outputs/tables/per_method_summary_200video_cloud.csv` — method-level results
- `outputs/tables/compression_200video_cloud.csv` — JPEG robustness results
- `outputs/figures/compression_200video_cloud.png` — compression curve
- `outputs/figures/hybrid_200video_tsne_test.png` — in-domain feature projection
- `outputs/figures/hybrid_200video_tsne_domain.png` — cross-domain feature projection
- `outputs/modal_downloads/hybrid_200video_full_best.pt` — best hybrid checkpoint
- `outputs/modal_downloads/xception_200video_full_best.pt` — best Xception checkpoint

---

## 15. Remaining group work

The core implementation is finished. The remaining work is primarily project
communication:

1. convert these results into the final implementation report;
2. select Grad-CAM and compression figures for the report;
3. explain the difference between validation and held-out test performance;
4. discuss compression sensitivity and cross-generation failure;
5. ensure every group member can explain their assigned component;
6. prepare the presentation/demo and final repository packaging.

Optional future technical work could add Celeb-DF, temporal modelling,
diffusion-aware training data, domain adaptation or self-supervised forensic
pretraining. These extensions are not required to describe the current
implementation as complete.
