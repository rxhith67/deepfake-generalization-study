# Results-integrity report (read-only; no models run)

Independent recomputations from raw saved predictions/scores: **83**; mismatches: **0**.

Dataset sizes verified from manifests: train 16,680 / val 1,380 / test 1,860 frames (62 videos: 30 real, 32 fake; 900 real + 960 fake frames); external normalized 576 (288/288); reconstruction FF++ secondary 600; external raw fakes 311 (see provenance).

Structural assertions passed: external CIs are image-level; new-protocol FF++ CIs are source-video-component; historical FF++ CIs are target-video; reconstruction rows have no ECE/NLL/Brier and are `non_probability`; reconstruction orientation reproduces the master AUCs with fixed signs (-MSE, +SSIM, -LPIPS); historical checkpoints never appear on the source-disjoint test set and new controls/LOMO never on the historical test set.

## Phrase scan of FINAL_EXPERIMENT_SUMMARY.md

- superior: 0 hit(s)
- more robust (unnegated): 0 hit(s)
- calibrated probability: 1 hit(s)
    - reconstruction scores are anomaly scores, never calibrated probabilities.
- universal (unnegated): 0 hit(s)
- better detector (unnegated): 0 hit(s)
- general detector: 0 hit(s)

## Recomputation table (AUC)

| check | recomputed | recorded | match |
|---|---|---|---|
| matched hybrid/ffpp_source_safe raw-AUC == master | 0.845661 | 0.845661 | yes |
| matched hybrid/external_normalized raw-AUC == master | 0.366741 | 0.366741 | yes |
| matched hybrid/external_original_matched raw-AUC == master | 0.400680 | 0.400680 | yes |
| jpeg hybrid/Q100 | 0.848995 | 0.848995 | yes |
| jpeg hybrid/Q90 | 0.826657 | 0.826657 | yes |
| jpeg hybrid/Q60 | 0.794529 | 0.794529 | yes |
| jpeg hybrid/Q40 | 0.766344 | 0.766344 | yes |
| jpeg hybrid/Q20 | 0.728706 | 0.728706 | yes |
| jpeg hybrid/Q10 | 0.698656 | 0.698656 | yes |
| freq hybrid/ffpp_source_safe/lowpass_0.1 | 0.414015 | 0.414015 | yes |
| freq hybrid/ffpp_source_safe/lowpass_0.3 | 0.557742 | 0.557742 | yes |
| freq hybrid/ffpp_source_safe/lowpass_0.5 | 0.756472 | 0.756472 | yes |
| freq hybrid/ffpp_source_safe/highpass_0.1 | 0.663263 | 0.663263 | yes |
| freq hybrid/ffpp_source_safe/highpass_0.3 | 0.582490 | 0.582490 | yes |
| freq hybrid/ffpp_source_safe/highpass_0.5 | 0.613812 | 0.613812 | yes |
| freq hybrid/external_normalized/lowpass_0.1 | 0.521267 | 0.521267 | yes |
| freq hybrid/external_normalized/lowpass_0.3 | 0.386236 | 0.386236 | yes |
| freq hybrid/external_normalized/lowpass_0.5 | 0.360460 | 0.360460 | yes |
| freq hybrid/external_normalized/highpass_0.1 | 0.217894 | 0.217894 | yes |
| freq hybrid/external_normalized/highpass_0.3 | 0.260296 | 0.260296 | yes |
| freq hybrid/external_normalized/highpass_0.5 | 0.468003 | 0.468003 | yes |
| matched xception/ffpp_source_safe raw-AUC == master | 0.866003 | 0.866003 | yes |
| matched xception/external_normalized raw-AUC == master | 0.314791 | 0.314791 | yes |
| matched xception/external_original_matched raw-AUC == master | 0.374578 | 0.374578 | yes |
| jpeg xception/Q100 | 0.865263 | 0.865263 | yes |
| jpeg xception/Q90 | 0.848178 | 0.848178 | yes |
| jpeg xception/Q60 | 0.797344 | 0.797344 | yes |
| jpeg xception/Q40 | 0.772030 | 0.772030 | yes |
| jpeg xception/Q20 | 0.692825 | 0.692825 | yes |
| jpeg xception/Q10 | 0.631959 | 0.631959 | yes |
| freq xception/ffpp_source_safe/lowpass_0.1 | 0.423697 | 0.423697 | yes |
| freq xception/ffpp_source_safe/lowpass_0.3 | 0.632485 | 0.632485 | yes |
| freq xception/ffpp_source_safe/lowpass_0.5 | 0.761027 | 0.761027 | yes |
| freq xception/ffpp_source_safe/highpass_0.1 | 0.636369 | 0.636369 | yes |
| freq xception/ffpp_source_safe/highpass_0.3 | 0.480122 | 0.480122 | yes |
| freq xception/ffpp_source_safe/highpass_0.5 | 0.536451 | 0.536451 | yes |
| freq xception/external_normalized/lowpass_0.1 | 0.490801 | 0.490801 | yes |
| freq xception/external_normalized/lowpass_0.3 | 0.407106 | 0.407106 | yes |
| freq xception/external_normalized/lowpass_0.5 | 0.341845 | 0.341845 | yes |
| freq xception/external_normalized/highpass_0.1 | 0.257608 | 0.257608 | yes |
| freq xception/external_normalized/highpass_0.3 | 0.346704 | 0.346704 | yes |
| freq xception/external_normalized/highpass_0.5 | 0.392349 | 0.392349 | yes |
| matched freq_cnn/ffpp_source_safe raw-AUC == master | 0.568521 | 0.568521 | yes |
| matched freq_cnn/external_normalized raw-AUC == master | 0.318239 | 0.318239 | yes |
| matched freq_cnn/external_original_matched raw-AUC == master | 0.456404 | 0.456404 | yes |
| jpeg freq_cnn/Q100 | 0.568213 | 0.568213 | yes |
| jpeg freq_cnn/Q90 | 0.577102 | 0.577102 | yes |
| jpeg freq_cnn/Q60 | 0.565524 | 0.565524 | yes |
| jpeg freq_cnn/Q40 | 0.563141 | 0.563141 | yes |
| jpeg freq_cnn/Q20 | 0.563352 | 0.563352 | yes |
| jpeg freq_cnn/Q10 | 0.570428 | 0.570428 | yes |
| freq freq_cnn/ffpp_source_safe/lowpass_0.1 | 0.437865 | 0.437865 | yes |
| freq freq_cnn/ffpp_source_safe/lowpass_0.3 | 0.472033 | 0.472033 | yes |
| freq freq_cnn/ffpp_source_safe/lowpass_0.5 | 0.511115 | 0.511115 | yes |
| freq freq_cnn/ffpp_source_safe/highpass_0.1 | 0.608634 | 0.608634 | yes |
| freq freq_cnn/ffpp_source_safe/highpass_0.3 | 0.592941 | 0.592941 | yes |
| freq freq_cnn/ffpp_source_safe/highpass_0.5 | 0.579179 | 0.579179 | yes |
| freq freq_cnn/external_normalized/lowpass_0.1 | 0.489083 | 0.489083 | yes |
| freq freq_cnn/external_normalized/lowpass_0.3 | 0.364644 | 0.364644 | yes |
| freq freq_cnn/external_normalized/lowpass_0.5 | 0.336540 | 0.336540 | yes |
| freq freq_cnn/external_normalized/highpass_0.1 | 0.309558 | 0.309558 | yes |
| freq freq_cnn/external_normalized/highpass_0.3 | 0.308196 | 0.308196 | yes |
| freq freq_cnn/external_normalized/highpass_0.5 | 0.307448 | 0.307448 | yes |
| lomo hybrid/Deepfakes unseen | 0.690079 | 0.690079 | yes |
| lomo xception/Deepfakes unseen | 0.688486 | 0.688486 | yes |
| lomo hybrid/Face2Face unseen | 0.577519 | 0.577519 | yes |
| lomo xception/Face2Face unseen | 0.691759 | 0.691759 | yes |
| lomo hybrid/FaceSwap unseen | 0.410046 | 0.410046 | yes |
| lomo xception/FaceSwap unseen | 0.351718 | 0.351718 | yes |
| lomo hybrid/NeuralTextures unseen | 0.463227 | 0.463227 | yes |
| lomo xception/NeuralTextures unseen | 0.580588 | 0.580588 | yes |
| historical hybrid/ffpp_historical | 0.883291 | 0.883291 | yes |
| historical hybrid/external_original | 0.385706 | 0.385706 | yes |
| historical xception/ffpp_historical | 0.861677 | 0.861677 | yes |
| historical xception/external_original | 0.389878 | 0.389878 | yes |
| historical ensemble/ffpp_historical | 0.880514 | 0.880514 | yes |
| historical ensemble/external_original | 0.382294 | 0.382294 | yes |
| recon mse/external_normalized (sign -1) | 0.352105 | 0.352105 | yes |
| recon lpips/external_normalized (sign -1) | 0.681351 | 0.681351 | yes |
| recon ssim/external_normalized (sign +1) | 0.442359 | 0.442359 | yes |
| recon mse/ffpp_source_safe (sign -1) | 0.633819 | 0.633819 | yes |
| recon lpips/ffpp_source_safe (sign -1) | 0.550139 | 0.550139 | yes |
| recon ssim/ffpp_source_safe (sign +1) | 0.607396 | 0.607396 | yes |
