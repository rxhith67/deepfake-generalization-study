# Generalization upgrade: measured results and implementation status

Source-video-disjoint derived protocol approved for all new training. Historical artifacts are unchanged and are NOT retrospectively labelled source/person-identity-safe. Tier 3 and Tier 4 are disabled.

## Historical baseline reproduction

Research question: Can the existing checkpoints reproduce their reported discrimination?

Method/data/models: Fresh inference; original model-specific preprocessing; historical validation thresholds. FF++ test and original external faces. HISTORICAL protocol only (target-video-disjoint): historical checkpoints, never mixed with the source-disjoint rows below. CI units: FF++ rows = target-video cluster bootstrap; external rows = image-level bootstrap; per-method rows have no CI. In this code base "hybrid" = CoAtNet; these are the historical (200-video) checkpoints.

Status: measured outputs available.

| model | test_dataset | test_manipulation | evaluation_unit | condition | auc | auc_ci_lower | auc_ci_upper | balanced_accuracy | ece | nll | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | ffpp_historical | all methods (pooled) | frame | original | 0.883291 | 0.843210 | 0.922283 | 0.784028 | 0.135111 | 0.729486 | 0.141883 |
| hybrid | ffpp_historical | all methods (pooled) | video | original | 0.903889 | 0.860278 | 0.950000 | 0.800000 | 0.086191 | 0.345694 | 0.106556 |
| hybrid | ffpp_historical | Deepfakes | frame | original | 0.913947 | N/A | N/A | 0.807778 | 0.178541 | 1.022099 | 0.186308 |
| hybrid | ffpp_historical | Face2Face | frame | original | 0.902206 | N/A | N/A | 0.803889 | 0.196868 | 1.165840 | 0.203110 |
| hybrid | ffpp_historical | FaceSwap | frame | original | 0.893241 | N/A | N/A | 0.802222 | 0.196779 | 1.142878 | 0.202604 |
| hybrid | ffpp_historical | NeuralTextures | frame | original | 0.823770 | N/A | N/A | 0.722222 | 0.230638 | 1.266729 | 0.245789 |
| xception | ffpp_historical | all methods (pooled) | frame | original | 0.861677 | 0.824483 | 0.902630 | 0.775139 | 0.135163 | 0.608662 | 0.147975 |
| xception | ffpp_historical | all methods (pooled) | video | original | 0.895833 | 0.850556 | 0.942229 | 0.762500 | 0.101326 | 0.348586 | 0.110765 |
| xception | ffpp_historical | Deepfakes | frame | original | 0.893670 | N/A | N/A | 0.821667 | 0.160967 | 0.776465 | 0.169396 |
| xception | ffpp_historical | Face2Face | frame | original | 0.880249 | N/A | N/A | 0.780556 | 0.153959 | 0.779238 | 0.177366 |
| xception | ffpp_historical | FaceSwap | frame | original | 0.847452 | N/A | N/A | 0.762778 | 0.162324 | 0.833337 | 0.189008 |
| xception | ffpp_historical | NeuralTextures | frame | original | 0.825338 | N/A | N/A | 0.735556 | 0.198707 | 0.967059 | 0.221975 |
| ensemble | ffpp_historical | all methods (pooled) | frame | original | 0.880514 | 0.842175 | 0.920705 | 0.784722 | 0.124358 | 0.550085 | 0.135308 |
| hybrid | external_original | all methods (pooled) | frame | original | 0.385706 | 0.338662 | 0.430327 | 0.524306 | 0.464320 | 3.125944 | 0.473872 |
| xception | external_original | all methods (pooled) | frame | original | 0.389878 | 0.343273 | 0.435591 | 0.512153 | 0.427762 | 2.296333 | 0.457849 |
| ensemble | external_original | all methods (pooled) | frame | original | 0.382294 | 0.334231 | 0.427300 | 0.522569 | 0.445640 | 2.562522 | 0.463369 |

Figure: `00_baseline_reproduction/test/roc_reliability_distributions.png`

Interpretation: fresh inference remains close to reported AUCs, while external ordering remains poor. Unexpected detail: small numerical differences occur across environments; they are recorded per sample.

Limitations: Historical target-video protocol has source/donor overlap. Local library versions differ from the old cloud image.

## External preprocessing/source bias

Research question: Does identical preprocessing explain the external failure?

Method/data/models: Raw metadata audit, identical five-landmark similarity alignment, RGB, explicit JPEG Q90; 576 matched images, 288 per class. Historical checkpoints. CIs are image-level bootstrap (external images carry no source-video grouping), NOT source-cluster intervals.

Status: measured outputs available.

| model | test_dataset | evaluation_unit | condition | auc | auc_ci_lower | auc_ci_upper | balanced_accuracy | ece | nll | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | external_original_matched | frame | original_matched | 0.385706 | 0.338662 | 0.430327 | 0.524306 | 0.464320 | 3.125944 | 0.473872 |
| xception | external_original_matched | frame | original_matched | 0.389878 | 0.343273 | 0.435591 | 0.512153 | 0.427762 | 2.296333 | 0.457849 |
| ensemble | external_original_matched | frame | original_matched | 0.382294 | 0.334231 | 0.427300 | 0.522569 | 0.445640 | 2.562522 | 0.463369 |
| hybrid | external_normalized | frame | normalized | 0.379123 | 0.333807 | 0.423357 | 0.487847 | 0.508660 | 3.403156 | 0.512011 |
| xception | external_normalized | frame | normalized | 0.332357 | 0.287651 | 0.378747 | 0.480903 | 0.488834 | 2.293219 | 0.499497 |
| ensemble | external_normalized | frame | normalized | 0.360810 | 0.315160 | 0.403583 | 0.480903 | 0.496599 | 2.706774 | 0.502541 |

Figure: `02_external_bias_audit/original_vs_normalized_auc.png`

Interpretation: normalization did not rescue discrimination; all normalized AUCs remain below 0.5. This contradicts a simple expectation that this normalization alone fixes transfer. It does not establish which source/generation differences cause failure.

Limitations: Alignment changes face geometry relative to training. Class/source/content remain confounded; no causal attribution.

## Calibration and threshold transfer (historical models, historical protocol)

Research question: How does confidence behave under domain shift?

Method/data/models: 15 equal-width probability bins, NLL clipped at 1e-7, Brier; frozen FF++ validation thresholds.

Status: measured outputs available.

| model | test_dataset | evaluation_unit | condition | auc | balanced_accuracy | ece | nll | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | ffpp_historical | frame | original | 0.883291 | 0.784028 | 0.135111 | 0.729486 | 0.141883 |
| xception | ffpp_historical | frame | original | 0.861677 | 0.775139 | 0.135163 | 0.608662 | 0.147975 |
| ensemble | ffpp_historical | frame | original | 0.880514 | 0.784722 | 0.124358 | 0.550085 | 0.135308 |
| hybrid | external_original | frame | original | 0.385706 | 0.524306 | 0.464320 | 3.125944 | 0.473872 |
| xception | external_original | frame | original | 0.389878 | 0.512153 | 0.427762 | 2.296333 | 0.457849 |
| ensemble | external_original | frame | original | 0.382294 | 0.522569 | 0.445640 | 2.562522 | 0.463369 |
| hybrid | external_normalized | frame | normalized | 0.379123 | 0.487847 | 0.508660 | 3.403156 | 0.512011 |
| xception | external_normalized | frame | normalized | 0.332357 | 0.480903 | 0.488834 | 2.293219 | 0.499497 |
| ensemble | external_normalized | frame | normalized | 0.360810 | 0.480903 | 0.496599 | 2.706774 | 0.502541 |

Figure: `03_calibration/reliability_diagrams/hybrid/roc_reliability_distributions.png`

Limitations: ECE depends on binning and prevalence; threshold transfer is not external calibration.

## Matched controls: source-video-disjoint protocol (NEW; not comparable to historical rows above)

Research question: How well do freshly trained, ImageNet-initialized controls discriminate on the stricter split, and how calibrated are they?

Method/data/models: checkpoints trained on the derived source-video-disjoint train split (16,680 images); threshold re-selected on source-disjoint validation only (balanced accuracy) and frozen; test = 1,860 frames, 62 videos (30 real, 32 fake). External sets are transfer evaluations only.

Status: measured outputs available for the models listed.

### FF++ source-disjoint test (95% CI: bootstrap over source-video connected components)

| model | evaluation_unit | num_samples | auc | auc_ci_lower | auc_ci_upper | balanced_accuracy | balanced_accuracy_ci_lower | balanced_accuracy_ci_upper | ece | ece_ci_lower | ece_ci_upper | nll | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RGB+FFT dual-domain CNN | frame | 1860 | 0.568521 | 0.420813 | 0.736329 | 0.536597 | 0.397906 | 0.702619 | 0.061675 | 0.052064 | 0.254837 | 0.680139 | 0.244080 |
| RGB+FFT dual-domain CNN | video | 62 | 0.573958 | 0.420833 | 0.750806 | 0.518750 | 0.362716 | 0.691272 | 0.072869 | 0.049233 | 0.269482 | 0.679254 | 0.243656 |
| CoAtNet | frame | 1860 | 0.845661 | 0.728366 | 0.922942 | 0.746111 | 0.640317 | 0.832136 | 0.180779 | 0.116080 | 0.298497 | 0.820351 | 0.205143 |
| CoAtNet | video | 62 | 0.866667 | 0.735704 | 0.957485 | 0.746875 | 0.626280 | 0.870489 | 0.116095 | 0.097304 | 0.295088 | 0.468221 | 0.150886 |
| Xception | frame | 1860 | 0.866003 | 0.762828 | 0.938063 | 0.782882 | 0.677904 | 0.857562 | 0.133719 | 0.081181 | 0.238757 | 0.637915 | 0.170863 |
| Xception | video | 62 | 0.913542 | 0.797528 | 0.986458 | 0.804167 | 0.681497 | 0.897206 | 0.134486 | 0.099208 | 0.276576 | 0.371518 | 0.122009 |

Alternative cluster definition (resampling target-video groups): 

| model | evaluation_unit | n_clusters_target_group | auc_ci_lower_target_group | auc_ci_upper_target_group | n_clusters_source_component |
| --- | --- | --- | --- | --- | --- |
| RGB+FFT dual-domain CNN | frame | 30.000000 | 0.424499 | 0.720184 | 26.000000 |
| RGB+FFT dual-domain CNN | video | 30.000000 | 0.421294 | 0.742862 | 26.000000 |
| CoAtNet | frame | 30.000000 | 0.746116 | 0.914159 | 26.000000 |
| CoAtNet | video | 30.000000 | 0.752743 | 0.947619 | 26.000000 |
| Xception | frame | 30.000000 | 0.782885 | 0.929891 | 26.000000 |
| Xception | video | 30.000000 | 0.822493 | 0.978337 | 26.000000 |

### Per-method test AUC (real vs one manipulation; 8 fake videos per method)

| model | test_manipulation | auc | auc_ci_lower | auc_ci_upper | balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| RGB+FFT dual-domain CNN | Deepfakes | 0.568296 | 0.420262 | 0.712393 | 0.527222 |
| RGB+FFT dual-domain CNN | Face2Face | 0.543523 | 0.394621 | 0.718546 | 0.535556 |
| RGB+FFT dual-domain CNN | FaceSwap | 0.542167 | 0.381616 | 0.735378 | 0.504306 |
| RGB+FFT dual-domain CNN | NeuralTextures | 0.620097 | 0.479780 | 0.784573 | 0.579306 |
| CoAtNet | Deepfakes | 0.809130 | 0.620498 | 0.960872 | 0.687778 |
| CoAtNet | Face2Face | 0.941634 | 0.854854 | 0.982541 | 0.860694 |
| CoAtNet | FaceSwap | 0.827407 | 0.679293 | 0.913980 | 0.735694 |
| CoAtNet | NeuralTextures | 0.804475 | 0.672736 | 0.887640 | 0.700278 |
| Xception | Deepfakes | 0.871028 | 0.726795 | 0.967879 | 0.780278 |
| Xception | Face2Face | 0.930458 | 0.864069 | 0.974377 | 0.855278 |
| Xception | FaceSwap | 0.838875 | 0.722911 | 0.928920 | 0.746944 |
| Xception | NeuralTextures | 0.823653 | 0.692059 | 0.924191 | 0.749028 |

### External transfer (frozen FF++-validation threshold; image-level bootstrap)

| model | test_dataset | auc | auc_ci_lower | auc_ci_upper | balanced_accuracy | ece | nll | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RGB+FFT dual-domain CNN | external_original_matched | 0.456404 | 0.410060 | 0.500796 | 0.472222 | 0.201046 | 0.815818 | 0.300185 |
| RGB+FFT dual-domain CNN | external_normalized | 0.318239 | 0.276481 | 0.360891 | 0.378472 | 0.255916 | 0.843773 | 0.315583 |
| CoAtNet | external_original_matched | 0.400680 | 0.356062 | 0.446628 | 0.506944 | 0.451235 | 2.809881 | 0.464940 |
| CoAtNet | external_normalized | 0.366741 | 0.320698 | 0.411489 | 0.493056 | 0.475900 | 3.178099 | 0.493367 |
| Xception | external_original_matched | 0.374578 | 0.328296 | 0.418812 | 0.487847 | 0.404005 | 2.389380 | 0.445584 |
| Xception | external_normalized | 0.314791 | 0.271070 | 0.360226 | 0.453125 | 0.453910 | 2.472135 | 0.477830 |

RGB+FFT dual-domain CNN (random initialization; trained fresh under this protocol, unrelated to the contaminated historical RGB+FFT checkpoint): validation AUC 0.8228 (epoch-2 best, early-stopped at epoch 5) but source-video-disjoint test AUC 0.5685, 95% CI 0.421-0.736 from source-component bootstrap (26 clusters). The interval includes 0.5, so reliable test discrimination is NOT shown; this does not imply that frequency information is ineffective in general (single seed, small model, small test set).

Interpretation: the wide intervals reflect only 26 independent source-video components (30 target groups), so small AUC differences between controls should not be read as ranking. External AUCs below 0.5 are reported as measured and are not inverted. Interval units: FF++ test rows use source-component cluster resampling; external rows use image-level resampling (external images have no source grouping).

Limitations: single seed; person identity disjointness is not established; frames within a video are not independent; external real/fake source differences remain after normalization. Files: `01_matched_controls/<model>/{evaluation,calibration,predictions}`.

### JPEG robustness (frozen clean thresholds; FF++ source-disjoint test; paired cluster-bootstrap CI on the change)

| model | test_dataset | condition | auc | auc_ci_lower | auc_ci_upper | auc_change | auc_change_ci_lower | auc_change_ci_upper | balanced_accuracy | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAtNet | ffpp_source_safe | reference | 0.845661 | 0.728366 | 0.922942 | 0.000000 | 0.000000 | 0.000000 | 0.746111 | 0.180779 |
| CoAtNet | ffpp_source_safe | Q100 | 0.848995 | 0.739003 | 0.924168 | 0.003333 | 0.000223 | 0.007805 | 0.747326 | 0.180049 |
| CoAtNet | ffpp_source_safe | Q90 | 0.826657 | 0.714755 | 0.903276 | -0.019004 | -0.034066 | 0.000838 | 0.728958 | 0.197328 |
| CoAtNet | ffpp_source_safe | Q60 | 0.794529 | 0.676970 | 0.883556 | -0.051133 | -0.073871 | -0.026158 | 0.700417 | 0.200496 |
| CoAtNet | ffpp_source_safe | Q40 | 0.766344 | 0.625485 | 0.868857 | -0.079318 | -0.116856 | -0.041723 | 0.677743 | 0.212765 |
| CoAtNet | ffpp_source_safe | Q20 | 0.728706 | 0.604991 | 0.825097 | -0.116955 | -0.151950 | -0.081334 | 0.626146 | 0.221424 |
| CoAtNet | ffpp_source_safe | Q10 | 0.698656 | 0.594360 | 0.800130 | -0.147005 | -0.216523 | -0.057567 | 0.569861 | 0.269266 |
| Xception | ffpp_source_safe | reference | 0.866003 | 0.762828 | 0.938063 | 0.000000 | 0.000000 | 0.000000 | 0.782882 | 0.133719 |
| Xception | ffpp_source_safe | Q100 | 0.865263 | 0.761791 | 0.936994 | -0.000741 | -0.003331 | 0.002006 | 0.779653 | 0.139177 |
| Xception | ffpp_source_safe | Q90 | 0.848178 | 0.740618 | 0.920720 | -0.017826 | -0.033332 | -0.003710 | 0.750069 | 0.149614 |
| Xception | ffpp_source_safe | Q60 | 0.797344 | 0.691551 | 0.883172 | -0.068660 | -0.096332 | -0.044210 | 0.716111 | 0.170904 |
| Xception | ffpp_source_safe | Q40 | 0.772030 | 0.669532 | 0.855069 | -0.093973 | -0.128453 | -0.063362 | 0.686875 | 0.188882 |
| Xception | ffpp_source_safe | Q20 | 0.692825 | 0.598342 | 0.771946 | -0.173178 | -0.236625 | -0.111497 | 0.620035 | 0.248888 |
| Xception | ffpp_source_safe | Q10 | 0.631959 | 0.538456 | 0.728177 | -0.234044 | -0.360761 | -0.105302 | 0.530903 | 0.367002 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | reference | 0.568521 | 0.420813 | 0.736329 | 0.000000 | 0.000000 | 0.000000 | 0.536597 | 0.061675 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | Q100 | 0.568213 | 0.420886 | 0.735502 | -0.000308 | -0.001502 | 0.000378 | 0.533056 | 0.061611 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | Q90 | 0.577102 | 0.430533 | 0.742448 | 0.008581 | -0.000253 | 0.020310 | 0.544549 | 0.066821 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | Q60 | 0.565524 | 0.417987 | 0.725201 | -0.002997 | -0.016485 | 0.006083 | 0.527361 | 0.059036 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | Q40 | 0.563141 | 0.416850 | 0.717075 | -0.005380 | -0.028728 | 0.012923 | 0.529375 | 0.061552 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | Q20 | 0.563352 | 0.419517 | 0.710207 | -0.005169 | -0.042360 | 0.027195 | 0.526111 | 0.112667 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | Q10 | 0.570428 | 0.447731 | 0.701377 | 0.001907 | -0.055104 | 0.051198 | 0.466771 | 0.119320 |

Figures: `04_jpeg_robustness/*_curves.png`. Raw predictions: `04_jpeg_robustness/predictions/`. `reference` = no extra compression/filter; auc_change = AUC minus reference AUC (negative = degradation); CIs on the change are paired cluster bootstraps. Threshold-dependent metrics use each model's frozen clean validation threshold (never re-tuned per condition).

Condition accounting: 7 conditions per model (reference, explicit Q100 re-encode, Q90, Q60, Q40, Q20, Q10) = 21 metric rows, but only 18 newly generated prediction files. The three `reference` rows reuse the clean matched-control predictions (`01_matched_controls/<model>/predictions/`); nothing is missing. Reference and Q100 are separate conditions.

Observed pattern (FF++ source-disjoint test, AUC reference -> Q10): CoAtNet 0.846 -> 0.699 (change -0.147); Xception 0.866 -> 0.632 (change -0.234). For both, AUC declines monotonically from Q100 to Q10, and an explicit Q100 re-encode changes AUC by at most 0.0034 (CoAtNet +0.0033, Xception -0.0007). Absolute AUC intervals overlap between CoAtNet and Xception at every quality, so neither is called more robust. The RGB+FFT dual-domain CNN stays near 0.57 across all conditions; because its reference AUC is already weak and its interval includes 0.5, this flat curve is a null/uninformative result, NOT evidence of compression robustness.

### Low/high-pass sensitivity (intervention, not proof of a universal artifact band)

| model | test_dataset | condition | auc | auc_ci_lower | auc_ci_upper | auc_change | auc_change_ci_lower | auc_change_ci_upper | balanced_accuracy | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAtNet | ffpp_source_safe | reference | 0.845661 | 0.728366 | 0.922942 | 0.000000 | 0.000000 | 0.000000 | 0.746111 | 0.180779 |
| CoAtNet | ffpp_source_safe | lowpass_0.1 | 0.414015 | 0.248761 | 0.546470 | -0.431646 | -0.552663 | -0.325537 | 0.464896 | 0.413336 |
| CoAtNet | ffpp_source_safe | lowpass_0.3 | 0.557742 | 0.335025 | 0.773679 | -0.287920 | -0.410382 | -0.145154 | 0.542778 | 0.344147 |
| CoAtNet | ffpp_source_safe | lowpass_0.5 | 0.756472 | 0.613835 | 0.873682 | -0.089189 | -0.137114 | -0.042438 | 0.668472 | 0.216722 |
| CoAtNet | ffpp_source_safe | highpass_0.1 | 0.663263 | 0.553166 | 0.763101 | -0.182399 | -0.299939 | -0.059541 | 0.499965 | 0.486816 |
| CoAtNet | ffpp_source_safe | highpass_0.3 | 0.582490 | 0.349643 | 0.808540 | -0.263172 | -0.424312 | -0.047059 | 0.500000 | 0.232714 |
| CoAtNet | ffpp_source_safe | highpass_0.5 | 0.613812 | 0.485138 | 0.763267 | -0.231849 | -0.403052 | 0.005509 | 0.500000 | 0.148352 |
| CoAtNet | external_normalized | reference | 0.366741 | 0.320698 | 0.411489 | 0.000000 | 0.000000 | 0.000000 | 0.493056 | 0.475900 |
| CoAtNet | external_normalized | lowpass_0.1 | 0.521267 | 0.474674 | 0.571026 | 0.154526 | 0.094337 | 0.212889 | 0.534722 | 0.299479 |
| CoAtNet | external_normalized | lowpass_0.3 | 0.386236 | 0.342113 | 0.432907 | 0.019495 | -0.011552 | 0.051726 | 0.487847 | 0.458156 |
| CoAtNet | external_normalized | lowpass_0.5 | 0.360460 | 0.316386 | 0.404826 | -0.006281 | -0.025868 | 0.014120 | 0.472222 | 0.489042 |
| CoAtNet | external_normalized | highpass_0.1 | 0.217894 | 0.180743 | 0.255681 | -0.148847 | -0.189806 | -0.108261 | 0.501736 | 0.513434 |
| CoAtNet | external_normalized | highpass_0.3 | 0.260296 | 0.221842 | 0.300337 | -0.106445 | -0.163544 | -0.048813 | 0.500000 | 0.319767 |
| CoAtNet | external_normalized | highpass_0.5 | 0.468003 | 0.422268 | 0.515127 | 0.101261 | 0.028200 | 0.173966 | 0.500000 | 0.139951 |
| Xception | ffpp_source_safe | reference | 0.866003 | 0.762828 | 0.938063 | 0.000000 | 0.000000 | 0.000000 | 0.782882 | 0.133719 |
| Xception | ffpp_source_safe | lowpass_0.1 | 0.423697 | 0.321451 | 0.510898 | -0.442307 | -0.556584 | -0.321184 | 0.468056 | 0.222068 |
| Xception | ffpp_source_safe | lowpass_0.3 | 0.632485 | 0.538931 | 0.727050 | -0.233519 | -0.299842 | -0.171910 | 0.596493 | 0.211631 |
| Xception | ffpp_source_safe | lowpass_0.5 | 0.761027 | 0.611931 | 0.890521 | -0.104977 | -0.164847 | -0.039261 | 0.669167 | 0.217246 |
| Xception | ffpp_source_safe | highpass_0.1 | 0.636369 | 0.422685 | 0.871851 | -0.229634 | -0.495152 | 0.066259 | 0.569514 | 0.345887 |
| Xception | ffpp_source_safe | highpass_0.3 | 0.480122 | 0.295104 | 0.709273 | -0.385882 | -0.602781 | -0.121533 | 0.469861 | 0.119958 |
| Xception | ffpp_source_safe | highpass_0.5 | 0.536451 | 0.272485 | 0.861629 | -0.329552 | -0.639863 | 0.023768 | 0.496111 | 0.137853 |
| Xception | external_normalized | reference | 0.314791 | 0.271070 | 0.360226 | 0.000000 | 0.000000 | 0.000000 | 0.453125 | 0.453910 |
| Xception | external_normalized | lowpass_0.1 | 0.490801 | 0.446728 | 0.538767 | 0.176010 | 0.119980 | 0.231360 | 0.482639 | 0.219652 |
| Xception | external_normalized | lowpass_0.3 | 0.407106 | 0.361804 | 0.453582 | 0.092315 | 0.055182 | 0.131206 | 0.463542 | 0.342009 |
| Xception | external_normalized | lowpass_0.5 | 0.341845 | 0.298329 | 0.386528 | 0.027054 | 0.005778 | 0.048774 | 0.440972 | 0.430017 |
| Xception | external_normalized | highpass_0.1 | 0.257608 | 0.217817 | 0.298182 | -0.057183 | -0.097469 | -0.020707 | 0.321181 | 0.472532 |
| Xception | external_normalized | highpass_0.3 | 0.346704 | 0.303186 | 0.392001 | 0.031913 | -0.018307 | 0.083436 | 0.394097 | 0.175863 |
| Xception | external_normalized | highpass_0.5 | 0.392349 | 0.346852 | 0.440241 | 0.077558 | 0.021836 | 0.136151 | 0.453125 | 0.142401 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | reference | 0.568521 | 0.420813 | 0.736329 | 0.000000 | 0.000000 | 0.000000 | 0.536597 | 0.061675 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | lowpass_0.1 | 0.437865 | 0.284482 | 0.587917 | -0.130656 | -0.362606 | 0.090464 | 0.500000 | 0.112960 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | lowpass_0.3 | 0.472033 | 0.303838 | 0.625993 | -0.096488 | -0.253451 | 0.038289 | 0.532778 | 0.103670 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | lowpass_0.5 | 0.511115 | 0.361678 | 0.662881 | -0.057406 | -0.146658 | 0.017362 | 0.538264 | 0.081979 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | highpass_0.1 | 0.608634 | 0.430723 | 0.827150 | 0.040113 | -0.079161 | 0.174016 | 0.500000 | 0.297301 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | highpass_0.3 | 0.592941 | 0.406002 | 0.829487 | 0.024420 | -0.089941 | 0.154498 | 0.500000 | 0.282658 |
| RGB+FFT dual-domain CNN | ffpp_source_safe | highpass_0.5 | 0.579179 | 0.371041 | 0.830360 | 0.010659 | -0.140812 | 0.167337 | 0.500000 | 0.247760 |
| RGB+FFT dual-domain CNN | external_normalized | reference | 0.318239 | 0.276481 | 0.360891 | 0.000000 | 0.000000 | 0.000000 | 0.378472 | 0.255916 |
| RGB+FFT dual-domain CNN | external_normalized | lowpass_0.1 | 0.489083 | 0.442316 | 0.536855 | 0.170844 | 0.119316 | 0.224101 | 0.498264 | 0.134459 |
| RGB+FFT dual-domain CNN | external_normalized | lowpass_0.3 | 0.364644 | 0.321435 | 0.409026 | 0.046405 | 0.013848 | 0.081146 | 0.446181 | 0.189235 |
| RGB+FFT dual-domain CNN | external_normalized | lowpass_0.5 | 0.336540 | 0.293223 | 0.380159 | 0.018302 | 0.002406 | 0.035181 | 0.387153 | 0.199881 |
| RGB+FFT dual-domain CNN | external_normalized | highpass_0.1 | 0.309558 | 0.265505 | 0.352364 | -0.008681 | -0.036917 | 0.018773 | 0.500000 | 0.345562 |
| RGB+FFT dual-domain CNN | external_normalized | highpass_0.3 | 0.308196 | 0.264678 | 0.351245 | -0.010043 | -0.038123 | 0.017794 | 0.500000 | 0.332860 |
| RGB+FFT dual-domain CNN | external_normalized | highpass_0.5 | 0.307448 | 0.264032 | 0.350167 | -0.010790 | -0.039349 | 0.017362 | 0.500000 | 0.302337 |

Figures: `07_frequency_sensitivity/*_curves.png`. Raw predictions: `07_frequency_sensitivity/predictions/`. `reference` = no extra compression/filter; auc_change = AUC minus reference AUC (negative = degradation); CIs on the change are paired cluster bootstraps. Threshold-dependent metrics use each model's frozen clean validation threshold (never re-tuned per condition).

Interpretation limits: these are input interventions on fixed, non-retrained models. Changes show sensitivity to information retained or removed by the filter, not that any band carries a universal deepfake signature. High-pass conditions frequently give balanced accuracy 0.5000, meaning the frozen threshold assigns nearly every filtered image to one class (a shift of the score distribution), so threshold-dependent metrics there mostly reflect calibration shift. External AUCs below 0.5 are reported as measured (not inverted); the external rise toward ~0.5 under strong low-pass is not evidence of better detection.

## Frequency spectrum analysis (descriptive; no model involved)

Research question: Do real and manipulated/generated images differ in average spatial-frequency energy?

Method/data: fixed 240-image sample per group (method-balanced, video-spread, seeded): four FF++ manipulations and FF++ real from the source-disjoint test manifest (all 1,200 sampled FF++ images verified to lie in that test set), and 240 real + 240 Stable Diffusion faces from the normalized external cohort. Mean-subtracted grayscale FFT, no window, DC excluded; energy ratios in fixed radius bands (0-0.1, 0.1-0.3, >0.3 of axis Nyquist). Each fake group is compared only with the real group of its own domain.

| group | low_energy_ratio | mid_energy_ratio | high_energy_ratio |
| --- | --- | --- | --- |
| External fake | 0.894949 | 0.082892 | 0.022159 |
| External real | 0.908306 | 0.072049 | 0.019646 |
| FF++ Deepfakes | 0.902660 | 0.080646 | 0.016695 |
| FF++ Face2Face | 0.906296 | 0.077036 | 0.016668 |
| FF++ FaceSwap | 0.899954 | 0.082354 | 0.017692 |
| FF++ NeuralTextures | 0.906707 | 0.078348 | 0.014945 |
| FF++ real | 0.886643 | 0.092254 | 0.021103 |

Figures: `06_frequency_analysis/{mean_spectra_and_differences,radial_power}.png`; per-image values in `spectral_energy_per_image.csv`; parameters in `parameters.json`.

Reading: band-energy ratios differ by at most about 0.01-0.02 between real and fake groups and no significance test or interval was computed, so these are descriptive patterns only. They are not proof of a universal deepfake frequency signature and not a causal explanation of any model result; external real and generated images also differ in source, resolution history and JPEG history (see Tier 1 bias audit).

## Leave-one-manipulation-out (LOMO), source-video-disjoint protocol

Research question: How much AUC is lost on a manipulation absent from training AND validation?

Method/data/models: same architectures/initializations as the matched controls; each run verified (checkpoint SHA-256, manifest hashes, held-out method absent from train and val, validation AUC vs best training-log epoch within 1e-3). Threshold frozen from LOMO validation only. Held-out cohort = real + held-out method from the shared test manifest; the comparison control is the matched control on the identical cohort.

Status: 8 of 8 runs evaluated so far (CoAtNet/Deepfakes, Xception/Deepfakes, CoAtNet/Face2Face, Xception/Face2Face, CoAtNet/FaceSwap, Xception/FaceSwap, CoAtNet/NeuralTextures, Xception/NeuralTextures).

### Unseen (held-out) manipulation vs control on the identical cohort

| held_out | model | auc | auc_ci_lower | auc_ci_upper | balanced_accuracy | control_auc_same_cohort | gap_control_minus_lomo | gap_ci_lower | gap_ci_upper | n_clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deepfakes | CoAtNet | 0.690079 | 0.444903 | 0.892465 | 0.609306 | 0.809130 | 0.119051 | 0.059950 | 0.195322 | 26 |
| Deepfakes | Xception | 0.688486 | 0.453827 | 0.882187 | 0.607639 | 0.871028 | 0.182542 | 0.083334 | 0.293520 | 26 |
| Face2Face | CoAtNet | 0.577519 | 0.438503 | 0.710418 | 0.493889 | 0.941634 | 0.364116 | 0.264940 | 0.465075 | 26 |
| Face2Face | Xception | 0.691759 | 0.568706 | 0.797396 | 0.576250 | 0.930458 | 0.238699 | 0.157947 | 0.324434 | 26 |
| FaceSwap | CoAtNet | 0.410046 | 0.229066 | 0.554980 | 0.406111 | 0.827407 | 0.417361 | 0.326650 | 0.506044 | 26 |
| FaceSwap | Xception | 0.351718 | 0.169043 | 0.503517 | 0.380972 | 0.838875 | 0.487157 | 0.370008 | 0.602116 | 26 |
| NeuralTextures | CoAtNet | 0.463227 | 0.369734 | 0.560519 | 0.467500 | 0.804475 | 0.341248 | 0.211301 | 0.435666 | 26 |
| NeuralTextures | Xception | 0.580588 | 0.498190 | 0.659490 | 0.536944 | 0.823653 | 0.243065 | 0.140243 | 0.337942 | 26 |

95% CIs: source-video-component bootstrap (gap CI is paired). Only 8 fake videos per method in the test set; intervals are wide.

### Known (seen) manipulations for the same LOMO models (mean of the 3 seen-method AUCs)

| held_out | model | known_methods_mean_auc | control_same_cohorts_mean_auc |
| --- | --- | --- | --- |
| Deepfakes | CoAtNet | 0.877100 | 0.857839 |
| Deepfakes | Xception | 0.838992 | 0.864329 |
| Face2Face | CoAtNet | 0.835414 | 0.813671 |
| Face2Face | Xception | 0.857296 | 0.844519 |
| FaceSwap | CoAtNet | 0.865205 | 0.851746 |
| FaceSwap | Xception | 0.870526 | 0.875046 |
| NeuralTextures | CoAtNet | 0.889151 | 0.859390 |
| NeuralTextures | Xception | 0.945972 | 0.880120 |

### Macro summary across the four held-out manipulations

| model | statistic | estimate | ci_lower | ci_upper | valid_replicates | n_clusters |
| --- | --- | --- | --- | --- | --- | --- |
| CoAtNet | macro_unseen_auc | 0.535218 | 0.394200 | 0.642531 | 1967 | 26 |
| CoAtNet | macro_control_auc_same_cohorts | 0.845661 | 0.728366 | 0.922942 | 1967 | 26 |
| CoAtNet | macro_gap_control_minus_lomo | 0.310444 | 0.254293 | 0.364281 | 1967 | 26 |
| Xception | macro_unseen_auc | 0.578138 | 0.454139 | 0.678855 | 1967 | 26 |
| Xception | macro_control_auc_same_cohorts | 0.866003 | 0.762828 | 0.938063 | 1967 | 26 |
| Xception | macro_gap_control_minus_lomo | 0.287866 | 0.243470 | 0.330658 | 1967 | 26 |

Macro CIs: each bootstrap replicate resamples the 26 source-video connected components once and applies that resample to all four held-out cohorts and to the matched control, so macro gaps are paired. Each cohort has only 8 fake videos (~30 real videos), so intervals are wide; CoAtNet and Xception macro intervals overlap heavily and no model ranking is made. Unseen AUCs below 0.5 (FaceSwap for both models, NeuralTextures for CoAtNet with CI reaching 0.56) mean the held-out fakes were scored as MORE real than real faces; they are reported as measured and not inverted.

Figures: `05_lomo/summary/{unseen_auc_heatmap,cross_manipulation_heatmaps,known_vs_unseen,paired_gaps}.png`; tables: `05_lomo/summary/{cross_manipulation,leave_one_out_summary,macro_summary}.csv`.

Limitations: single seed; person identity disjointness not established; 8 fake videos per method. Files: `05_lomo/lomo_<method>/<model>/{predictions_test_all.csv,evaluation/,confidence_distribution.png}`.

## Representation analysis (exploratory; new source-disjoint controls)

Method: penultimate (classifier-input) features for 1,680 sampled test images (FF++ source-disjoint test plus 480 normalized external), one joint PCA/t-SNE per model. PCA is centered only (original units). Silhouette scores use Euclidean distance in the ORIGINAL feature space, never on t-SNE coordinates. Effective rank = exp(entropy of normalized covariance eigenvalues).

| model | samples | dimensions | effective_rank | PCs_90% | PCs_95% | PCs_99% | silhouette_real_vs_fake | silhouette_within_FF++ | silhouette_within_external | silhouette_domain | silhouette_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAtNet | 1680 | 768 | 52.379005 | 150 | 253 | 481 | 0.060209 | 0.110865 | 0.027360 | 0.110698 | 0.004893 |
| Xception | 1680 | 2048 | 99.475189 | 236 | 405 | 829 | 0.059382 | 0.108939 | 0.025868 | 0.106075 | -0.003314 |
| RGB+FFT dual-domain CNN | 1680 | 64 | 1.537894 | 1 | 2 | 7 | 0.109867 | 0.188513 | 0.069366 | 0.281527 | -0.155487 |

Reading: all silhouette values are low (<0.3), i.e. real/fake classes are not compact, well-separated clusters in these features, and domain (FF++ vs external) separates more than real/fake for the two pretrained models. In t-SNE, FF++ frames form tight per-video clusters, so visible clusters mostly reflect video/identity rather than class. The RGB+FFT dual-domain CNN feature space is nearly one-dimensional (effective rank about 1.5; one PC explains 90% of variance), an observation that coincides with its weak test AUC; no causal link between low effective rank and poor discrimination is established (single seed, no ablation), but silhouette values there are not comparable across models with different dimensionality. t-SNE is exploratory: no distances or cluster sizes in the plots are treated as evidence. Files: `09_representation_analysis/<model>/`.

## Grad-CAM failure analysis (qualitative; Xception, fake-logit target)

Panels: external_normalized, ffpp_test, lomo_Deepfakes, lomo_Face2Face, lomo_FaceSwap, lomo_NeuralTextures. Each has up to 2 TP, TN, FP and FN cases from distinct source videos (first eligible frames in manifest order; not hand-picked), using the frozen validation threshold. LOMO panels use each fold's own LOMO checkpoint on the held-out cohort. CoAtNet is not shown because its final stage is attention, not a convolutional map. No region masks exist, so no quantitative or anatomical claims are made. Observation (unquantified): several normalized-external images contain flat-colored padding bands from the alignment warp, including some FP/FN cases, so alignment artifacts may confound those panels. Files: `10_gradcam_failure_analysis/xception/<condition>/gradcam.png`.

## Tier 2: VAE round-trip reconstruction scores (AEROBLADE-style; single autoencoder; NOT a full AEROBLADE reproduction)

Provenance and pre-registered score orientation: `11_reconstruction_detection/PROVENANCE_AND_PROTOCOL.md` (written before any score existed).

**Verified:** the dataset card (`desingh/DeepFakeFace`, `OpenRL/DeepFakeFace`) states text2img is "generated by Stable Diffusion V1.5"; the paper (arXiv:2309.02218) gives 512x512 images and the prompt template "name, celebrity, age" (read via text extraction of the HTML version); local metadata confirm all 311 raw fakes are 512x512 JPEGs. **Not established:** the exact SD checkpoint, the exact VAE/decoder used at generation, sampler settings, or JPEG re-encoding history. **Compatibility assumption (not a fact):** the reconstruction VAE is the `vae/` of `stable-diffusion-v1-5/stable-diffusion-v1-5` pinned at commit `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`; it is not claimed to be the original generator VAE, so a null/negative result cannot be attributed to the method itself.

Method: normalized 224px crop -> bicubic 256 -> VAE posterior mode -> decode (float32, deterministic; identical for both classes). MSE, SSIM, standard AlexNet LPIPS. Fake-oriented scores fixed in advance: -MSE, +SSIM, -LPIPS. AUCs below 0.5 are reported as measured, never inverted. These are anomaly scores; no ECE/NLL/threshold. Smoke test (16 images) passed: shapes/ranges, finite, bit-identical repeat, class-blind input format.

### External normalized cohort (576 images, 288 per class; image-level bootstrap 95% CI)

| model | score_orientation | auc | auc_ci_lower | auc_ci_upper | num_samples |
| --- | --- | --- | --- | --- | --- |
| SD15_VAE_mse | -MSE | 0.352105 | 0.306401 | 0.396177 | 576 |
| SD15_VAE_ssim | +SSIM | 0.442359 | 0.393907 | 0.489292 | 576 |
| SD15_VAE_lpips | -LPIPS | 0.681351 | 0.635721 | 0.722522 | 576 |

### Raw metric statistics by class

| metric | class | n | mean | std | median | q25 | q75 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mse | real | 288 | 0.000752 | 0.001897 | 0.000443 | 0.000250 | 0.000875 | 0.000067 | 0.030932 |
| mse | generated | 288 | 0.000885 | 0.000762 | 0.000652 | 0.000424 | 0.001111 | 0.000103 | 0.006598 |
| ssim | real | 288 | 0.905613 | 0.075057 | 0.920213 | 0.872957 | 0.956386 | 0.241319 | 0.993162 |
| ssim | generated | 288 | 0.901405 | 0.051600 | 0.912125 | 0.870212 | 0.939547 | 0.703640 | 0.983984 |
| lpips | real | 288 | 0.035152 | 0.022202 | 0.033601 | 0.026147 | 0.039942 | 0.005689 | 0.347135 |
| lpips | generated | 288 | 0.027281 | 0.008529 | 0.026384 | 0.020866 | 0.031996 | 0.011785 | 0.060784 |

### Existing external classifier results on the same cohort (matched source-disjoint controls; frozen thresholds, image-level CI)

| model | auc | auc_ci_lower | auc_ci_upper |
| --- | --- | --- | --- |
| CoAtNet | 0.366741 | 0.320698 | 0.411489 |
| Xception | 0.314791 | 0.271070 | 0.360226 |
| RGB+FFT dual-domain CNN | 0.318239 | 0.276481 | 0.360891 |

Reading (uncertainty-aware): the pre-registered orientation gives AUC 0.352 for -MSE (CI 0.306-0.396) and 0.442 for +SSIM (CI 0.394-0.489), i.e. generated images have HIGHER MSE and slightly lower SSIM than real ones, the opposite of the hypothesis; -LPIPS gives 0.681 (CI 0.636-0.723), in the hypothesized direction. The three metrics therefore disagree, which argues against reading any single value as detection. The -MSE interval overlaps the classifiers' intervals (about 0.27-0.41), so it is not distinguishable from them; the -LPIPS interval lies above 0.5 and does not overlap them, so on this cohort its ordering differs from the classifiers'. This does not show the reconstruction method is a better detector, and it is not evidence of a general diffusion-image detector. Reconstruction scores are uncalibrated anomaly scores, not probabilities: real and fake differ in resolution history (median face crop 88 vs 159 px, so real faces are upsampled more), alignment padding (some fakes contain large flat bands that trivially reconstruct; one "real" example is a non-face MTCNN detection), JPEG history, and the VAE is only an assumed match. No causal attribution is made.

### Secondary diagnostic: FF++ source-disjoint subset (600 frames = 120 per label/method group; CI over source-video components)

| model | score_orientation | auc | auc_ci_lower | auc_ci_upper | num_samples |
| --- | --- | --- | --- | --- | --- |
| SD15_VAE_mse | -MSE | 0.633819 | 0.484666 | 0.832946 | 600 |
| SD15_VAE_ssim | +SSIM | 0.607396 | 0.390263 | 0.841121 | 600 |
| SD15_VAE_lpips | -LPIPS | 0.550139 | 0.398627 | 0.734691 | 600 |

All pooled FF++ intervals include 0.5, so no discrimination is shown. This subset is not the method's intended setting (face-swap/reenactment are not latent-diffusion outputs) and does not alter the external interpretation. Per-method values: `11_reconstruction_detection/ffpp/metrics.csv`.

Files: `11_reconstruction_detection/{external,ffpp,smoke}/` (scores, descriptive statistics, ROC/distribution plots, original/reconstruction/error-map examples).

## Remaining required analyses

Completed and recorded above: baseline reproduction, external bias audit, historical calibration, spectrum analysis, three matched controls with calibration, standardized JPEG, low/high-pass sensitivity, and all 8 LOMO runs with macro summary, representation analysis, and Grad-CAM. Tier 1 is substantially complete and Tier 2 (VAE reconstruction) is complete for the external cohort and the FF++ secondary subset. Tier 3/4 are disabled. Not done: Grad-CAM for CoAtNet/RGB+FFT, LOMO-model embeddings, and a VAE with established generator provenance.

## Reproducibility

See `protocol/summary.json`, experiment `run.json` files, prediction `.meta.json` files, `master_results.csv`, `test_report.txt` and `IMPLEMENTATION_PLAN.md`. Confidence-interval units: historical FF++ = target-video clusters; new source-disjoint FF++ (controls, JPEG, filtering, LOMO, FF++ reconstruction subset) = source-video connected components (26 clusters in the test set; target-video groups reported as a sensitivity check); all external sets (historical and new, classifiers and reconstruction) = image-level bootstrap. Reconstruction scores are anomaly scores, never calibrated probabilities. `INTEGRITY_REPORT.md` lists the raw-artifact recomputation of headline numbers.
