# External source/preprocessing audit

Original and normalized comparisons use identical retained sample IDs. Both classes use the same five-landmark similarity alignment and explicit JPEG Q90. This differs from historical bounding-box-only crops. Source/content confounding remains; a change in AUC is not proof of a single causal explanation. Raw metadata precedes normalization. See normalization_complete.json for retention and external_metadata.csv for detection failures.

condition  normalized  original_matched
model                                  
ensemble     0.360810          0.382294
hybrid       0.379123          0.385706
xception     0.332357          0.389878
