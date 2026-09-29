# JPEG robustness accounting

7 conditions x 3 models = 21 metric rows in metrics.csv. predictions/ holds 18 files (3 models x 6 explicit re-encodes: Q100,Q90,Q60,Q40,Q20,Q10). The 'reference' condition (no additional compression) reuses the clean matched-control predictions in 01_matched_controls/<model>/predictions/<model>_test_predictions.csv, so no reference files are duplicated here. Reference and explicit Q100 are separate conditions.

