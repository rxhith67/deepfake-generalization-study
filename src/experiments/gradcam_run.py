"""Grad-CAM failure panels (fake-logit target) for Xception: normal FF++, each LOMO fold, normalized external.

Cases are the first frames of distinct source videos in each TP/TN/FP/FN category (deterministic; no visual cherry-picking).
Panels are qualitative only: no region masks exist, so no anatomical/quantitative attribution claims are made.
"""
import json
from pathlib import Path

import pandas as pd

from src.experiments.common import configuration
from src.experiments.interpretation import cam_panels
from src.experiments.manifests import METHODS


def main(model="xception", only=None):
    config = configuration()
    root = Path(config["output_dir"])
    checkpoint = root / "01_matched_controls" / model / f"{model}_best.pt"
    protocol = root / "protocol"
    jobs = [("ffpp_test", protocol / "control" / "test.csv",
             pd.read_csv(root / "01_matched_controls" / model / "predictions" / f"{model}_test_predictions.csv")),
            ("external_normalized", root / "02_external_bias_audit" / "normalized.csv",
             pd.read_csv(root / "01_matched_controls" / model / "predictions" / f"{model}_external_normalized_predictions.csv"))]
    for method in METHODS:
        lomo = root / "05_lomo" / f"lomo_{method}" / model
        allp = pd.read_csv(lomo / "predictions_test_all.csv")
        held = allp[(allp.label == 0) | (allp.method == method)].reset_index(drop=True)
        jobs.append((f"lomo_{method}", protocol / f"lomo_{method}" / "test.csv", held, lomo / f"{model}_best.pt"))
    for job in jobs:
        if only and not job[0].startswith(only):
            continue
        condition, manifest, predictions = job[:3]
        # LOMO panels must explain the LOMO checkpoint that produced those predictions, not the control model.
        used = job[3] if len(job) == 4 else checkpoint
        print("CAM", condition, used.name, flush=True)
        cam_panels(config, used, manifest, predictions.reset_index(drop=True), condition)
    print("CAMDONE")


if __name__ == "__main__":
    import sys
    main(only=sys.argv[1] if len(sys.argv) > 1 else None)
