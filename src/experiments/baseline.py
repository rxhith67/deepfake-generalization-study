"""Fresh inference of historical checkpoints; never retrain or alter them."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.common import configuration, record_run, save_csv, atomic_text, update_master
from src.experiments.evaluation import infer, measurement, strict_ensemble, probability_figures
from src.experiments.manifests import canonical_id


def run(config):
    root = Path(config["output_dir"]) / "00_baseline_reproduction"
    record_run(root, config, config["checkpoints"].values())
    results, comparisons, all_predictions, by_split = [], [], [], {}
    datasets = {"val": (Path(config["data"]["historical_splits"]) / "val.csv", "ffpp_historical"),
                "test": (Path(config["data"]["historical_splits"]) / "test.csv", "ffpp_historical"),
                "external": (Path(config["data"]["external_csv"]), "external_original")}
    for split, (manifest, dataset) in datasets.items():
        frames = []
        for model, checkpoint in config["checkpoints"].items():
            rows = infer(checkpoint, manifest, config, root / f"{model}_{split}_predictions.csv", dataset,
                         "external_test" if split == "external" else split, config["historical_protocol_id"])
            frames.append(rows)
            all_predictions.append(rows)
            old_path = (Path("outputs/modal_downloads") / f"{model}_{split}_predictions.csv" if split != "external"
                        else Path("outputs/tables") / f"{model}_200video_cloud_deepfakeface_predictions.csv")
            old = pd.read_csv(old_path)
            old["sample_id"] = old.image_path.map(canonical_id)
            old = old.set_index("sample_id", verify_integrity=True)
            new = rows.set_index("sample_id", verify_integrity=True)
            if set(old.index) != set(new.index) or not np.array_equal(old.loc[new.index].label, new.label):
                raise ValueError("Historical comparison sample/label mismatch")
            difference = np.abs(old.loc[new.index].prob_fake.to_numpy() - new.prob_fake.to_numpy())
            comparisons.append({"model": model, "split": split, "n": len(rows),
                                "max_probability_difference": float(difference.max()),
                                "mean_probability_difference": float(difference.mean())})
            results.append(measurement(rows, "baseline_reproduction", config, ci=split != "val"))
            if split == "test":
                results.append(measurement(rows, "baseline_reproduction", config, unit="video", ci=True))
                for method in sorted(set(rows.loc[rows.label == 1, "method"])):
                    item = measurement(rows[(rows.label == 0) | (rows.method == method)], "baseline_per_method", config)
                    item["test_manipulation"] = method
                    results.append(item)
        ensemble_config = json.loads(Path("outputs/tables/ensemble_200video_cloud_frame.json").read_text())
        ensemble = strict_ensemble(frames, ensemble_config["weights"], ensemble_config["calibration_metrics"]["threshold"])
        frames.append(ensemble)
        save_csv(root / f"ensemble_{split}_predictions.csv", ensemble)
        all_predictions.append(ensemble)
        results.append(measurement(ensemble, "baseline_reproduction", config, ci=split != "val"))
        by_split[split] = {r.model.iloc[0]:r for r in frames}
        if split != "val":
            probability_figures(by_split[split], root / split)
    save_csv(root / "predictions.csv", pd.concat(all_predictions, ignore_index=True))
    save_csv(root / "metrics.csv", pd.DataFrame(results))
    save_csv(root / "prediction_comparison.csv", pd.DataFrame(comparisons))
    update_master(config["output_dir"], [r for r in results if r["split"] != "val"])
    atomic_text(root / "baseline_reproduction.md", "# Historical baseline reproduction\n\nFresh checkpoint inference, not just saved-score recalculation. "
                "Historical target-video protocol only; source/person identity safety is not asserted.\n\n"
                "See metrics.csv and prediction_comparison.csv for actual values and numerical differences. "
                "Thresholds and frame ensemble weights are frozen from historical FF++ validation. "
                "Bootstrap uses target-video clusters for FF++ and images externally.\n")
    print(pd.DataFrame(results)[["model","test_dataset","evaluation_unit","auc"]].to_string(index=False))


if __name__ == "__main__":
    run(configuration())
