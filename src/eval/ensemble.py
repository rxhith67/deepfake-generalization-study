"""Fuse independently trained detectors and report frame/video metrics.

When validation predictions are supplied, the operating threshold is selected
on validation data only.  For a two-model ensemble, the mixture weight is also
selected on validation AUC, keeping the test set completely untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import find_optimal_threshold, per_group_reports, report_all
from src.evaluate import aggregate_predictions, binary_method_reports
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path, help="CSV outputs from src.evaluate")
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/ensemble.csv"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--calibration-predictions",
        nargs="+",
        type=Path,
        help="Matching validation prediction CSVs used to tune weight/threshold",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Optional non-negative model weights (normalised automatically)",
    )
    parser.add_argument(
        "--calibration-level",
        choices=["frame", "video"],
        default="frame",
        help="Unit on which validation weight and threshold selection is performed",
    )
    return parser.parse_args()


def combine_predictions(paths: list[Path], weights: list[float]) -> pd.DataFrame:
    """Merge prediction files by path and return their weighted probability."""
    if len(paths) < 2:
        raise ValueError("Provide at least two prediction CSV files")
    if len(paths) != len(weights):
        raise ValueError("The number of weights must match the number of models")
    weight_array = np.asarray(weights, dtype=float)
    if np.any(weight_array < 0) or not np.isfinite(weight_array).all() or weight_array.sum() <= 0:
        raise ValueError("Weights must be finite, non-negative, and not all zero")
    weight_array /= weight_array.sum()

    frames = [pd.read_csv(path) for path in paths]
    key = "image_path"
    required = {key, "label", "prob_fake"}
    for path, frame in zip(paths, frames):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    # Preserve dataset/method/video metadata from the first model so that the
    # ensemble receives the same detailed evaluation as a single detector.
    excluded = {"prob_fake", "prediction"}
    base = frames[0][[column for column in frames[0].columns if column not in excluded]].copy()
    probability_columns = []
    for index, (path, frame) in enumerate(zip(paths, frames)):
        column = f"prob_fake_{index}"
        label_column = f"label_{index}"
        base = base.merge(
            frame[[key, "label", "prob_fake"]].rename(
                columns={"label": label_column, "prob_fake": column}
            ),
            on=key,
            validate="one_to_one",
        )
        if not (base["label"].astype(int) == base[label_column].astype(int)).all():
            raise ValueError(f"Labels disagree in {path}")
        base.drop(columns=label_column, inplace=True)
        probability_columns.append(column)
    base["prob_fake"] = base[probability_columns].to_numpy() @ weight_array
    return base


def calibration_unit(rows: pd.DataFrame, level: str) -> pd.DataFrame:
    """Return frame rows or validation-only video aggregates."""
    if level == "frame":
        return rows
    if "video_id" not in rows:
        raise ValueError("Video-level calibration requires a video_id column")
    columns = [column for column in ("dataset", "method", "video_id") if column in rows]
    return aggregate_predictions(rows, columns)


def optimise_two_model_weight(
    paths: list[Path], calibration_level: str = "frame"
) -> tuple[list[float], float]:
    """Select a two-model mixture on validation AUC, then its threshold."""
    if len(paths) != 2:
        weights = [1.0 / len(paths)] * len(paths)
        rows = calibration_unit(combine_predictions(paths, weights), calibration_level)
        return weights, find_optimal_threshold(rows["label"], rows["prob_fake"])
    candidates: list[tuple[float, float, float]] = []
    for first_weight in np.linspace(0.0, 1.0, 101):
        weights = [float(first_weight), float(1.0 - first_weight)]
        rows = calibration_unit(combine_predictions(paths, weights), calibration_level)
        threshold = find_optimal_threshold(rows["label"], rows["prob_fake"])
        metrics = report_all(rows["label"], rows["prob_fake"], threshold)
        candidates.append((float(metrics["auc"]), float(metrics["balanced_accuracy"]), first_weight))
    _, _, best_first_weight = max(candidates)
    weights = [float(best_first_weight), float(1.0 - best_first_weight)]
    rows = calibration_unit(combine_predictions(paths, weights), calibration_level)
    return weights, find_optimal_threshold(rows["label"], rows["prob_fake"])


def detailed_report(rows: pd.DataFrame, threshold: float) -> dict[str, object]:
    report: dict[str, object] = {
        "metrics": report_all(rows["label"], rows["prob_fake"], threshold),
        "binary_per_method": binary_method_reports(rows, threshold),
    }
    if "method" in rows:
        report["per_method"] = per_group_reports(
            rows["label"], rows["prob_fake"], rows["method"].fillna("unknown"), threshold
        )
    if "video_id" in rows:
        columns = [column for column in ("dataset", "method", "video_id") if column in rows]
        video_rows = aggregate_predictions(rows, columns)
        report["video_metrics"] = report_all(
            video_rows["label"], video_rows["prob_fake"], threshold
        )
        report["video_binary_per_method"] = binary_method_reports(video_rows, threshold)
    return report


def main() -> None:
    args = parse_args()
    if len(args.predictions) < 2:
        raise SystemExit("Provide at least two prediction CSV files")
    if args.calibration_predictions and len(args.calibration_predictions) != len(args.predictions):
        raise SystemExit("Calibration and evaluation prediction counts must match")
    if args.weights and len(args.weights) != len(args.predictions):
        raise SystemExit("The number of weights must match the number of models")

    weights = args.weights or [1.0 / len(args.predictions)] * len(args.predictions)
    threshold = args.threshold
    calibration_metrics = None
    if args.calibration_predictions:
        if not args.weights:
            weights, calibrated_threshold = optimise_two_model_weight(
                args.calibration_predictions, args.calibration_level
            )
        else:
            calibration_rows = calibration_unit(
                combine_predictions(args.calibration_predictions, weights), args.calibration_level
            )
            calibrated_threshold = find_optimal_threshold(
                calibration_rows["label"], calibration_rows["prob_fake"]
            )
        threshold = threshold if threshold is not None else calibrated_threshold
        calibration_rows = calibration_unit(
            combine_predictions(args.calibration_predictions, weights), args.calibration_level
        )
        calibration_metrics = report_all(
            calibration_rows["label"], calibration_rows["prob_fake"], threshold
        )
    threshold = 0.5 if threshold is None else threshold

    base = combine_predictions(args.predictions, weights)
    base["prediction"] = (base["prob_fake"] >= threshold).astype(int)
    report = detailed_report(base, threshold)
    report.update(
        {
            "prediction_files": [str(path) for path in args.predictions],
            "weights": [float(value) for value in weights],
            "calibration_level": args.calibration_level,
            "calibration_metrics": calibration_metrics,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.output, index=False)
    write_json(report, args.output.with_suffix(".json"))
    print(report["metrics"])
    print(f"Weights: {weights}; threshold: {threshold}")


if __name__ == "__main__":
    main()
