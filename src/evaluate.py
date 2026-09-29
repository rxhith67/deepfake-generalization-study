"""Evaluate a saved checkpoint on any project-format manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FaceDataset
from src.data.transforms import build_transform, transform_options
from src.eval.metrics import per_group_reports, report_all
from src.utils import load_checkpoint, resolve_device, write_json


def aggregate_predictions(
    rows: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Average frame probabilities into video/source-level predictions."""
    missing = [column for column in group_columns if column not in rows]
    if missing:
        raise ValueError(f"Cannot aggregate without columns: {missing}")
    grouped = (
        rows.groupby(group_columns, dropna=False, sort=False)
        .agg(label=("label", "first"), prob_fake=("prob_fake", "mean"), frames=("label", "size"))
        .reset_index()
    )
    inconsistent = rows.groupby(group_columns, dropna=False)["label"].nunique().max()
    if int(inconsistent) > 1:
        raise ValueError("An aggregation group contains conflicting labels")
    return grouped


def binary_method_reports(
    rows: pd.DataFrame, threshold: float
) -> dict[str, dict[str, object]]:
    """Compare every fake method against the shared real examples."""
    if "method" not in rows:
        return {}
    methods = sorted(set(rows.loc[rows["label"].astype(int) == 1, "method"].astype(str)))
    reports: dict[str, dict[str, object]] = {}
    for method in methods:
        subset = rows[(rows["label"].astype(int) == 0) | (rows["method"].astype(str) == method)]
        reports[method] = report_all(subset["label"], subset["prob_fake"], threshold)
    return reports


@torch.inference_mode()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[float], list[int]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    indices: list[int] = []
    offset = 0
    for batch in loader:
        if len(batch) == 3:
            images, targets, metadata = batch
            batch_indices = metadata.get("index", range(offset, offset + len(targets)))
            indices.extend(int(value) for value in batch_indices)
        else:
            images, targets = batch
            indices.extend(range(offset, offset + len(targets)))
        logits = model(images.to(device, non_blocking=True)).flatten()
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(targets.int().cpu().tolist())
        offset += len(targets)
    return labels, probabilities, indices


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    data_root: str | Path,
    device_name: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
    threshold: float | None = None,
    group_column: str | None = "method",
    jpeg_quality: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    transform_kwargs = transform_options(config)
    threshold = float(
        threshold if threshold is not None else checkpoint.get("threshold", 0.5)
    )
    dataset = FaceDataset(
        csv_path,
        root=data_root,
        transform=build_transform(jpeg_quality=jpeg_quality, **transform_kwargs),
        return_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(
            config.get("training", {}).get("pin_memory", device.type == "cuda")
        ),
    )
    labels, probabilities, indices = predict(model, loader, device)
    predictions = (torch.tensor(probabilities) >= threshold).int().tolist()
    rows = dataset.frame.iloc[indices].copy().reset_index(drop=True)
    rows["prob_fake"] = probabilities
    rows["prediction"] = predictions
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "model": checkpoint["model_name"],
        "dataset_csv": str(csv_path),
        "jpeg_quality": jpeg_quality,
        "metrics": report_all(labels, probabilities, threshold),
    }
    if group_column and group_column in rows.columns:
        result[f"per_{group_column}"] = per_group_reports(
            labels, probabilities, rows[group_column].fillna("unknown").astype(str), threshold
        )
    result["binary_per_method"] = binary_method_reports(rows, threshold)
    if "video_id" in rows.columns:
        group_columns = [column for column in ("dataset", "method", "video_id") if column in rows]
        video_rows = aggregate_predictions(rows, group_columns)
        video_rows["prediction"] = (video_rows["prob_fake"] >= threshold).astype(int)
        result["video_metrics"] = report_all(
            video_rows["label"], video_rows["prob_fake"], threshold
        )
        result["video_binary_per_method"] = binary_method_reports(video_rows, threshold)
    return result, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--group-column", default="method")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, predictions = evaluate_checkpoint(
        args.checkpoint,
        args.csv,
        args.data_root,
        args.device,
        args.batch_size,
        args.num_workers,
        args.threshold,
        args.group_column,
    )
    write_json(result, args.output)
    prediction_path = args.predictions or args.output.with_suffix(".predictions.csv")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    print(result["metrics"])
    print(f"Metrics: {args.output}\nPredictions: {prediction_path}")


if __name__ == "__main__":
    main()
