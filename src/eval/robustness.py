"""Evaluate checkpoint degradation over controlled JPEG quality levels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import evaluate_checkpoint
from src.utils import write_json


def jpeg_compress(image_uint8: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip a BGR/RGB uint8 image through JPEG in memory."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    ok, buffer = cv2.imencode(".jpg", image_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed")
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, type=Path, nargs="+", help="One or more model checkpoints"
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--qualities", nargs="+", type=int, default=[100, 90, 60, 40, 20, 10])
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/tables/compression.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("outputs/figures/compression.png"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    details = {}
    for checkpoint in args.checkpoint:
        model_details = {}
        model_name = checkpoint.stem
        for quality in args.qualities:
            result, _ = evaluate_checkpoint(
                checkpoint,
                args.csv,
                args.data_root,
                args.device,
                args.batch_size,
                args.num_workers,
                jpeg_quality=quality,
            )
            model_name = str(result["model"])
            metrics = result["metrics"]
            rows.append(
                {
                    "model": model_name,
                    "quality": quality,
                    **{key: value for key, value in metrics.items() if key != "confusion"},
                }
            )
            model_details[str(quality)] = result
            print(
                f"model={model_name} quality={quality}: "
                f"accuracy={metrics['accuracy']:.4f}, auc={metrics['auc']:.4f}"
            )
        details[model_name] = model_details
    table = pd.DataFrame(rows).sort_values(["model", "quality"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    write_json(details, args.output_csv.with_suffix(".json"))

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for model_name, model_rows in table.groupby("model"):
        axis.plot(model_rows["quality"], model_rows["accuracy"], marker="o", label=model_name)
    axis.set(xlabel="JPEG quality", ylabel="Score", ylim=(0, 1.02), title="Compression robustness")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_figure, dpi=200)
    plt.close(fig)
    print(f"Table: {args.output_csv}\nFigure: {args.output_figure}")


if __name__ == "__main__":
    main()
