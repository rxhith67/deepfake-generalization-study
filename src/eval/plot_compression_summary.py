"""Plot multiple compression-result CSVs on one comparison chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_compression_summary(inputs: list[Path], output: Path) -> pd.DataFrame:
    """Combine robustness tables and plot accuracy/AUC against JPEG quality."""
    frames = [pd.read_csv(path) for path in inputs]
    table = pd.concat(frames, ignore_index=True)
    required = {"model", "quality", "accuracy", "auc"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Compression tables are missing columns: {sorted(missing)}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for model, rows in table.groupby("model", sort=False):
        rows = rows.sort_values("quality")
        axes[0].plot(rows["quality"], rows["accuracy"], marker="o", label=model)
        axes[1].plot(rows["quality"], rows["auc"], marker="o", label=model)
    axes[0].set_title("Accuracy")
    axes[1].set_title("ROC AUC")
    for axis in axes:
        axis.set_xlabel("JPEG quality")
        axis.set_ylim(0.45, 1.0)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="best")
    figure.suptitle("FaceForensics++ compression robustness")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = plot_compression_summary(args.inputs, args.output)
    print(f"Plotted {table['model'].nunique()} models from {len(table)} rows: {args.output}")


if __name__ == "__main__":
    main()
