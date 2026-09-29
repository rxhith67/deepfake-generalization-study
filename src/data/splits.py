"""Create deterministic train/validation/test CSVs grouped by video ID."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_manifest(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_column: str = "video_id",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split rows while keeping every group wholly within one partition."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
    required = {"image_path", "label", group_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot split an empty manifest")
    groups = frame[group_column].astype(str)
    if groups.nunique() < 3:
        raise ValueError("At least three distinct groups are required")

    first = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, remainder_idx = next(first.split(frame, frame["label"], groups))
    remainder = frame.iloc[remainder_idx]
    relative_val = val_ratio / (val_ratio + test_ratio)
    second = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=seed + 1)
    val_local, test_local = next(
        second.split(remainder, remainder["label"], remainder[group_column].astype(str))
    )
    result = {
        "train": frame.iloc[train_idx].reset_index(drop=True),
        "val": remainder.iloc[val_local].reset_index(drop=True),
        "test": remainder.iloc[test_local].reset_index(drop=True),
    }
    group_sets = {name: set(part[group_column].astype(str)) for name, part in result.items()}
    if any(
        group_sets[left] & group_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("Group leakage detected")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--group-column", default="video_id")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    splits = split_manifest(
        frame,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.group_column,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        destination = args.output_dir / f"{name}.csv"
        part.to_csv(destination, index=False)
        print(f"{name}: {len(part)} frames, {part[args.group_column].nunique()} groups -> {destination}")


if __name__ == "__main__":
    main()

