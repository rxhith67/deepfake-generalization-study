"""Limit correlated face crops per source video using uniform frame indices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def subsample_groups(
    frame: pd.DataFrame, limit: int, group_columns: list[str]
) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be positive")
    missing = set(group_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing group columns: {sorted(missing)}")
    selected = []
    for _, group in frame.groupby(group_columns, dropna=False, sort=False):
        group = group.sort_values("image_path")
        if len(group) > limit:
            indices = np.linspace(0, len(group) - 1, limit, dtype=int)
            group = group.iloc[np.unique(indices)]
        selected.append(group)
    return pd.concat(selected, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames-per-video", required=True, type=int)
    parser.add_argument("--group-columns", nargs="+", default=["method", "video_id"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    selected = subsample_groups(frame, args.frames_per_video, args.group_columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    print(f"Selected {len(selected)} of {len(frame)} rows -> {args.output}")


if __name__ == "__main__":
    main()

