"""Combine project-format CSV manifests with duplicate/path validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["image_path", "label", "video_id", "method", "dataset"]


def combine_manifests(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one manifest is required")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame[REQUIRED_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)
    conflicts = combined.groupby("image_path")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("The same image path has conflicting labels")
    return combined.drop_duplicates(subset="image_path", keep="first").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = combine_manifests(args.manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    counts = combined.groupby(["label", "method"]).size().to_dict()
    print(f"Saved {len(combined)} rows -> {args.output}; counts={counts}")


if __name__ == "__main__":
    main()

