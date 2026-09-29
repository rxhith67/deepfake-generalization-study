"""Intersect two manifests by trailing path components and emit matched pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def path_key(value: str, depth: int = 2) -> str:
    parts = Path(value).parts
    return Path(*parts[-depth:]).as_posix()


def pair_manifests(first: Path, second: Path, output: Path, key_depth: int = 2) -> pd.DataFrame:
    """Keep only keys present once in each manifest and share their group ID."""
    left = pd.read_csv(first).copy()
    right = pd.read_csv(second).copy()
    for frame, name in ((left, str(first)), (right, str(second))):
        if "image_path" not in frame or "label" not in frame:
            raise ValueError(f"Manifest lacks image_path/label columns: {name}")
        frame["_pair_key"] = frame["image_path"].map(lambda value: path_key(value, key_depth))
        if frame["_pair_key"].duplicated().any():
            raise ValueError(f"Pair keys are not unique in {name}; increase --key-depth")

    shared = sorted(set(left["_pair_key"]) & set(right["_pair_key"]))
    left = left.set_index("_pair_key").loc[shared].reset_index()
    right = right.set_index("_pair_key").loc[shared].reset_index()
    left["video_id"] = left["_pair_key"]
    right["video_id"] = right["_pair_key"]
    paired = pd.concat([left, right], ignore_index=True).drop(columns="_pair_key")
    output.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output, index=False)
    return paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key-depth", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired = pair_manifests(args.first, args.second, args.output, args.key_depth)
    print(f"Saved {len(paired) // 2} matched pairs ({len(paired)} rows): {args.output}")


if __name__ == "__main__":
    main()
