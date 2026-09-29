"""Rewrite absolute local crop paths as portable dataset-relative paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def portable_image_path(value: str, marker: str = "/processed/") -> str:
    normalised = str(value).replace("\\", "/")
    if marker not in normalised:
        raise ValueError(f"Image path does not contain {marker!r}: {value}")
    return normalised.split(marker, 1)[1]


def convert_manifest(source: Path, destination: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    if "image_path" not in frame:
        raise ValueError(f"{source} has no image_path column")
    frame["image_path"] = frame["image_path"].map(portable_image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("train.csv", "val.csv", "test.csv"):
        output = args.output_dir / name
        frame = convert_manifest(args.input_dir / name, output)
        print(f"{name}: {len(frame)} rows -> {output}")


if __name__ == "__main__":
    main()

