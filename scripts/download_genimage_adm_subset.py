"""Download the compact GenImage ADM shard used by the cross-generation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "nebula/GenImage-arrow"
FILES = (
    "data/test/ADM/data-00001-of-00003.arrow",
    "data/test/ADM/dataset_info.json",
    "data/test/ADM/mapping.json",
    "data/test/ADM/test.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/genimage_arrow"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for filename in FILES:
        path = hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            local_dir=args.output,
        )
        print(path)


if __name__ == "__main__":
    main()
