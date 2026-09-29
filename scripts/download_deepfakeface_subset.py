"""Range-download paired real and Stable Diffusion faces from DeepFakeFace."""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path, PurePosixPath

import fsspec
from huggingface_hub import hf_hub_url
from tqdm import tqdm

REPO_ID = "desingh/DeepFakeFace"
SOURCES = {"real": "wiki.zip", "fake": "text2img.zip"}


def relative_members(names: list[str], root: str) -> set[str]:
    """Return image member paths relative to an archive's top-level directory."""
    prefix = f"{root}/"
    return {
        name[len(prefix) :]
        for name in names
        if name.startswith(prefix) and name.lower().endswith((".jpg", ".jpeg", ".png"))
    }


def choose_pairs(real_names: list[str], fake_names: list[str], count: int, seed: int) -> list[str]:
    """Choose diverse matching pairs in a few range-download-friendly runs."""
    real = relative_members(real_names, "wiki")
    paired = [
        name[len("text2img/") :]
        for name in fake_names
        if name.startswith("text2img/")
        and name[len("text2img/") :] in real
        and name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(paired) < count:
        raise ValueError(f"Requested {count} pairs, but only {len(paired)} are available")
    rng = random.Random(seed)
    block_count = min(10, count)
    selected: list[str] = []
    base, remainder = divmod(count, block_count)
    for block in range(block_count):
        take = base + (1 if block < remainder else 0)
        low = block * len(paired) // block_count
        high = (block + 1) * len(paired) // block_count
        start = rng.randint(low, max(low, high - take))
        selected.extend(paired[start : start + take])
    return selected


def _open_remote_zip(filename: str) -> tuple[object, zipfile.ZipFile]:
    url = hf_hub_url(REPO_ID, filename, repo_type="dataset")
    # Faces are small; a modest block avoids fetching megabytes around each
    # randomly selected ZIP member while still amortising HTTP range requests.
    remote = fsspec.open(url, "rb", block_size=512 * 1024).open()
    return remote, zipfile.ZipFile(remote)


def download_subset(output: Path, count: int = 500, seed: int = 42) -> list[str]:
    """Download ``count`` matching real/fake image pairs using HTTP range requests."""
    real_remote, real_zip = _open_remote_zip(SOURCES["real"])
    fake_remote, fake_zip = _open_remote_zip(SOURCES["fake"])
    try:
        selected = choose_pairs(real_zip.namelist(), fake_zip.namelist(), count, seed)
        for relative in tqdm(selected, desc="paired faces"):
            safe_relative = Path(*PurePosixPath(relative).parts)
            for label, archive in (("real", real_zip), ("fake", fake_zip)):
                destination = output / label / safe_relative
                if destination.is_file():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                root = "wiki" if label == "real" else "text2img"
                destination.write_bytes(archive.read(f"{root}/{relative}"))
    finally:
        real_zip.close()
        fake_zip.close()
        real_remote.close()
        fake_remote.close()
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/deepfakeface_subset"))
    parser.add_argument("--count", type=int, default=500, help="Number of matching pairs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = download_subset(args.output, args.count, args.seed)
    print(f"Downloaded {len(selected)} paired real/Stable-Diffusion faces to {args.output}")


if __name__ == "__main__":
    main()
