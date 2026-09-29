"""Extract a balanced, face-only evaluation manifest from a GenImage Arrow shard."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from src.data.detect_faces import create_mtcnn, detect_and_crop_face


def read_arrow_rows(path: Path) -> list[dict[str, object]]:
    """Read the fields needed from a Hugging Face ``save_to_disk`` Arrow shard."""
    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_stream(source).read_all().select(["image_path", "image", "label"])
    return table.to_pylist()


def prepare_face_subset(
    arrow_path: Path,
    output_dir: Path,
    manifest_path: Path,
    detector: Any,
    *,
    max_per_class: int = 250,
    image_size: int = 224,
    margin: float = 0.30,
    seed: int = 42,
) -> tuple[list[dict[str, object]], int]:
    """Detect faces in shuffled Arrow rows and save at most ``max_per_class`` per label."""
    source_rows = read_arrow_rows(arrow_path)
    random.Random(seed).shuffle(source_rows)
    counts = {0: 0, 1: 0}
    manifest_rows: list[dict[str, object]] = []
    missed = 0

    for source in source_rows:
        label = int(source["label"])
        if label not in counts or counts[label] >= max_per_class:
            continue
        encoded = np.frombuffer(source["image"], dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            missed += 1
            continue
        crop = detect_and_crop_face(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            detector,
            image_size=image_size,
            margin=margin,
        )
        if crop is None:
            missed += 1
            continue

        original = Path(str(source["image_path"]))
        class_name = "fake" if label else "real"
        destination = output_dir / class_name / f"{original.stem}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        manifest_rows.append(
            {
                "image_path": str(destination.resolve()),
                "label": label,
                "video_id": original.stem,
                "method": "ADM" if label else "real",
                "dataset": "genimage_adm",
            }
        )
        counts[label] += 1
        if all(count >= max_per_class for count in counts.values()):
            break

    # Cross-generation accuracy and threshold metrics should not be dominated by
    # whichever source happens to contain more detectable faces.
    balanced_count = min(counts.values())
    kept = {0: 0, 1: 0}
    balanced_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        label = int(row["label"])
        if kept[label] < balanced_count:
            balanced_rows.append(row)
            kept[label] += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "label", "video_id", "method", "dataset"]
        )
        writer.writeheader()
        writer.writerows(balanced_rows)
    return balanced_rows, missed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", required=True, type=Path, help="Downloaded GenImage Arrow shard")
    parser.add_argument("--output", required=True, type=Path, help="Face crop directory")
    parser.add_argument("--manifest", required=True, type=Path, help="Output manifest CSV")
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, missed = prepare_face_subset(
        args.arrow,
        args.output,
        args.manifest,
        create_mtcnn(args.device),
        max_per_class=args.max_per_class,
        image_size=args.image_size,
        margin=args.margin,
        seed=args.seed,
    )
    real = sum(int(row["label"]) == 0 for row in rows)
    fake = len(rows) - real
    print(f"Saved {len(rows)} face crops (real={real}, fake={fake}); no face/decode={missed}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
