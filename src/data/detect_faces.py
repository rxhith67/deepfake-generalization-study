"""Detect the largest face, add a margin, and create crop manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def create_mtcnn(device: str = "cpu") -> Any:
    """Construct MTCNN lazily so unrelated project commands still work."""
    try:
        from facenet_pytorch import MTCNN
    except ImportError as error:
        raise RuntimeError(
            "Face detection requires facenet-pytorch. Install requirements.txt first."
        ) from error
    # keep_all gives us explicit control over selecting the largest detection.
    return MTCNN(keep_all=True, device=device, post_process=False)


def expand_box(
    box: np.ndarray | list[float], width: int, height: int, margin: float = 0.30
) -> tuple[int, int, int, int]:
    """Expand an xyxy box by ``margin`` of its largest side and clip to image."""
    x1, y1, x2, y2 = map(float, box)
    pad = margin * max(x2 - x1, y2 - y1)
    return (
        max(0, int(np.floor(x1 - pad))),
        max(0, int(np.floor(y1 - pad))),
        min(width, int(np.ceil(x2 + pad))),
        min(height, int(np.ceil(y2 + pad))),
    )


def detect_and_crop_face(
    frame_rgb: np.ndarray,
    detector: Any,
    image_size: int = 224,
    margin: float = 0.30,
) -> np.ndarray | None:
    """Return an RGB crop around the largest detected face, or ``None``."""
    boxes, _ = detector.detect(frame_rgb)
    return crop_largest_face(frame_rgb, boxes, image_size, margin)


def crop_largest_face(
    frame_rgb: np.ndarray,
    boxes: np.ndarray | None,
    image_size: int = 224,
    margin: float = 0.30,
) -> np.ndarray | None:
    """Crop the largest supplied face box from an RGB frame."""
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box = boxes[int(np.argmax(areas))]
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = expand_box(box, width, height, margin)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def detect_and_crop_faces(
    frames_rgb: list[np.ndarray],
    detector: Any,
    image_size: int = 224,
    margin: float = 0.30,
) -> list[np.ndarray | None]:
    """Batch MTCNN inference for frames that share a video resolution."""
    if not frames_rgb:
        return []
    batch_boxes, _ = detector.detect(frames_rgb)
    return [
        crop_largest_face(frame, boxes, image_size, margin)
        for frame, boxes in zip(frames_rgb, batch_boxes)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Frame directory")
    parser.add_argument("--output", required=True, type=Path, help="Face crop directory")
    parser.add_argument("--manifest", type=Path, help="Output CSV (default: OUTPUT/manifest.csv)")
    parser.add_argument("--label", required=True, type=int, choices=[0, 1])
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--margin", default=0.30, type=float)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing crop files while rebuilding the complete manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = create_mtcnn(args.device)
    images = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found below {args.input}")
    manifest_path = args.manifest or args.output / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missed = 0
    reused = 0
    for source in images:
        relative = source.relative_to(args.input)
        destination = (args.output / relative).with_suffix(".jpg")
        if args.skip_existing and destination.is_file():
            reused += 1
            rows.append(
                {
                    "image_path": str(destination.resolve()),
                    "label": args.label,
                    "video_id": relative.parent.as_posix(),
                    "method": args.method,
                    "dataset": args.dataset,
                }
            )
            continue
        frame_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            missed += 1
            continue
        crop = detect_and_crop_face(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            detector,
            args.image_size,
            args.margin,
        )
        if crop is None:
            missed += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        # extract_frames writes <video_id>/<frame>.jpg, so parent is the group.
        rows.append(
            {
                # Absolute paths make generated manifests unambiguous even when
                # the configured dataset root differs from the current folder.
                "image_path": str(destination.resolve()),
                "label": args.label,
                "video_id": relative.parent.as_posix(),
                "method": args.method,
                "dataset": args.dataset,
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "video_id", "method", "dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Saved {len(rows)} crops ({reused} reused); skipped {missed}; "
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
