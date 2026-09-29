"""Extract sampled video frames directly to aligned face crops.

This combines frame extraction and face detection without retaining a second,
large directory of intermediate full-resolution frames.  It is intended for
expanding FaceForensics++ on storage-constrained machines.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from src.data.detect_faces import create_mtcnn, detect_and_crop_faces
from src.data.extract_frames import extract_frames, iter_videos


def video_group_id(stem: str, mode: str = "first") -> str:
    """Return the split group, optionally grouping a manipulated target identity."""
    if mode == "stem":
        return stem
    if mode == "first":
        return stem.split("_", 1)[0]
    raise ValueError(f"Unknown video ID mode: {mode}")


def existing_rows(
    output: Path,
    stem: str,
    label: int,
    method: str,
    dataset: str,
    id_mode: str,
) -> list[dict[str, object]]:
    """Describe already-created crops for an idempotent resume."""
    return [
        {
            "image_path": str(path.resolve()),
            "label": label,
            "video_id": video_group_id(stem, id_mode),
            "method": method,
            "dataset": dataset,
        }
        for path in sorted((output / stem).glob("*.jpg"))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Video or directory")
    parser.add_argument("--output", required=True, type=Path, help="Face crop directory")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label", required=True, type=int, choices=[0, 1])
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", default="FaceForensics++")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument(
        "--detection-batch-size",
        type=int,
        default=4,
        help="MTCNN batch size within each video (reduce if GPU memory is limited)",
    )
    parser.add_argument("--video-id-mode", choices=["first", "stem"], default="first")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = iter_videos(args.input)
    if not videos:
        raise SystemExit(f"No supported videos found below {args.input}")
    detector = create_mtcnn(args.device)
    rows: list[dict[str, object]] = []
    missed = 0
    reused = 0
    processed = 0
    for video in tqdm(videos, desc=args.method):
        prior = existing_rows(
            args.output,
            video.stem,
            args.label,
            args.method,
            args.dataset,
            args.video_id_mode,
        )
        if args.skip_existing and len(prior) >= args.num_frames:
            rows.extend(prior)
            reused += len(prior)
            continue
        if args.skip_existing:
            rows.extend(prior)
            reused += len(prior)
        frames = extract_frames(video, args.num_frames)
        destination_dir = args.output / video.stem
        pending = [
            (index, frame)
            for index, frame in enumerate(frames)
            if not (args.skip_existing and (destination_dir / f"{index:06d}.jpg").is_file())
        ]
        batch_size = max(1, args.detection_batch_size)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            crops = detect_and_crop_faces(
                [frame for _, frame in batch], detector, args.image_size, args.margin
            )
            for (index, _), crop in zip(batch, crops):
                if crop is None:
                    missed += 1
                    continue
                destination = destination_dir / f"{index:06d}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write face crop: {destination}")
                rows.append(
                    {
                        "image_path": str(destination.resolve()),
                        "label": args.label,
                        "video_id": video_group_id(video.stem, args.video_id_mode),
                        "method": args.method,
                        "dataset": args.dataset,
                    }
                )
        processed += 1

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "label", "video_id", "method", "dataset"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Saved {len(rows)} manifest rows ({reused} crops reused, {processed} videos processed, "
        f"{missed} frames without a face) -> {args.manifest}"
    )


if __name__ == "__main__":
    main()
