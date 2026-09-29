"""Uniformly sample frames from one video or a directory of videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def extract_frames(video_path: str | Path, num_frames: int = 30) -> list[np.ndarray]:
    """Return up to ``num_frames`` RGB frames sampled at uniform indices."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    requested = min(num_frames, total)
    indices = np.unique(np.linspace(0, total - 1, requested, dtype=int))
    # Decode once from the beginning. Repeated random seeks are particularly
    # expensive for inter-frame codecs such as H.264 because each seek may
    # decode forward from a preceding keyframe.
    frames: list[np.ndarray] = []
    target_position = 0
    for frame_index in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index == int(indices[target_position]):
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            target_position += 1
            if target_position == len(indices):
                break
    cap.release()
    return frames


def save_frames(video_path: str | Path, output_dir: str | Path, num_frames: int) -> int:
    """Extract and save JPEG frames below ``output_dir/<video_stem>``."""
    output = Path(output_dir) / Path(video_path).stem
    output.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(video_path, num_frames)
    for number, frame_rgb in enumerate(frames):
        destination = output / f"{number:06d}.jpg"
        if not cv2.imwrite(str(destination), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write frame: {destination}")
    return len(frames)


def iter_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def has_complete_frame_set(
    video_path: str | Path, output_dir: str | Path, num_frames: int
) -> bool:
    """Return whether a video already has the requested number of JPEG frames."""
    frame_dir = Path(output_dir) / Path(video_path).stem
    return len(list(frame_dir.glob("*.jpg"))) == num_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Video or directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have NUM_FRAMES JPEGs in the output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = iter_videos(args.input)
    if not videos:
        raise SystemExit(f"No supported videos found below {args.input}")
    total = 0
    skipped = 0
    for video in videos:
        relative_parent = video.parent.relative_to(args.input) if args.input.is_dir() else Path()
        output_dir = args.output / relative_parent
        if args.skip_existing and has_complete_frame_set(video, output_dir, args.num_frames):
            skipped += 1
            continue
        total += save_frames(video, output_dir, args.num_frames)
    print(
        f"Saved {total} frames from {len(videos) - skipped} videos to {args.output}; "
        f"skipped {skipped} complete videos"
    )


if __name__ == "__main__":
    main()
