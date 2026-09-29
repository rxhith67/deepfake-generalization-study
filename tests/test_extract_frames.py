from pathlib import Path

import cv2
import numpy as np

from src.data.extract_frames import extract_frames, has_complete_frame_set, save_frames


def _write_video(path: Path, frame_count: int = 10) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def test_extracts_uniform_number_of_rgb_frames(tmp_path):
    video = tmp_path / "sample.avi"
    _write_video(video)
    frames = extract_frames(video, 4)
    assert len(frames) == 4
    assert frames[0].shape == (24, 32, 3)


def test_save_frames_uses_video_directory(tmp_path):
    video = tmp_path / "sample.avi"
    _write_video(video, 5)
    assert save_frames(video, tmp_path / "out", 3) == 3
    assert len(list((tmp_path / "out" / "sample").glob("*.jpg"))) == 3
    assert has_complete_frame_set(video, tmp_path / "out", 3)
    assert not has_complete_frame_set(video, tmp_path / "out", 4)
