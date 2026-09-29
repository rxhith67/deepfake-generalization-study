import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from src.data import detect_faces
from src.data.detect_faces import detect_and_crop_face, detect_and_crop_faces, expand_box


class StubDetector:
    def detect(self, image):
        return np.asarray([[5, 5, 25, 25], [10, 10, 50, 40]], dtype=float), None


class EmptyDetector:
    def detect(self, image):
        return None, None


class BatchDetector:
    def detect(self, images):
        boxes = np.empty(len(images), dtype=object)
        boxes[0] = np.asarray([[5, 5, 25, 25]], dtype=float)
        boxes[1] = None
        return boxes, None


def test_expand_box_clips_boundaries():
    assert expand_box([1, 2, 9, 10], width=10, height=10, margin=0.5) == (0, 0, 10, 10)


def test_crop_selects_largest_face_and_resizes():
    image = np.zeros((60, 70, 3), dtype=np.uint8)
    image[10:40, 10:50] = [255, 0, 0]
    crop = detect_and_crop_face(image, StubDetector(), image_size=32, margin=0)
    assert crop is not None
    assert crop.shape == (32, 32, 3)
    assert crop[..., 0].mean() > 200


def test_no_face_returns_none():
    assert detect_and_crop_face(np.zeros((20, 20, 3), dtype=np.uint8), EmptyDetector()) is None


def test_batch_detection_preserves_missing_faces():
    images = [np.zeros((30, 30, 3), dtype=np.uint8) for _ in range(2)]
    crops = detect_and_crop_faces(images, BatchDetector(), image_size=16, margin=0)
    assert crops[0].shape == (16, 16, 3)
    assert crops[1] is None


def test_cli_manifest_writes_absolute_crop_path(tmp_path, monkeypatch):
    source_dir = tmp_path / "frames" / "video_1"
    source_dir.mkdir(parents=True)
    cv2.imwrite(str(source_dir / "frame.jpg"), np.zeros((60, 70, 3), dtype=np.uint8))
    output = tmp_path / "faces"
    manifest = tmp_path / "manifest.csv"
    monkeypatch.setattr(detect_faces, "create_mtcnn", lambda device: StubDetector())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detect_faces",
            "--input",
            str(tmp_path / "frames"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--label",
            "1",
        ],
    )
    detect_faces.main()
    with manifest.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert (tmp_path / "faces" / "video_1" / "frame.jpg").resolve() == Path(row["image_path"])
