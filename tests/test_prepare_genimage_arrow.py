import csv
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from src.data.prepare_genimage_arrow import prepare_face_subset, read_arrow_rows


class FullFrameDetector:
    def detect(self, image):
        height, width = image.shape[:2]
        return np.asarray([[0, 0, width, height]], dtype=float), None


def _write_arrow(path: Path) -> None:
    rows = []
    for label in (0, 0, 1, 1):
        image = np.full((24, 30, 3), label * 100, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        rows.append(
            {
                "image_path": f"ADM/test/{'ai' if label else 'nature'}/{label}_{len(rows)}.png",
                "image": encoded.tobytes(),
                "label": label,
            }
        )
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def test_read_and_prepare_balanced_face_subset(tmp_path):
    arrow = tmp_path / "part.arrow"
    _write_arrow(arrow)
    assert len(read_arrow_rows(arrow)) == 4

    manifest = tmp_path / "manifest.csv"
    rows, missed = prepare_face_subset(
        arrow,
        tmp_path / "faces",
        manifest,
        FullFrameDetector(),
        max_per_class=1,
        image_size=16,
    )
    assert missed == 0
    assert sorted(int(row["label"]) for row in rows) == [0, 1]
    with manifest.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 2
    assert all(Path(row["image_path"]).is_file() for row in saved)
