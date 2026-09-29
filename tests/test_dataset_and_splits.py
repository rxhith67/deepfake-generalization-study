import cv2
import numpy as np
import pandas as pd

from src.data.dataset import FaceDataset, make_balanced_sampler
from src.data.splits import split_manifest
from src.data.transforms import build_transform


def test_dataset_load_and_balanced_sampler(tmp_path):
    rows = []
    for index, label in enumerate([0, 0, 0, 1]):
        path = tmp_path / f"{index}.jpg"
        cv2.imwrite(str(path), np.full((20, 20, 3), index * 20, dtype=np.uint8))
        rows.append({"image_path": path.name, "label": label, "video_id": f"v{index}"})
    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    dataset = FaceDataset(csv_path, tmp_path, build_transform(32))
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert label.item() == 0
    assert len(list(make_balanced_sampler(dataset))) == 4


def test_group_split_has_no_video_leakage():
    frame = pd.DataFrame(
        [
            {"image_path": f"{video}_{frame}.jpg", "label": video % 2, "video_id": f"v{video}"}
            for video in range(20)
            for frame in range(2)
        ]
    )
    splits = split_manifest(frame, seed=9)
    sets = {name: set(part.video_id) for name, part in splits.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sum(map(len, splits.values())) == len(frame)

