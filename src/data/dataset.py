"""CSV-backed face image dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


REQUIRED_COLUMNS = {"image_path", "label"}


class FaceDataset(Dataset):
    """Load RGB face crops described by the project manifest schema."""

    def __init__(
        self,
        csv_file: str | Path,
        root: str | Path = ".",
        transform: Callable[..., dict[str, Any]] | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.root = Path(root)
        self.frame = pd.read_csv(self.csv_file)
        missing = REQUIRED_COLUMNS - set(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest {csv_file} is missing columns: {sorted(missing)}")
        labels = set(self.frame["label"].astype(int).unique())
        if not labels <= {0, 1}:
            raise ValueError(f"Labels must be 0 or 1, found {sorted(labels)}")
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.frame)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.resolve_path(str(row["image_path"]))
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        if not self.return_metadata:
            return image, label
        metadata = {
            column: ("" if pd.isna(row[column]) else str(row[column]))
            for column in self.frame.columns
            if column not in {"label"}
        }
        metadata["index"] = int(index)
        return image, label, metadata


def make_balanced_sampler(dataset: FaceDataset) -> WeightedRandomSampler:
    """Return inverse-frequency sample weights for binary class balancing."""
    labels = dataset.frame["label"].astype(int).to_numpy()
    counts = pd.Series(labels).value_counts().to_dict()
    weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

