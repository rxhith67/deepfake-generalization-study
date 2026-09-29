"""Albumentations pipelines shared by training and evaluation."""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_compression(quality_lower: int, quality_upper: int, p: float) -> A.BasicTransform:
    """Support both Albumentations 1.x and 2.x compression signatures."""
    try:
        return A.ImageCompression(
            quality_lower=quality_lower, quality_upper=quality_upper, p=p
        )
    except TypeError:  # Albumentations >= 2
        return A.ImageCompression(quality_range=(quality_lower, quality_upper), p=p)


def build_transform(
    image_size: int = 224,
    train: bool = False,
    augmentation: dict[str, Any] | None = None,
    jpeg_quality: int | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    interpolation: str = "linear",
) -> A.Compose:
    """Build a resize/augment pipeline with model-specific normalisation."""
    aug = augmentation or {}
    interpolation_code = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }.get(interpolation.lower())
    if interpolation_code is None:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    operations: list[A.BasicTransform] = [
        A.Resize(image_size, image_size, interpolation=interpolation_code)
    ]
    if train:
        jitter = float(aug.get("color_jitter", 0.2))
        q_min, q_max = aug.get("jpeg_quality_range", [30, 100])
        operations.extend(
            [
                A.HorizontalFlip(p=float(aug.get("hflip", 0.5))),
                A.Rotate(
                    limit=float(aug.get("rotate_limit", 10)),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                    hue=min(jitter / 2, 0.5),
                    p=0.5,
                ),
                _image_compression(int(q_min), int(q_max), float(aug.get("jpeg_prob", 0.5))),
            ]
        )
    elif jpeg_quality is not None and jpeg_quality < 100:
        operations.append(_image_compression(jpeg_quality, jpeg_quality, 1.0))
    operations.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(operations)


def denormalize_image(
    tensor: Any,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Convert a normalised CHW tensor into float RGB in [0, 1]."""
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray(std) + np.asarray(mean)
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def transform_options(config: dict[str, Any]) -> dict[str, Any]:
    """Read serialisable transform settings from a merged experiment config."""
    training = config.get("training", {})
    return {
        "image_size": int(training.get("image_size", 224)),
        "mean": tuple(float(value) for value in training.get("normalization_mean", IMAGENET_MEAN)),
        "std": tuple(float(value) for value in training.get("normalization_std", IMAGENET_STD)),
        "interpolation": str(training.get("interpolation", "linear")),
    }
