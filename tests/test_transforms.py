import numpy as np

from src.data.transforms import build_transform, transform_options


def test_model_specific_transform_options_and_normalisation():
    config = {
        "training": {
            "image_size": 12,
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
            "interpolation": "bicubic",
        }
    }
    options = transform_options(config)
    tensor = build_transform(**options)(image=np.full((6, 6, 3), 255, dtype=np.uint8))["image"]
    assert tuple(tensor.shape) == (3, 12, 12)
    assert np.isclose(float(tensor.mean()), 1.0)
