import numpy as np

from src.eval.robustness import jpeg_compress


def test_jpeg_compress_shape_and_validation():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    assert jpeg_compress(image, 40).shape == image.shape

