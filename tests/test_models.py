import pytest
import torch

from src.models import MODEL_REGISTRY, build_model
from src.models.freq_cnn import DualDomainCNN


@pytest.mark.parametrize("name", ["meso", "freq_cnn"])
def test_local_model_output_shape(name):
    model = build_model(name, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 1)


def test_fft_features_are_finite_and_standardized():
    magnitude = DualDomainCNN.fft_magnitude(torch.randn(2, 3, 32, 32))
    assert torch.isfinite(magnitude).all()
    assert torch.allclose(magnitude.mean(dim=(-2, -1)), torch.zeros(2, 1), atol=1e-5)


def test_registry_has_four_required_models():
    assert set(MODEL_REGISTRY) == {"meso", "xception", "freq_cnn", "hybrid"}

