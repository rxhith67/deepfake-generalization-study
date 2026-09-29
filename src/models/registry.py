"""Central model registry used by training, evaluation, and interpretability."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .freq_cnn import DualDomainCNN
from .hybrid import build_hybrid_vit
from .meso import MesoInception4
from .xception import build_xception


def _meso(num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    del pretrained
    return MesoInception4(num_classes=num_classes)


def _frequency(num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    del pretrained
    return DualDomainCNN(num_classes=num_classes)


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "meso": _meso,
    "xception": build_xception,
    "freq_cnn": _frequency,
    "hybrid": build_hybrid_vit,
}


def build_model(name: str, num_classes: int = 1, pretrained: bool = False, **_: object) -> nn.Module:
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from error
    return builder(num_classes=num_classes, pretrained=pretrained)


def freeze_first_fraction(model: nn.Module, fraction: float = 0.60) -> int:
    """Freeze the first fraction of leaf modules that own parameters."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    leaves = [module for module in model.modules() if len(list(module.children())) == 0 and list(module.parameters(recurse=False))]
    cutoff = int(len(leaves) * fraction)
    for module in leaves[:cutoff]:
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = False
    return cutoff


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

