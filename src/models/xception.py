"""Xception transfer-learning model."""

from __future__ import annotations

from torch import nn


def build_xception(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    import timm

    available = set(timm.list_models())
    name = "xception" if "xception" in available else "legacy_xception"
    if name not in available:
        raise RuntimeError("Installed timm version does not provide Xception")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

