"""CoAtNet convolution/attention hybrid detector."""

from __future__ import annotations

from torch import nn


def build_hybrid_vit(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    import timm

    candidates = ("coatnet_0_rw_224", "coatnet_0_224")
    available = set(timm.list_models())
    name = next((candidate for candidate in candidates if candidate in available), None)
    if name is None:
        raise RuntimeError("Installed timm version does not provide CoAtNet-0")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

