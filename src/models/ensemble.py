"""Probability-level ensemble for independently trained binary detectors."""

from __future__ import annotations

import torch
from torch import nn


class LogitAverageEnsemble(nn.Module):
    """Average probabilities and return an equivalent, numerically safe logit."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("An ensemble requires at least two models")
        self.models = nn.ModuleList(models)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = torch.stack([torch.sigmoid(model(inputs)) for model in self.models])
        mean_probability = probabilities.mean(dim=0).clamp(1e-6, 1 - 1e-6)
        return torch.logit(mean_probability)

