"""Dual spatial/spectral CNN inspired by SpecXNet."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, output_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(inputs))


class DualDomainCNN(nn.Module):
    """Fuse spatial RGB features with log FFT-magnitude features."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.spatial = SmallCNN(in_channels=3, output_dim=128)
        self.spectral = SmallCNN(in_channels=1, output_dim=128)
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    @staticmethod
    def fft_magnitude(rgb: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        spectrum = torch.fft.fftshift(torch.fft.fft2(gray.unsqueeze(1)), dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(spectrum))
        # Per-image standardisation prevents the DC component from dominating.
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (magnitude - mean) / std

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(inputs)
        spectral = self.spectral(self.fft_magnitude(inputs))
        return self.head(torch.cat([spatial, spectral], dim=1))

