"""MesoInception-4-style compact baseline."""

from __future__ import annotations

import torch
from torch import nn


class InceptionLayer(nn.Module):
    def __init__(self, in_channels: int, a: int, b: int, c: int, d: int) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, a, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, b, 1), nn.Conv2d(b, b, 3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c, 1), nn.Conv2d(c, c, 3, padding=2, dilation=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, d, 1), nn.Conv2d(d, d, 3, padding=3, dilation=3)
        )
        self.batch_norm = nn.BatchNorm2d(a + b + c + d)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                self.branch1(inputs),
                self.branch2(inputs),
                self.branch3(inputs),
                self.branch4(inputs),
            ],
            dim=1,
        )
        return torch.relu(self.batch_norm(features))


class MesoInception4(nn.Module):
    """Small detector returning one unnormalised fake-class logit per image."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.inception1 = InceptionLayer(3, 1, 4, 4, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.inception2 = InceptionLayer(11, 2, 4, 4, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(12, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4)
        self.feature_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 7 * 7, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool1(self.inception1(inputs))
        features = self.pool2(self.inception2(features))
        features = self.pool3(torch.relu(self.bn3(self.conv3(features))))
        features = self.pool4(torch.relu(self.bn4(self.conv4(features))))
        return self.head(self.feature_pool(features))

