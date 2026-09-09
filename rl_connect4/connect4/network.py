"""Q-network: two-channel board -> 7 column Q-values."""

from __future__ import annotations

import torch
from torch import nn

from .env import COLS, ROWS


class QNet(nn.Module):
    def __init__(self, channels: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * ROWS * COLS, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, COLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))
