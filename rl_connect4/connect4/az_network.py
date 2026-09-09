"""AlphaZero policy+value network for Connect-4.

Two-headed: shared conv trunk (residual blocks) feeds a policy head (COLS
logits) and a scalar value head (tanh, expected outcome for current player).
"""

from __future__ import annotations

import torch
from torch import nn

from .env import COLS, ROWS


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class AZNet(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * ROWS * COLS, COLS)

        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(ROWS * COLS, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.stem(x))
        p = torch.relu(self.p_bn(self.p_conv(h))).flatten(1)
        p_logits = self.p_fc(p)
        v = torch.relu(self.v_bn(self.v_conv(h))).flatten(1)
        v = torch.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return p_logits, v
