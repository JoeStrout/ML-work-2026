"""AlphaZero policy+value network for TrianGO.

Input is encode.encode(): (NUM_PLANES, 9, 9).  The 3x3 convolutions are
masked to the hex neighborhood (encode.HEX_KERNEL_MASK) and features are
zeroed off the board after every block, so the trunk is a hex-grid convnet.

Outputs:
  policy logits (NUM_ACTIONS,) in game.py's action numbering.  Placement
    logits come from a per-point 1x1 head; capture logits from head
    features pooled over each triangle's points and over its corners.
    Action 0 is never legal and gets a large negative logit.
  value logits (MAX_PLAYERS,) over seats relative to the player to move
    (seat 0 = to move).  Seats not in the game are masked out, so softmax
    gives each seat's probability of winning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from board import GRID_COLS, GRID_RC, NUM_POINTS, TRIANGLES
from encode import HEX_KERNEL_MASK, MAX_PLAYERS, NUM_PLANES, ON_BOARD, SEAT
from game import NUM_ACTIONS

NEG = -1e4  # logit for impossible outputs


class HexConv(nn.Conv2d):
    """3x3 convolution over a point and its six hex neighbors."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(cin, cout, 3, padding=1, bias=False)
        self.register_buffer("hex_mask", torch.from_numpy(HEX_KERNEL_MASK), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.hex_mask, None, padding=1)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = HexConv(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = HexConv(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y) * board


class TrianGONet(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 6,
                 head_channels: int = 32, value_channels: int = 8) -> None:
        super().__init__()
        self.arch = dict(channels=channels, blocks=blocks,
                         head_channels=head_channels, value_channels=value_channels)

        cells = [GRID_RC[i][0] * GRID_COLS + GRID_RC[i][1] for i in range(1, NUM_POINTS + 1)]
        self.register_buffer("point_cells", torch.tensor(cells), persistent=False)
        tri_points = torch.zeros(len(TRIANGLES), NUM_POINTS)
        tri_corners = torch.zeros(len(TRIANGLES), NUM_POINTS)
        for t in TRIANGLES:
            for p in t.points:
                tri_points[t.index, p - 1] = 1 / len(t.points)
            for p in t.corners:
                tri_corners[t.index, p - 1] = 1 / 3
        self.register_buffer("tri_points", tri_points, persistent=False)
        self.register_buffer("tri_corners", tri_corners, persistent=False)
        self.center = GRID_RC[28]

        self.stem = HexConv(NUM_PLANES, channels)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(ResBlock(channels) for _ in range(blocks))

        self.p_conv = nn.Conv2d(channels, head_channels, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(head_channels)
        self.place = nn.Conv1d(head_channels, 1, 1)
        self.capture = nn.Sequential(
            nn.Conv1d(2 * head_channels, head_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_channels, 1, 1),
        )

        self.v_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(value_channels)
        self.v_fc = nn.Sequential(
            nn.Linear(value_channels * NUM_POINTS + channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, MAX_PLAYERS),
        )

    def points(self, g: torch.Tensor) -> torch.Tensor:
        """(B, C, 9, 9) grid features -> (B, C, 55) point features."""
        return g.flatten(2)[:, :, self.point_cells]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        board = x[:, ON_BOARD:ON_BOARD + 1]
        h = torch.relu(self.stem_bn(self.stem(x))) * board
        for block in self.blocks:
            h = block(h, board)

        p = self.points(torch.relu(self.p_bn(self.p_conv(h))))           # (B, hc, 55)
        place = self.place(p).squeeze(1)                                  # (B, 55)
        tri = torch.cat([p @ self.tri_points.T, p @ self.tri_corners.T], 1)
        capture = self.capture(tri).squeeze(1)                            # (B, 136)
        logits = torch.cat([place.new_full((x.shape[0], 1), NEG), place, capture], 1)

        v = self.points(torch.relu(self.v_bn(self.v_conv(h)))).flatten(1)
        g = self.points(h).mean(2)
        v = self.v_fc(torch.cat([v, g], 1))
        present = x[:, SEAT:SEAT + MAX_PLAYERS, self.center[0], self.center[1]] > 0.5
        v = v.masked_fill(~present, NEG)
        return logits, v


assert NUM_ACTIONS == 1 + NUM_POINTS + len(TRIANGLES)


def save_checkpoint(path: str, net: TrianGONet, **extra) -> None:
    torch.save({"net": net.state_dict(), "arch": net.arch, **extra}, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[TrianGONet, dict]:
    ckpt = torch.load(path, map_location=device)
    net = TrianGONet(**ckpt["arch"]).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    return net, ckpt
