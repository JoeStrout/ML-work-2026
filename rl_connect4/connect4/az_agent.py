"""Thin wrapper that turns an AZNet + MCTS into an evaluator / player."""

from __future__ import annotations

import numpy as np
import torch

from .env import Connect4Env
from .mcts import MCTS


class AZPlayer:
    """MCTS-driven greedy player (used for eval and play harness)."""

    def __init__(self, net, device: torch.device, sims: int = 100,
                 c_puct: float = 1.5) -> None:
        self.net = net
        self.net.eval()
        self.device = device
        self.mcts = MCTS(net, device, sims=sims, c_puct=c_puct)

    def act(self, env: Connect4Env, rng: np.random.Generator) -> int:
        counts = self.mcts.run(env, add_noise=False, rng=rng)
        return int(np.argmax(counts))
