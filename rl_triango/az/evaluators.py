"""Leaf evaluators for MCTS.

An evaluator maps a list of (non-terminal) states to a list of
(legal_actions, priors, value_vector) tuples, where priors align with
legal_actions and value_vector[player - 1] is that player's win probability.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from encode import MAX_PLAYERS, encode, relative_players

from .mcts import outcome


class NetEvaluator:
    """Batched network evaluation."""

    def __init__(self, net, device: torch.device) -> None:
        self.net = net
        self.device = device

    @torch.no_grad()
    def __call__(self, states):
        self.net.eval()
        x = torch.from_numpy(np.stack([encode(s) for s in states])).to(self.device)
        logits, vlogits = self.net(x)
        logits = logits.float().cpu().numpy()
        values = torch.softmax(vlogits.float(), 1).cpu().numpy()
        out = []
        for s, lg, pv in zip(states, logits, values):
            actions = s.legal_actions()
            p = lg[actions]
            p = np.exp(p - p.max())
            p /= p.sum()
            v = np.zeros(MAX_PLAYERS)
            for k, player in enumerate(relative_players(s)):
                v[player - 1] = pv[k]
            out.append((actions, p, v))
        return out


class RolloutEvaluator:
    """Uniform priors; value from one random playout.  Needs no training,
    so MCTS with this is a fixed yardstick whose strength grows with sims."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def __call__(self, states):
        choice = self.rng.choice
        out = []
        for s in states:
            actions = s.legal_actions()
            t = s.copy()
            while not t.is_over:
                t.play(choice(t.legal_actions()))
            out.append((actions, np.full(len(actions), 1 / len(actions)), outcome(t.winner)))
        return out
