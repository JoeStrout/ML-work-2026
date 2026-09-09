"""Self-play support: a fixed-size pool of past network snapshots, and a
network-backed opponent that reads its weights from a shared QNet."""

from __future__ import annotations

import numpy as np
import torch

from .env import Connect4Env
from .network import QNet


class NetworkOpponent:
    """QNet-driven opponent. Swap weights via `load(state_dict)` between
    episodes — no per-episode allocation. `temperature=0` → argmax; >0 →
    Boltzmann sampling from softmax(Q / T) over legal moves."""

    def __init__(self, net: QNet, device: torch.device,
                 temperature: float = 0.1) -> None:
        self.net = net
        self.net.eval()
        self.device = device
        self.temperature = temperature

    def load(self, state_dict: dict) -> None:
        self.net.load_state_dict(state_dict)

    @torch.no_grad()
    def act(self, env: Connect4Env, rng: np.random.Generator) -> int:
        s = torch.from_numpy(env.observation()).unsqueeze(0).to(self.device)
        q = self.net(s).squeeze(0).cpu().numpy()
        legal = env.legal_actions()
        q[~legal] = -np.inf
        if self.temperature <= 0.0:
            return int(np.argmax(q))
        q = q - np.max(q[legal])
        p = np.exp(q / self.temperature)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))


class SnapshotPool:
    """Ring buffer of past network state_dicts, held on CPU, plus an optional
    permanent `anchor` snapshot that is always available for sampling — used
    to keep an early competent checkpoint from ever being overwritten."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.snapshots: list[dict] = []
        self.idx = 0
        self.anchor: dict | None = None

    def __len__(self) -> int:
        return len(self.snapshots) + (1 if self.anchor is not None else 0)

    def set_anchor(self, state_dict: dict) -> None:
        self.anchor = {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def add(self, state_dict: dict) -> None:
        cpu_sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        if len(self.snapshots) < self.capacity:
            self.snapshots.append(cpu_sd)
        else:
            self.snapshots[self.idx] = cpu_sd
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, rng: np.random.Generator) -> dict:
        n = len(self)
        i = int(rng.integers(0, n))
        if self.anchor is not None:
            if i == 0:
                return self.anchor
            i -= 1
        return self.snapshots[i]
