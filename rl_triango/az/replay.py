"""Replay buffer of (encoding, policy, winner seat) samples.

Positions are stored once in point form; each sampled position gets a
random one of the 12 board symmetries, so augmentation costs no memory.
"""

from __future__ import annotations

import os

import numpy as np

from board import NUM_POINTS, NUM_SYMMETRIES
from encode import NUM_PLANES, to_grid, transform_points, transform_policy
from game import NUM_ACTIONS


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.x = np.zeros((capacity, NUM_PLANES, NUM_POINTS), dtype=np.float16)
        self.pi = np.zeros((capacity, NUM_ACTIONS), dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.next = 0

    def __len__(self) -> int:
        return self.size

    def add(self, x: np.ndarray, pi: np.ndarray, z: np.ndarray) -> None:
        for k in range(len(z)):
            i = self.next
            self.x[i], self.pi[i], self.z[i] = x[k], pi[k], z[k]
            self.next = (i + 1) % self.capacity
        self.size = min(self.size + len(z), self.capacity)

    def save(self, path: str, **meta) -> None:
        """Write the contents, oldest first, plus any scalar metadata.
        Writes a temporary file and renames it, so an interrupted save
        never leaves a corrupt buffer at path."""
        if self.size < self.capacity:
            order = np.arange(self.size)
        else:
            order = np.r_[self.next:self.capacity, 0:self.next]
        tmp = path + ".tmp.npz"
        np.savez(tmp, x=self.x[order], pi=self.pi[order], z=self.z[order], **meta)
        os.replace(tmp, path)

    def load(self, path: str) -> dict:
        """Replace the contents with a saved buffer, keeping the newest
        `capacity` samples.  Returns the metadata saved with it."""
        with np.load(path) as f:
            x, pi, z = f["x"], f["pi"], f["z"]
            meta = {k: f[k].item() for k in f.files if k not in ("x", "pi", "z")}
        n = min(len(z), self.capacity)
        self.x[:n], self.pi[:n], self.z[:n] = x[len(z) - n:], pi[len(z) - n:], z[len(z) - n:]
        self.size = n
        self.next = n % self.capacity
        return meta

    def sample(self, n: int, rng: np.random.Generator):
        """(x grid (n, NUM_PLANES, 9, 9) float32, pi (n, NUM_ACTIONS), z (n,))."""
        idx = rng.integers(0, self.size, size=n)
        syms = rng.integers(0, NUM_SYMMETRIES, size=n)
        x = self.x[idx].astype(np.float32)
        pi = self.pi[idx]
        for s in range(1, NUM_SYMMETRIES):
            m = syms == s
            if m.any():
                x[m] = transform_points(x[m], s)
                pi[m] = transform_policy(pi[m], s)
        return to_grid(x), pi, self.z[idx]
