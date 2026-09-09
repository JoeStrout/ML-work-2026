"""Uniform replay buffer for DQN transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    states: np.ndarray       # (B, 2, ROWS, COLS) float32
    actions: np.ndarray      # (B,) int64
    rewards: np.ndarray      # (B,) float32 — accumulated n-step return
    next_states: np.ndarray  # (B, 2, ROWS, COLS) float32
    dones: np.ndarray        # (B,) float32
    next_legal: np.ndarray   # (B, COLS) bool
    discounts: np.ndarray    # (B,) float32 — gamma^k for k actual rollout steps


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, int, int], n_actions: int) -> None:
        self.capacity = capacity
        self.n_actions = n_actions
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_legal = np.zeros((capacity, n_actions), dtype=bool)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal: np.ndarray,
        discount: float,
    ) -> None:
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.next_legal[i] = next_legal
        self.discounts[i] = discount
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        idx = rng.integers(0, self.size, size=batch_size)
        return Batch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
            next_legal=self.next_legal[idx],
            discounts=self.discounts[idx],
        )
