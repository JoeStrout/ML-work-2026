"""Baseline opponents to train and evaluate against."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import COLS, Connect4Env


class Opponent(Protocol):
    def act(self, env: Connect4Env, rng: np.random.Generator) -> int: ...


class RandomOpponent:
    def act(self, env: Connect4Env, rng: np.random.Generator) -> int:
        legal = np.flatnonzero(env.legal_actions())
        return int(rng.choice(legal))


class OnePlyOpponent:
    """Plays a winning move if available, blocks opponent's winning move,
    otherwise plays randomly."""

    def act(self, env: Connect4Env, rng: np.random.Generator) -> int:
        legal = np.flatnonzero(env.legal_actions())
        me = env.current_player

        # Winning move?
        for a in legal:
            probe = env.clone()
            _, _, done, info = probe.step(int(a))
            if done and info["winner"] == me:
                return int(a)

        # Block opponent's winning move?
        opp = 3 - me
        for a in legal:
            probe = env.clone()
            probe.current_player = opp  # pretend opponent moves
            _, _, done, info = probe.step(int(a))
            if done and info["winner"] == opp:
                return int(a)

        return int(rng.choice(legal))


def get_opponent(name: str) -> Opponent:
    name = name.lower()
    if name == "random":
        return RandomOpponent()
    if name in ("oneply", "one_ply", "heuristic"):
        return OnePlyOpponent()
    raise ValueError(f"unknown opponent: {name}")
