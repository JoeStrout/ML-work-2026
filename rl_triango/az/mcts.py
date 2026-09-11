"""PUCT Monte Carlo tree search for 2-4 players.

Evaluations return a value *vector*: one win probability per player
(index player - 1).  Each node belongs to the player to move there, and
its edge statistics are from that player's point of view: W[i] sums the
mover's component of the value of every simulation through actions[i].
There is no sign flipping, so the same code handles any number of players
and turns skipped by players who cannot move.

Searches are driven in lockstep over many trees (run_search) so that leaf
evaluations can be batched: each step, every tree descends to one leaf,
the evaluator scores all the leaves at once, and each tree backs up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from encode import MAX_PLAYERS


@dataclass
class MCTSConfig:
    sims: int = 200              # root visits per search
    c_puct: float = 1.25         # values are in [0, 1]
    fpu_reduction: float = 0.1   # unvisited child Q = parent's mean Q minus this
    dirichlet_alpha: float = 0.3
    noise_eps: float = 0.25      # root prior mix when add_noise is on


def outcome(winner: int) -> np.ndarray:
    """Value vector for a finished game."""
    v = np.zeros(MAX_PLAYERS)
    v[winner - 1] = 1.0
    return v


class Node:
    __slots__ = ("mover", "actions", "P", "N", "W", "total", "wsum", "children", "terminal")

    def __init__(self) -> None:
        self.actions = None    # list of legal actions, once expanded
        self.children = {}     # index into actions -> Node
        self.terminal = None   # value vector if the game is over here

    def expand(self, mover: int, actions: list[int], priors: np.ndarray,
               value: np.ndarray) -> None:
        self.mover = mover
        self.actions = actions
        self.P = priors
        self.N = np.zeros(len(actions))
        self.W = np.zeros(len(actions))
        self.total = 0
        self.wsum = value[mover - 1]   # includes this node's own evaluation

    def select(self, c_puct: float, fpu_reduction: float) -> int:
        fpu = self.wsum / (self.total + 1) - fpu_reduction
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), fpu)
        u = (c_puct * math.sqrt(self.total + 1)) * self.P / (1 + self.N)
        return int(np.argmax(q + u))


def _backup(path: list[tuple[Node, int]], value: np.ndarray) -> None:
    for node, i in path:
        v = value[node.mover - 1]
        node.N[i] += 1
        node.W[i] += v
        node.total += 1
        node.wsum += v


class Tree:
    """Search tree for one position.  The state is never mutated."""

    def __init__(self, state, cfg: MCTSConfig, add_noise: bool = False,
                 rng: np.random.Generator | None = None) -> None:
        if state.is_over:
            raise ValueError("cannot search a finished game")
        self.state = state
        self.cfg = cfg
        self.add_noise = add_noise
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = Node()
        self._pending = None

    @property
    def visits(self) -> int:
        return self.root.total

    def select_leaf(self):
        """Descend to an unexpanded node and return its state for evaluation.
        If the descent ends in a finished game, back it up and return None."""
        node, state, path = self.root, self.state.copy(), []
        c_puct, fpu_reduction = self.cfg.c_puct, self.cfg.fpu_reduction
        while node.actions is not None:
            i = node.select(c_puct, fpu_reduction)
            path.append((node, i))
            state.play(node.actions[i])
            child = node.children.get(i)
            if child is None:
                child = node.children[i] = Node()
            node = child
            if state.is_over:
                if node.terminal is None:
                    node.terminal = outcome(state.winner)
                _backup(path, node.terminal)
                return None
        self._pending = (node, path)
        return state

    def expand(self, state, actions: list[int], priors: np.ndarray, value: np.ndarray) -> None:
        """Finish the simulation started by select_leaf with its evaluation."""
        node, path = self._pending
        self._pending = None
        if node is self.root and self.add_noise and len(actions) > 1:
            noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(actions))
            priors = (1 - self.cfg.noise_eps) * priors + self.cfg.noise_eps * noise
        node.expand(state.to_move, actions, priors, value)
        _backup(path, value)

    def root_visits(self) -> tuple[list[int], np.ndarray]:
        return self.root.actions, self.root.N.copy()


def run_search(trees: list[Tree], evaluator, sims: int) -> None:
    """Search every tree until its root has `sims` visits.

    evaluator(states) -> [(legal_actions, priors, value_vector), ...]
    """
    # The root's own expansion isn't a root visit, hence the extra step.
    while True:
        pending = []
        for t in trees:
            if t.root.actions is None or t.visits < sims:
                s = t.select_leaf()
                if s is not None:
                    pending.append((t, s))
        if pending:
            for (t, s), result in zip(pending, evaluator([s for _, s in pending])):
                t.expand(s, *result)
        elif all(t.visits >= sims for t in trees):
            return
