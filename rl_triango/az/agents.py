"""Players.  Every agent chooses moves for a batch of states at once:
agent.choose(states) -> [action, ...]."""

from __future__ import annotations

import random

import numpy as np

from .evaluators import NetEvaluator, RolloutEvaluator
from .mcts import MCTSConfig, Tree, run_search


class RandomAgent:
    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        self.rng = random.Random(seed)
        self.name = name

    def choose(self, states):
        return [self.rng.choice(s.legal_actions()) for s in states]


class MCTSAgent:
    """Batched MCTS over any evaluator; plays the most-visited move."""

    def __init__(self, evaluator, cfg: MCTSConfig, seed: int | None = None,
                 name: str = "mcts") -> None:
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.name = name

    def search(self, states, add_noise: bool = False):
        """Search each state; returns [(legal_actions, root_visit_counts), ...].
        Positions with a single legal move are not searched."""
        results = [None] * len(states)
        trees, where = [], []
        for k, s in enumerate(states):
            actions = s.legal_actions()
            if len(actions) == 1:
                results[k] = (actions, np.ones(1))
            else:
                trees.append(Tree(s, self.cfg, add_noise, self.rng))
                where.append(k)
        if trees:
            run_search(trees, self.evaluator, self.cfg.sims)
        for k, t in zip(where, trees):
            results[k] = t.root_visits()
        return results

    def choose(self, states):
        return [actions[int(np.argmax(n))] for actions, n in self.search(states)]


def net_agent(net, device, sims: int, name: str = "az", **mcts_kwargs) -> MCTSAgent:
    return MCTSAgent(NetEvaluator(net, device), MCTSConfig(sims=sims, **mcts_kwargs), name=name)


def rollout_agent(sims: int, seed: int | None = None) -> MCTSAgent:
    """Plain MCTS with random playouts: the untrained yardstick.  Priors are
    uniform, so it needs a larger c_puct than the network does to explore."""
    cfg = MCTSConfig(sims=sims, c_puct=3.0, fpu_reduction=0.0)
    return MCTSAgent(RolloutEvaluator(seed), cfg, seed, name=f"rollout{sims}")
