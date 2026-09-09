"""PUCT MCTS for AlphaZero-style search.

Value convention: v(node) is the expected outcome for the player *about to
move* at that node. Values are negated across parent/child boundaries.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .env import COLS, Connect4Env


class MCTSNode:
    __slots__ = ("prior", "children", "N", "W", "is_expanded",
                 "is_terminal", "terminal_value")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.children: dict[int, "MCTSNode"] = {}
        self.N = 0
        self.W = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0

    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(self, net, device: torch.device, sims: int,
                 c_puct: float = 1.5, dirichlet_alpha: float = 1.0,
                 root_noise_eps: float = 0.25) -> None:
        self.net = net
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise_eps = root_noise_eps

    @torch.no_grad()
    def _evaluate(self, env: Connect4Env) -> tuple[np.ndarray, float]:
        obs = torch.from_numpy(env.observation()).unsqueeze(0).to(self.device)
        p_logits, v = self.net(obs)
        legal = env.legal_actions()
        p = p_logits.squeeze(0).cpu().numpy()
        p[~legal] = -1e9
        p = p - p.max()
        p = np.exp(p)
        p = p / p.sum()
        return p, float(v.item())

    def _select_child(self, parent: MCTSNode) -> tuple[int, MCTSNode]:
        total_N = sum(c.N for c in parent.children.values())
        sqrt_total = math.sqrt(total_N + 1)
        best_score = -float("inf")
        best_a, best_child = -1, None
        for a, c in parent.children.items():
            # Q is from child's perspective; parent chooses so negate.
            q = -c.q()
            u = self.c_puct * c.prior * sqrt_total / (1 + c.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
                best_child = c
        return best_a, best_child  # type: ignore[return-value]

    def run(self, root_env: Connect4Env, add_noise: bool,
            rng: np.random.Generator) -> np.ndarray:
        """Return visit-count vector over COLS after `sims` simulations."""
        root = MCTSNode(prior=1.0)
        p, _ = self._evaluate(root_env)
        legal = root_env.legal_actions()

        if add_noise:
            legal_idx = np.flatnonzero(legal)
            noise = rng.dirichlet([self.dirichlet_alpha] * len(legal_idx))
            for i, a in enumerate(legal_idx):
                p[a] = (1 - self.root_noise_eps) * p[a] + self.root_noise_eps * noise[i]

        for a in np.flatnonzero(legal):
            root.children[int(a)] = MCTSNode(prior=float(p[a]))
        root.is_expanded = True

        for _ in range(self.sims):
            self._simulate(root, root_env.clone())

        counts = np.zeros(COLS, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.N
        return counts

    def _simulate(self, root: MCTSNode, env: Connect4Env) -> None:
        path = [root]
        node = root

        # Selection: descend until we hit a leaf (unexpanded or terminal).
        while node.is_expanded and not node.is_terminal:
            a, node = self._select_child(node)
            _, r, done, _ = env.step(a)
            path.append(node)
            if done:
                node.is_terminal = True
                # r=1 iff the player who just moved won; that player was
                # `node`'s parent's player, so `node`'s player receives -r.
                node.terminal_value = -r

        # Evaluation.
        if node.is_terminal:
            v = node.terminal_value
        else:
            p, v = self._evaluate(env)
            for a in np.flatnonzero(env.legal_actions()):
                node.children[int(a)] = MCTSNode(prior=float(p[a]))
            node.is_expanded = True

        # Backpropagation with per-level sign flip.
        for anc in reversed(path):
            anc.N += 1
            anc.W += v
            v = -v
