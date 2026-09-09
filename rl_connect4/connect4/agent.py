"""DQN agent: ε-greedy action selection + one-step Q-learning update.

Illegal actions are masked out at both action selection and bootstrap time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import Config
from .network import QNet
from .replay import Batch


class DQNAgent:
    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.online = QNet(cfg.channels, cfg.hidden).to(device)
        self.target = QNet(cfg.channels, cfg.hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)

    def epsilon(self, step: int) -> float:
        c = self.cfg
        frac = min(1.0, step / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        step: int,
        rng: np.random.Generator,
        greedy: bool = False,
        temperature: float = 0.0,
    ) -> int:
        """Select an action.

        temperature > 0 → Boltzmann sampling from softmax(Q / T) over legal moves
        (ignores greedy/eps). temperature == 0 → greedy or ε-greedy per `greedy`.
        """
        if temperature <= 0.0:
            eps = 0.0 if greedy else self.epsilon(step)
            legal_idx = np.flatnonzero(legal_mask)
            if not greedy and rng.random() < eps:
                return int(rng.choice(legal_idx))
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q = self.online(s).squeeze(0).cpu().numpy()
            q[~legal_mask] = -np.inf
            return int(np.argmax(q))

        s = torch.from_numpy(state).unsqueeze(0).to(self.device)
        q = self.online(s).squeeze(0).cpu().numpy()
        q[~legal_mask] = -np.inf
        # Numerically stable softmax; -inf entries → 0 probability.
        q = q - np.max(q[legal_mask])
        p = np.exp(q / temperature)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    def learn(self, batch: Batch) -> float:
        c = self.cfg
        d = self.device
        s = torch.from_numpy(batch.states).to(d)
        a = torch.from_numpy(batch.actions).to(d)
        r = torch.from_numpy(batch.rewards).to(d)
        s2 = torch.from_numpy(batch.next_states).to(d)
        done = torch.from_numpy(batch.dones).to(d)
        legal2 = torch.from_numpy(batch.next_legal).to(d)
        disc = torch.from_numpy(batch.discounts).to(d)

        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_target = self.target(s2)
            q_next_target = q_next_target.masked_fill(~legal2, float("-inf"))
            if c.double_dqn:
                q_next_online = self.online(s2).masked_fill(~legal2, float("-inf"))
                a_star = q_next_online.argmax(dim=1, keepdim=True)
                v_next = q_next_target.gather(1, a_star).squeeze(1)
            else:
                v_next = q_next_target.max(dim=1).values
            v_next = torch.where(torch.isfinite(v_next), v_next, torch.zeros_like(v_next))
            # n-step return: r is the accumulated k-step reward, disc is gamma^k.
            # (disc is 0 for transitions that hit terminal within the horizon.)
            target = r + disc * (1.0 - done) * v_next

        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.opt.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str) -> None:
        torch.save({"online": self.online.state_dict()}, path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device)
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["online"])
