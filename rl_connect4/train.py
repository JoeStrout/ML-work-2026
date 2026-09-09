"""DQN training loop for Connect-4 against a scripted opponent.

The learner alternates sides each episode so it experiences both first-
and second-player positions. Transitions are stored from the learner's
perspective; the opponent's reply is folded into the environment step
before the next-state is recorded (so `r - gamma * V(s')` is correct:
V(s') is the opponent's best value against us, which is our loss).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from connect4.agent import DQNAgent
from connect4.config import Config
from connect4.env import COLS, ROWS, Connect4Env
from connect4.network import QNet
from connect4.opponents import get_opponent
from connect4.replay import ReplayBuffer
from connect4.selfplay import NetworkOpponent, SnapshotPool


def evaluate(agent: DQNAgent, opp_name: str, n_games: int,
             rng: np.random.Generator, temperature: float = 0.0) -> dict:
    opp = get_opponent(opp_name)
    wins = losses = draws = 0
    for g in range(n_games):
        env = Connect4Env()
        env.reset()
        agent_player = 1 if g % 2 == 0 else 2
        while not env.done:
            if env.current_player == agent_player:
                a = agent.act(env.observation(), env.legal_actions(), step=0,
                              rng=rng, greedy=True, temperature=temperature)
            else:
                a = opp.act(env, rng)
            env.step(a)
        if env.winner == agent_player:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1
    return {"win": wins / n_games, "loss": losses / n_games, "draw": draws / n_games}


def play_opponent_reply(env: Connect4Env, opp, rng: np.random.Generator) -> tuple[float, bool]:
    """Have opponent take a move if it's their turn and game not over.
    Returns (reward_to_learner_from_this_reply, done)."""
    if env.done:
        return 0.0, True
    a = opp.act(env, rng)
    _, r_opp, done, _ = env.step(a)
    # r_opp is +1 iff the opponent won; that is -1 for us.
    return -r_opp, done


class NStepAccumulator:
    """Turns a stream of (s, a, r) round-transitions into n-step transitions
    for the replay buffer. Each output stores gamma^k as the bootstrap
    discount, where k <= n is the actual rollout length (k < n only when the
    episode terminated inside the horizon)."""

    def __init__(self, n_step: int, gamma: float) -> None:
        self.n = n_step
        self.gamma = gamma
        self.pending: list[dict] = []

    def push(self, state, action, reward, next_state, next_legal, done, buffer) -> None:
        # Accumulate this reward into every already-pending transition.
        for e in self.pending:
            e["reward"] += e["discount"] * reward
            e["discount"] *= self.gamma
            e["count"] += 1

        # Add the freshly-taken action as a new pending transition.
        self.pending.append({
            "state": state, "action": action,
            "reward": reward, "discount": self.gamma, "count": 1,
        })

        if done:
            # Flush everything with discount=0 (no bootstrap past terminal).
            for e in self.pending:
                buffer.add(e["state"], e["action"], e["reward"],
                           next_state, True, next_legal, 0.0)
            self.pending.clear()
        else:
            # Anything that's collected n rewards is ripe.
            while self.pending and self.pending[0]["count"] >= self.n:
                e = self.pending.pop(0)
                buffer.add(e["state"], e["action"], e["reward"],
                           next_state, False, next_legal, e["discount"])

    def reset(self) -> None:
        self.pending.clear()


def train(cfg: Config, resume: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    agent = DQNAgent(cfg, device)
    if resume:
        agent.load(resume)
        print(f"resumed weights from {resume} "
              f"(eps schedule: {cfg.eps_start:.2f} -> {cfg.eps_end:.2f})")
    accum = NStepAccumulator(cfg.n_step, cfg.gamma)
    buffer = ReplayBuffer(cfg.replay_capacity, (2, ROWS, COLS), cfg.n_actions)

    selfplay = cfg.opponent == "selfplay"
    if selfplay:
        pool = SnapshotPool(cfg.snapshot_pool_size)
        pool.set_anchor(agent.online.state_dict())  # protect warm-start weights
        pool.add(agent.online.state_dict())
        opp_net = QNet(cfg.channels, cfg.hidden).to(device)
        net_opp = NetworkOpponent(opp_net, device, temperature=cfg.temperature)
        random_opp = get_opponent("random")
        oneply_opp = get_opponent("oneply")

        def sample_opp():
            r = rng.random()
            if r < cfg.sp_mix_random:
                return random_opp
            if r < cfg.sp_mix_random + cfg.sp_mix_oneply:
                return oneply_opp
            net_opp.load(pool.sample(rng))
            return net_opp

        opp = sample_opp()
    else:
        pool = None
        opp = get_opponent(cfg.opponent)

    env = Connect4Env()
    env.reset()
    agent_player = 1
    # If agent plays second, let opponent open.
    if agent_player == 2:
        play_opponent_reply(env, opp, rng)

    step = 0
    ep_returns: deque[float] = deque(maxlen=100)
    ep_return = 0.0
    losses: deque[float] = deque(maxlen=100)
    t0 = time.time()

    while step < cfg.total_steps:
        state = env.observation()
        legal = env.legal_actions()
        action = agent.act(state, legal, step, rng, greedy=False,
                           temperature=cfg.temperature)
        _, r_self, done, _ = env.step(action)
        reward = r_self  # +1 if we just won, else 0

        # Opponent replies (if game not over).
        r_from_opp, done = play_opponent_reply(env, opp, rng)
        reward += r_from_opp

        next_state = env.observation()
        next_legal = env.legal_actions() if not done else np.zeros(COLS, dtype=bool)

        accum.push(state, action, reward, next_state, next_legal, done, buffer)
        ep_return += reward
        step += 1

        if done:
            ep_returns.append(ep_return)
            ep_return = 0.0
            accum.reset()
            env.reset()
            agent_player = 3 - agent_player  # alternate sides
            if selfplay:
                opp = sample_opp()
            if agent_player == 2:
                play_opponent_reply(env, opp, rng)

        if selfplay and step > 0 and step % cfg.snapshot_every == 0:
            pool.add(agent.online.state_dict())

        if len(buffer) >= cfg.warmup_steps and step % cfg.train_every == 0:
            batch = buffer.sample(cfg.batch_size, rng)
            loss = agent.learn(batch)
            losses.append(loss)

        if step % cfg.target_sync_every == 0:
            agent.sync_target()

        if step % cfg.log_every == 0:
            avg_ret = np.mean(ep_returns) if ep_returns else float("nan")
            avg_loss = np.mean(losses) if losses else float("nan")
            sps = step / max(1e-6, time.time() - t0)
            explore = (f"T {cfg.temperature:.2f}" if cfg.temperature > 0
                       else f"eps {agent.epsilon(step):.3f}")
            print(f"step {step:>7d} | {explore} | ep_ret {avg_ret:+.3f} "
                  f"| loss {avg_loss:.4f} | {sps:.0f} steps/s")

        if step % cfg.eval_every == 0:
            for name in ("random", "oneply"):
                stats = evaluate(agent, name, cfg.eval_games, rng,
                                 temperature=cfg.temperature)
                print(f"  eval vs {name:6s}: win {stats['win']:.2f} "
                      f"draw {stats['draw']:.2f} loss {stats['loss']:.2f}")
            os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
            agent.save(cfg.ckpt_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", default="random",
                   choices=["random", "oneply", "selfplay"])
    p.add_argument("--total-steps", type=int, default=Config.total_steps)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=Config.ckpt_path,
                   help="where to save checkpoints")
    p.add_argument("--resume", default=None,
                   help="load weights from this checkpoint before training")
    p.add_argument("--eps-start", type=float, default=None,
                   help="override initial epsilon (default 1.0 fresh, 0.3 resumed)")
    p.add_argument("--eps-decay-steps", type=int, default=None,
                   help="override epsilon decay horizon in env steps")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    eps_start = a.eps_start if a.eps_start is not None else (0.3 if a.resume else Config.eps_start)
    eps_decay = a.eps_decay_steps if a.eps_decay_steps is not None else Config.eps_decay_steps
    cfg = Config(
        opponent=a.opponent,
        total_steps=a.total_steps,
        seed=a.seed,
        ckpt_path=a.ckpt,
        eps_start=eps_start,
        eps_decay_steps=eps_decay,
    )
    train(cfg, resume=a.resume)
