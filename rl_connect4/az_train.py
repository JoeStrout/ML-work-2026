"""AlphaZero training loop for Connect-4.

One iteration = self-play a batch of games, append (state, pi, z) triples
to the replay buffer, then run several epochs of supervised training on
random minibatches drawn from the buffer.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from connect4.az_agent import AZPlayer
from connect4.az_network import AZNet
from connect4.env import COLS, Connect4Env
from connect4.mcts import MCTS
from connect4.opponents import get_opponent


@dataclass
class AZConfig:
    # Network
    channels: int = 128
    blocks: int = 4

    # MCTS
    sims_per_move: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    root_noise_eps: float = 0.25

    # Self-play
    games_per_iter: int = 25
    temp_moves: int = 10           # sample from visit counts for this many opening plies
    eval_sims: int = 100           # sims per move at eval time

    # Training
    buffer_capacity: int = 30_000  # (state, pi, z) positions
    batch_size: int = 128
    train_batches_per_iter: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0

    # Loop
    iterations: int = 60
    eval_every: int = 5            # iterations between evals
    eval_games: int = 40
    ckpt_path: str = "checkpoints/az.pt"
    seed: int = 0


def self_play_game(mcts: MCTS, cfg: AZConfig, rng: np.random.Generator):
    """Play one game with MCTS. Returns (trajectory, winner) where trajectory
    is a list of (state, pi_target, player_at_state)."""
    env = Connect4Env()
    env.reset()
    traj: list[tuple[np.ndarray, np.ndarray, int]] = []
    move = 0
    while not env.done:
        counts = mcts.run(env, add_noise=True, rng=rng)
        pi = counts / counts.sum()
        traj.append((env.observation().copy(), pi, env.current_player))
        if move < cfg.temp_moves:
            a = int(rng.choice(COLS, p=pi))
        else:
            a = int(np.argmax(counts))
        env.step(a)
        move += 1
    winner = env.winner
    return [(s, pi, player) for (s, pi, player) in traj], winner


def augment(state: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal mirror — Connect-4's only symmetry."""
    return state[:, :, ::-1].copy(), pi[::-1].copy()


def train_iter(net: AZNet, opt: torch.optim.Optimizer, buffer: deque,
               cfg: AZConfig, device: torch.device,
               rng: np.random.Generator) -> tuple[float, float]:
    net.train()
    total_p_loss = total_v_loss = 0.0
    for _ in range(cfg.train_batches_per_iter):
        idx = rng.integers(0, len(buffer), size=cfg.batch_size)
        states = np.stack([buffer[i][0] for i in idx])
        pis = np.stack([buffer[i][1] for i in idx])
        zs = np.array([buffer[i][2] for i in idx], dtype=np.float32)

        s = torch.from_numpy(states).to(device)
        pi_t = torch.from_numpy(pis).to(device)
        z_t = torch.from_numpy(zs).to(device)

        p_logits, v = net(s)
        log_p = torch.log_softmax(p_logits, dim=1)
        p_loss = -(pi_t * log_p).sum(dim=1).mean()
        v_loss = (v - z_t).pow(2).mean()
        loss = p_loss + cfg.value_weight * v_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        total_p_loss += float(p_loss.item())
        total_v_loss += float(v_loss.item())
    n = cfg.train_batches_per_iter
    return total_p_loss / n, total_v_loss / n


def evaluate(net: AZNet, cfg: AZConfig, device: torch.device,
             rng: np.random.Generator, opp_name: str, n_games: int) -> dict:
    az = AZPlayer(net, device, sims=cfg.eval_sims, c_puct=cfg.c_puct)
    opp = get_opponent(opp_name)
    wins = losses = draws = 0
    for g in range(n_games):
        env = Connect4Env()
        env.reset()
        az_player = 1 if g % 2 == 0 else 2
        while not env.done:
            if env.current_player == az_player:
                a = az.act(env, rng)
            else:
                a = opp.act(env, rng)
            env.step(a)
        if env.winner == az_player:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1
    return {"win": wins / n_games, "loss": losses / n_games, "draw": draws / n_games}


def train(cfg: AZConfig, resume: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    net = AZNet(cfg.channels, cfg.blocks).to(device)
    if resume:
        sd = torch.load(resume, map_location=device)
        net.load_state_dict(sd["net"])
        print(f"resumed from {resume}")
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    buffer: deque = deque(maxlen=cfg.buffer_capacity)

    for it in range(1, cfg.iterations + 1):
        t0 = time.time()
        # Self-play with net in eval mode (BN uses running stats).
        net.eval()
        mcts = MCTS(net, device, sims=cfg.sims_per_move, c_puct=cfg.c_puct,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    root_noise_eps=cfg.root_noise_eps)
        n_positions = n_wins_first = 0
        for _ in range(cfg.games_per_iter):
            traj, winner = self_play_game(mcts, cfg, rng)
            for state, pi, player in traj:
                z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
                buffer.append((state.astype(np.float32), pi.astype(np.float32), z))
                s_aug, pi_aug = augment(state, pi)
                buffer.append((s_aug.astype(np.float32), pi_aug.astype(np.float32), z))
                n_positions += 1
            if winner == 1:
                n_wins_first += 1
        sp_time = time.time() - t0

        # Training pass.
        t0 = time.time()
        if len(buffer) >= cfg.batch_size:
            p_loss, v_loss = train_iter(net, opt, buffer, cfg, device, rng)
        else:
            p_loss = v_loss = float("nan")
        tr_time = time.time() - t0

        print(f"iter {it:>3d} | buf {len(buffer):>6d} | "
              f"p_loss {p_loss:.3f} v_loss {v_loss:.3f} | "
              f"p1_wins {n_wins_first}/{cfg.games_per_iter} | "
              f"sp {sp_time:.1f}s tr {tr_time:.1f}s")

        if it % cfg.eval_every == 0 or it == cfg.iterations:
            for opp_name in ("random", "oneply"):
                stats = evaluate(net, cfg, device, rng, opp_name, cfg.eval_games)
                print(f"  eval vs {opp_name:6s}: win {stats['win']:.2f} "
                      f"draw {stats['draw']:.2f} loss {stats['loss']:.2f}")
            os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
            torch.save({"net": net.state_dict()}, cfg.ckpt_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=AZConfig.iterations)
    p.add_argument("--sims", type=int, default=AZConfig.sims_per_move)
    p.add_argument("--games-per-iter", type=int, default=AZConfig.games_per_iter)
    p.add_argument("--ckpt", default=AZConfig.ckpt_path)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = AZConfig(
        iterations=a.iterations,
        sims_per_move=a.sims,
        games_per_iter=a.games_per_iter,
        ckpt_path=a.ckpt,
        seed=a.seed,
    )
    train(cfg, resume=a.resume)
