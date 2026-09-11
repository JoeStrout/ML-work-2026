"""AlphaZero training for TrianGO.

One iteration: self-play games_per_iter games (batched MCTS with root
noise), add their positions to the replay buffer, then run train_steps
minibatch updates on randomly symmetrized samples from it.

Every eval_every iterations the net (MCTS, no noise) plays the random
agent, plain rollout MCTS, and the net as of the previous eval; then the
raw policy (sims=1, i.e. the net's top move) plays the previous eval's raw
policy ("prev_raw").  These are measurements only; training is pure
self-play.

The replay buffer is saved to ckpt_dir/replay.npz every iteration, and
--resume reloads the one in the resumed checkpoint's directory.

Every Config field is a command-line flag, e.g. --games-per-iter 32.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from az.agents import MCTSAgent, RandomAgent, net_agent, rollout_agent
from az.arena import win_rate
from az.evaluators import NetEvaluator
from az.mcts import MCTSConfig
from az.network import TrianGONet, save_checkpoint
from az.replay import ReplayBuffer
from az.selfplay import self_play


@dataclass
class Config:
    num_players: int = 2

    # Network
    channels: int = 128
    blocks: int = 6

    # MCTS (self-play)
    sims: int = 200
    c_puct: float = 1.25
    fpu_reduction: float = 0.1
    dirichlet_alpha: float = 0.3
    noise_eps: float = 0.25

    # Self-play
    games_per_iter: int = 64
    parallel_games: int = 64
    temp_moves: int = 12          # plies sampled from visit counts; greedy after

    # Training
    buffer_capacity: int = 150_000
    batch_size: int = 256
    train_steps: int = 250
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loop / eval
    iterations: int = 100
    eval_every: int = 5
    eval_games: int = 20
    eval_sims: int = 100
    rollout_sims: int = 400
    raw_eval_games: int = 200     # raw policy vs previous raw policy (fast)
    snapshot_every: int = 10
    ckpt_dir: str = "checkpoints"
    seed: int = 0


def train_steps(net, opt, buf: ReplayBuffer, cfg: Config, device, rng) -> dict:
    net.train()
    p_sum = v_sum = acc_sum = 0.0
    for _ in range(cfg.train_steps):
        x, pi, z = buf.sample(cfg.batch_size, rng)
        x = torch.from_numpy(x).to(device)
        pi = torch.from_numpy(pi).to(device)
        z = torch.from_numpy(z).to(device)
        logits, vlogits = net(x)
        p_loss = -(pi * F.log_softmax(logits, 1)).sum(1).mean()
        v_loss = F.cross_entropy(vlogits, z)
        opt.zero_grad()
        (p_loss + v_loss).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        p_sum += p_loss.item()
        v_sum += v_loss.item()
        acc_sum += (vlogits.argmax(1) == z).float().mean().item()
    n = cfg.train_steps
    return {"p_loss": p_sum / n, "v_loss": v_sum / n, "v_acc": acc_sum / n}


def evaluate(net, prev_net, cfg: Config, device, it: int) -> dict:
    """Win rates of the net against each yardstick (chance is 1/num_players)."""
    n = cfg.num_players
    az = net_agent(net, device, cfg.eval_sims, c_puct=cfg.c_puct, fpu_reduction=cfg.fpu_reduction)
    prev = net_agent(prev_net, device, cfg.eval_sims, c_puct=cfg.c_puct,
                     fpu_reduction=cfg.fpu_reduction, name="prev")
    opponents = {
        "random": RandomAgent(seed=it),
        f"rollout{cfg.rollout_sims}": rollout_agent(cfg.rollout_sims, seed=it),
        "prev": prev,
    }
    results = {name: win_rate([az] + [opp] * (n - 1), cfg.eval_games, seed=it)
               for name, opp in opponents.items()}
    raw = net_agent(net, device, 1, name="raw")
    prev_raw = net_agent(prev_net, device, 1, name="prev_raw")
    results["prev_raw"] = win_rate([raw] + [prev_raw] * (n - 1), cfg.raw_eval_games, seed=it)
    return results


def train(cfg: Config, resume: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.ckpt_dir, "train_log.jsonl")

    net = TrianGONet(cfg.channels, cfg.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start = 1
    if resume:
        ckpt = torch.load(resume, map_location=device)
        net = TrianGONet(**ckpt["arch"]).to(device)
        net.load_state_dict(ckpt["net"])
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            for group in opt.param_groups:   # the saved state carries the old lr
                group["lr"] = cfg.lr
                group["weight_decay"] = cfg.weight_decay
        start = ckpt.get("iteration", 0) + 1
        print(f"resumed from {resume} at iteration {start}")
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")
    with open(log_path, "a") as f:
        f.write(json.dumps({"config": dataclasses.asdict(cfg), "start": start}) + "\n")

    buf = ReplayBuffer(cfg.buffer_capacity)
    buf_path = os.path.join(cfg.ckpt_dir, "replay.npz")
    if resume:
        saved = os.path.join(os.path.dirname(resume) or ".", "replay.npz")
        if os.path.exists(saved):
            meta = buf.load(saved)
            print(f"loaded {len(buf):,} replay positions from {saved} "
                  f"(saved after iteration {meta.get('iteration', '?')})")
        else:
            print(f"no replay buffer at {saved}; starting empty")
    mcts_cfg = MCTSConfig(sims=cfg.sims, c_puct=cfg.c_puct, fpu_reduction=cfg.fpu_reduction,
                          dirichlet_alpha=cfg.dirichlet_alpha, noise_eps=cfg.noise_eps)
    prev_net = copy.deepcopy(net)

    for it in range(start, cfg.iterations + 1):
        t0 = time.time()
        agent = MCTSAgent(NetEvaluator(net, device), mcts_cfg, seed=cfg.seed * 100_003 + it)
        x, pi, z, sp = self_play(agent, cfg.games_per_iter, cfg.parallel_games,
                                 cfg.num_players, cfg.temp_moves, rng)
        buf.add(x, pi, z)
        sp_time = time.time() - t0

        t0 = time.time()
        tr = train_steps(net, opt, buf, cfg, device, rng)
        tr_time = time.time() - t0

        print(f"iter {it:>4d} | buf {len(buf):>6d} | p_loss {tr['p_loss']:.3f} "
              f"v_loss {tr['v_loss']:.3f} v_acc {tr['v_acc']:.2f} | "
              f"len {sp['mean_length']:.1f} caps {sp['mean_captures']:.1f} "
              f"p1 {sp['first_player_wins']:.2f} | sp {sp_time:.0f}s tr {tr_time:.0f}s",
              flush=True)
        record = {"iteration": it, **tr, **sp, "sp_time": sp_time, "tr_time": tr_time}

        extra = {"opt": opt.state_dict(), "iteration": it, "config": dataclasses.asdict(cfg)}
        if it % cfg.eval_every == 0 or it == cfg.iterations:
            t0 = time.time()
            ev = evaluate(net, prev_net, cfg, device, it)
            prev_net = copy.deepcopy(net)
            print("  eval  " + "  ".join(f"vs {k} {v:.2f}" for k, v in ev.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            record["eval"] = ev
        save_checkpoint(os.path.join(cfg.ckpt_dir, "az.pt"), net, **extra)
        buf.save(buf_path, iteration=it)
        if it % cfg.snapshot_every == 0:
            save_checkpoint(os.path.join(cfg.ckpt_dir, f"az_{it:04d}.pt"), net, **extra)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def parse_args() -> tuple[Config, str | None]:
    ap = argparse.ArgumentParser(description="AlphaZero training for TrianGO.")
    for f in dataclasses.fields(Config):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    args = vars(ap.parse_args())
    resume = args.pop("resume")
    return Config(**args), resume


if __name__ == "__main__":
    train(*parse_args())
