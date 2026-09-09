"""Hyperparameters. Tune here rather than in train.py."""

from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    n_actions: int = 7

    # Network / optimizer
    channels: int = 128
    hidden: int = 256
    lr: float = 3e-4
    grad_clip: float = 10.0

    # Replay
    replay_capacity: int = 100_000
    warmup_steps: int = 2_000
    batch_size: int = 256

    # DQN
    gamma: float = 0.99
    n_step: int = 3                 # multi-step return length
    target_sync_every: int = 1_000  # env steps
    train_every: int = 4            # env steps per gradient step
    double_dqn: bool = True

    # Exploration
    # Boltzmann temperature: when > 0, action selection is softmax(Q / T)
    # over legal moves, and the ε-greedy schedule below is bypassed.
    temperature: float = 0.1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000

    # Training loop
    total_steps: int = 300_000
    log_every: int = 2_000
    eval_every: int = 10_000
    eval_games: int = 100

    # Opponent schedule
    opponent: str = "random"        # "random", "oneply", or "selfplay"

    # Self-play
    snapshot_every: int = 10_000    # env steps between snapshots
    snapshot_pool_size: int = 20    # ring buffer of past selves
    sp_mix_random: float = 0.2      # fraction of self-play episodes vs random
    sp_mix_oneply: float = 0.2      # fraction vs oneply (remainder vs pool)

    # Misc
    seed: int = 0
    ckpt_path: str = "checkpoints/dqn.pt"
