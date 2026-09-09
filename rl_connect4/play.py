"""Interactive play harness: human vs. a chosen opponent.

Defaults to the AlphaZero net, which is by far the strongest player here.

Examples:
    python play.py                                  # vs AlphaZero
    python play.py --az-sims 400 --as O             # stronger, AZ moves first
    python play.py --opponent oneply
    python play.py --opponent dqn --ckpt checkpoints/dqn_selfplay.pt
"""

from __future__ import annotations

import argparse

import numpy as np

from connect4.env import COLS, Connect4Env
from connect4.opponents import get_opponent

DEFAULT_AZ_CKPT = "checkpoints/az.pt"


def make_opponent(name: str, ckpt: str | None, temperature: float,
                  az_sims: int = 100):
    if name == "dqn":
        import torch
        from connect4.agent import DQNAgent
        from connect4.config import Config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = DQNAgent(Config(), device)
        if not ckpt:
            raise SystemExit("--ckpt is required with --opponent dqn")
        agent.load(ckpt)

        class DQNOpp:
            def act(self, env: Connect4Env, rng: np.random.Generator) -> int:
                return agent.act(env.observation(), env.legal_actions(),
                                 step=0, rng=rng, greedy=True,
                                 temperature=temperature)
        return DQNOpp()
    if name == "az":
        import torch
        from connect4.az_agent import AZPlayer
        from connect4.az_network import AZNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = ckpt or DEFAULT_AZ_CKPT
        net = AZNet().to(device)
        sd = torch.load(ckpt, map_location=device)
        net.load_state_dict(sd["net"])
        return AZPlayer(net, device, sims=az_sims)
    return get_opponent(name)


def prompt_human(env: Connect4Env) -> int:
    legal = np.flatnonzero(env.legal_actions()).tolist()
    while True:
        raw = input(f"your move (columns {legal}, q to quit): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            raise SystemExit(0)
        try:
            a = int(raw)
        except ValueError:
            print("  please enter a column number.")
            continue
        if a not in legal:
            print(f"  column {a} is not legal.")
            continue
        return a


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", default="az",
                   choices=["random", "oneply", "dqn", "az"])
    p.add_argument("--ckpt", default=None,
                   help="checkpoint path for --opponent dqn/az "
                        f"(default {DEFAULT_AZ_CKPT} for az)")
    p.add_argument("--az-sims", type=int, default=100,
                   help="MCTS sims per move for --opponent az")
    p.add_argument("--as", dest="human_glyph", default="X", choices=["X", "O"],
                   help="X moves first, O second")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Boltzmann temperature for the DQN opponent")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    opp = make_opponent(args.opponent, args.ckpt, args.temperature,
                        az_sims=args.az_sims)
    human_player = 1 if args.human_glyph == "X" else 2

    env = Connect4Env()
    env.reset()
    print(f"You are {args.human_glyph} vs {args.opponent}. Columns 0..{COLS-1}.\n")
    print(env.render(), "\n")

    while not env.done:
        if env.current_player == human_player:
            a = prompt_human(env)
        else:
            a = opp.act(env, rng)
            print(f"opponent plays column {a}")
        env.step(a)
        print(env.render(), "\n")

    if env.winner == 0:
        print("draw.")
    elif env.winner == human_player:
        print("you win!")
    else:
        print("you lose.")


if __name__ == "__main__":
    main()
