"""Evaluate a saved DQN checkpoint against a scripted opponent."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from connect4.agent import DQNAgent
from connect4.az_agent import AZPlayer
from connect4.az_network import AZNet
from connect4.config import Config
from connect4.env import Connect4Env
from connect4.network import QNet
from connect4.opponents import get_opponent
from connect4.selfplay import NetworkOpponent


def play_game(agent, opp, agent_player: int,
              rng: np.random.Generator, render: bool = False,
              temperature: float = 0.0) -> int:
    env = Connect4Env()
    env.reset()
    while not env.done:
        if env.current_player == agent_player:
            if isinstance(agent, AZPlayer):
                a = agent.act(env, rng)
            else:
                a = agent.act(env.observation(), env.legal_actions(),
                              step=0, rng=rng, greedy=True,
                              temperature=temperature)
        else:
            a = opp.act(env, rng)
        env.step(a)
        if render:
            print(env.render())
            print()
    return env.winner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=Config.ckpt_path)
    p.add_argument("--opponent", default="oneply",
                   choices=["random", "oneply", "ckpt", "az"],
                   help="'ckpt' pairs with --vs-ckpt (DQN head-to-head); "
                        "'az' pairs with --vs-ckpt for AlphaZero opponent")
    p.add_argument("--vs-ckpt", default=None,
                   help="opponent checkpoint (required if --opponent ckpt/az)")
    p.add_argument("--az", action="store_true",
                   help="treat --ckpt as an AlphaZero checkpoint")
    p.add_argument("--az-sims", type=int, default=100,
                   help="MCTS sims per move for AZ agents (both sides)")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Boltzmann temperature for both sides "
                        "(0=argmax; ~0.05-0.3 gives realistic diversity)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Our side.
    if args.az:
        az_net = AZNet().to(device)
        sd = torch.load(args.ckpt, map_location=device)
        az_net.load_state_dict(sd["net"])
        agent = AZPlayer(az_net, device, sims=args.az_sims)
    else:
        agent = DQNAgent(cfg, device)
        agent.load(args.ckpt)

    # Opponent side.
    if args.opponent == "ckpt":
        if not args.vs_ckpt:
            raise SystemExit("--vs-ckpt is required with --opponent ckpt")
        opp_net = QNet(cfg.channels, cfg.hidden).to(device)
        sd = torch.load(args.vs_ckpt, map_location=device)
        opp_net.load_state_dict(sd["online"])
        opp = NetworkOpponent(opp_net, device, temperature=args.temperature)
        opp_label = f"ckpt({args.vs_ckpt})"
    elif args.opponent == "az":
        if not args.vs_ckpt:
            raise SystemExit("--vs-ckpt is required with --opponent az")
        opp_net = AZNet().to(device)
        sd = torch.load(args.vs_ckpt, map_location=device)
        opp_net.load_state_dict(sd["net"])
        opp = AZPlayer(opp_net, device, sims=args.az_sims)
        opp_label = f"az({args.vs_ckpt})"
    else:
        opp = get_opponent(args.opponent)
        opp_label = args.opponent
    rng = np.random.default_rng(args.seed)

    wins = losses = draws = 0
    for g in range(args.games):
        ap = 1 if g % 2 == 0 else 2
        winner = play_game(agent, opp, ap, rng,
                           render=args.render and g < 2,
                           temperature=args.temperature)
        if winner == ap:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    n = args.games
    print(f"vs {opp_label}  ({n} games): "
          f"win {wins/n:.2f}  draw {draws/n:.2f}  loss {losses/n:.2f}")


if __name__ == "__main__":
    main()
