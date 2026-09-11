#!/usr/bin/env python3
"""Play TrianGO in the terminal against AlphaZero (or rollout MCTS).

Examples:
    python play_az.py                            # you are Red, AZ is Blue
    python play_az.py --human B --sims 800       # AZ moves first, thinks harder
    python play_az.py --opponent rollout --sims 2000
"""

import argparse
import time

import torch

from az.agents import net_agent, rollout_agent
from az.network import load_checkpoint
from board import CHAR_TO_PLAYER
from game import GameState
from play import HELP, describe, human_turn, name, show


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", "--players", type=int, default=2, choices=(2, 3, 4))
    ap.add_argument("--human", default="R", help="color(s) you play, e.g. R or RG")
    ap.add_argument("--opponent", default="az", choices=("az", "rollout"))
    ap.add_argument("--ckpt", default="checkpoints/az.pt")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    color = False if args.no_color else None
    if args.opponent == "az":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, _ = load_checkpoint(args.ckpt, device)
        agent = net_agent(net, device, args.sims)
    else:
        agent = rollout_agent(args.sims)

    state = GameState(args.players)
    try:
        humans = {CHAR_TO_PLAYER[c] for c in args.human.upper() if c not in ", "}
    except KeyError as e:
        ap.error(f"unknown color {e}; use R, B, G, or W")
    computer = set(state.players) - humans
    history = []
    print(HELP)
    show(state, color)
    while not state.is_over:
        p = state.to_move
        if p in computer:
            t0 = time.time()
            action = agent.choose([state])[0]
            print(f"({name(p, color)} thought for {time.time() - t0:.1f}s)")
        else:
            action = human_turn(state, history, computer, color)
            if action is None:
                continue
        history.append(state.copy())
        state.play(action)
        print()
        print(describe(state, p, action, color))
        for q in state.passed:
            print(f"{name(q, color)} has no legal move and is skipped.")
        show(state, color)
    print()
    print(f"Game over after {state.num_moves} moves.  {name(state.winner, color)} wins!")


if __name__ == "__main__":
    main()
