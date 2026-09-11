"""Head-to-head evaluation between two agents.

Agent specs:
    random                 uniform random legal moves
    rollout[:SIMS]         plain MCTS with random playouts (default --rollout-sims)
    PATH.pt[:SIMS]         AlphaZero checkpoint, MCTS at SIMS (default --sims)

Examples:
    python eval.py checkpoints/az.pt random
    python eval.py checkpoints/az.pt rollout:2000 --games 40
    python eval.py checkpoints/az.pt checkpoints/az_0050.pt --sims 200

Games come in pairs sharing a random opening (--opening plies) with the
agents swapping who moves first.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from az.agents import RandomAgent, net_agent, rollout_agent
from az.arena import play_games
from az.network import load_checkpoint


def make_agent(spec: str, args, device, seed: int):
    name, _, sims = spec.rpartition(":")
    if not (sims.isdigit() and name):
        name, sims = spec, None
    if name == "random":
        return RandomAgent(seed)
    if name == "rollout":
        return rollout_agent(int(sims or args.rollout_sims), seed)
    net, _ = load_checkpoint(name, device)
    return net_agent(net, device, int(sims or args.sims), name=spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--rollout-sims", type=int, default=400)
    ap.add_argument("--opening", type=int, default=2, help="random opening plies")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = make_agent(args.a, args, device, args.seed)
    b = make_agent(args.b, args, device, args.seed + 1)

    t0 = time.time()
    winners = play_games([a, b], args.games, args.opening, args.seed)
    n = len(winners)
    wins = sum(w == 0 for w in winners)
    rate = wins / n
    ci = 1.96 * math.sqrt(rate * (1 - rate) / n)
    first = [w == 0 for g, w in enumerate(winners) if g % 2 == 0]
    second = [w == 0 for g, w in enumerate(winners) if g % 2 == 1]
    print(f"{args.a} vs {args.b}: {wins}/{n} = {rate:.3f} ± {ci:.3f}  "
          f"(as first {sum(first)}/{len(first)}, as second {sum(second)}/{len(second)})  "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
