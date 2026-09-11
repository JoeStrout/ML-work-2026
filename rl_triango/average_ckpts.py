"""Average the weights of several checkpoints (stochastic weight averaging).

    python average_ckpts.py checkpoints/az_0360.pt ... checkpoints/az_0400.pt -o avg.pt

Parameters are averaged.  BatchNorm running statistics are then recomputed
by running the averaged net over the newest --bn-samples replay-buffer
positions (with random symmetries), since averaged weights don't match
averaged statistics.  --bn-samples 0 keeps the averaged statistics.
A single checkpoint with --bn-samples > 0 just recomputes its statistics.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

from az.network import TrianGONet, load_checkpoint, save_checkpoint
from az.replay import ReplayBuffer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--bn-samples", type=int, default=20_000)
    ap.add_argument("--replay", default="checkpoints/replay.npz")
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = [load_checkpoint(p, device)[0].state_dict() for p in args.ckpts]
    arch = load_checkpoint(args.ckpts[0], device)[0].arch
    net = TrianGONet(**arch).to(device)
    net.load_state_dict({k: sum(s[k] for s in states) / len(states) if v.is_floating_point() else v
                         for k, v in states[0].items()})

    if args.bn_samples:
        buf = ReplayBuffer(args.bn_samples)   # keeps the newest positions
        buf.load(args.replay)
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None              # plain average over all batches
        net.train()
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for _ in range(max(1, len(buf) // args.batch_size)):
                x, _, _ = buf.sample(args.batch_size, rng)
                net(torch.from_numpy(x).to(device))
        net.eval()

    save_checkpoint(args.out, net, averaged_from=args.ckpts, bn_samples=args.bn_samples)
    print(f"averaged {len(args.ckpts)} checkpoint(s) -> {args.out}"
          + (f" (BN stats from {args.bn_samples} positions)" if args.bn_samples else " (BN stats averaged)"))


if __name__ == "__main__":
    main()
