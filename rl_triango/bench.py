#!/usr/bin/env python3
"""Benchmark move generation and move application.

Times the core operations that any search or self-play code leans on,
over a fixed set of positions sampled from seeded random games:

    legal_actions   list the legal actions for the player to move
    copy            clone a GameState
    play            apply one action (includes finding who moves next)
    encode          encode a position for the network (needs numpy)
    playout         whole random games from the start (moves/sec)

It also prints a checksum of everything those operations returned, so an
optimization that changes behavior is caught immediately.

    python3 bench.py                      # run and print results
    python3 bench.py --save base.json     # ...and save them
    python3 bench.py --compare base.json  # show speedup vs. a saved run
    python3 bench.py --profile            # cProfile of random playouts
"""

import argparse
import hashlib
import json
import platform
import random
import sys
import time

from game import GameState


def sample_positions(num_players, num_games, seed):
    """All non-terminal positions from num_games seeded random games, each
    paired with the action that was played there."""
    rng = random.Random(seed)
    positions = []
    for _ in range(num_games):
        s = GameState(num_players, rng.choice(GameState(num_players).players))
        while not s.is_over:
            action = rng.choice(sorted(s.legal_actions()))
            positions.append((s.copy(), action))
            s.play(action)
    return positions


def state_key(s):
    return (s.board.to_string(), s.hand, s.captured, s.lost, s.to_move,
            s.winner, s.passed, s.num_moves)


def checksum(positions):
    h = hashlib.sha256()
    for s, action in positions:
        h.update(repr(sorted(s.legal_actions())).encode())
        t = s.copy()
        t.play(action)
        h.update(repr(state_key(t)).encode())
    return h.hexdigest()[:16]


def best_time(fn, reps, setup=None):
    """Minimum wall time of fn(arg) over reps runs; setup() makes arg untimed."""
    best = float("inf")
    for _ in range(reps):
        arg = setup() if setup else None
        t0 = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_players(num_players, args):
    positions = sample_positions(num_players, args.games, args.seed + num_players)
    states = [s for s, _ in positions]
    actions = [a for _, a in positions]
    n = len(positions)
    results = {"positions": n, "checksum": checksum(positions)}

    def run_legal(_):
        for s in states:
            s.legal_actions()

    def run_copy(_):
        for s in states:
            s.copy()

    def run_play(copies):
        for s, a in zip(copies, actions):
            s.play(a)

    results["legal_actions_us"] = best_time(run_legal, args.reps) / n * 1e6
    results["copy_us"] = best_time(run_copy, args.reps) / n * 1e6
    results["play_us"] = best_time(run_play, args.reps,
                                   setup=lambda: [s.copy() for s in states]) / n * 1e6
    try:
        import encode
    except ImportError:  # no numpy
        encode = None
    if encode:
        def run_encode(_):
            for s in states:
                encode.encode(s)
        results["encode_us"] = best_time(run_encode, args.reps) / n * 1e6

    # Whole random games, as a self-play-style workload.
    rng = random.Random(args.seed)
    moves = 0
    t0 = time.perf_counter()
    for _ in range(args.games):
        s = GameState(num_players)
        while not s.is_over:
            s.play(rng.choice(s.legal_actions()))
            moves += 1
    elapsed = time.perf_counter() - t0
    results["playout_moves_per_s"] = moves / elapsed
    results["playout_games_per_s"] = args.games / elapsed
    return results


COLUMNS = [
    ("legal_actions_us", "legal_actions", "µs", False),
    ("copy_us", "copy", "µs", False),
    ("play_us", "play", "µs", False),
    ("encode_us", "encode", "µs", False),
    ("playout_moves_per_s", "playout", "moves/s", True),
    ("playout_games_per_s", "playout", "games/s", True),
]


def print_results(results, baseline=None):
    for key, label, unit, higher_is_better in COLUMNS:
        cells = []
        for np_key, r in results.items():
            v = r.get(key)
            if v is None:
                cells.append(f"{'n/a':>10}" + (" " * 8 if baseline else ""))
                continue
            cell = f"{v:10.1f}" if v < 1e4 else f"{v:10.0f}"
            b = baseline.get(np_key, {}).get(key) if baseline else None
            if b:
                speedup = v / b if higher_is_better else b / v
                cell += f" ({speedup:4.2f}x)"
            elif baseline:
                cell += " " * 8
            cells.append(cell)
        print(f"{label:>14} {unit:>8}  " + "  ".join(cells))
    print(f"{'positions':>23}  " + "  ".join(
        f"{r['positions']:10d}" + (" " * 8 if baseline else "") for r in results.values()))
    for np_key, r in results.items():
        note = ""
        if baseline and np_key in baseline:
            note = ("  (matches baseline)" if r["checksum"] == baseline[np_key]["checksum"]
                    else "  ** DIFFERS FROM BASELINE -- behavior changed! **")
        print(f"  {np_key} checksum {r['checksum']}{note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-g", "--games", type=int, default=100,
                    help="random games per player count (default 100)")
    ap.add_argument("--reps", type=int, default=5, help="timing repetitions; best is kept")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("-p", "--players", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--save", metavar="FILE", help="save results as JSON")
    ap.add_argument("--compare", metavar="FILE", help="compare with saved results")
    ap.add_argument("--profile", action="store_true", help="profile random playouts instead")
    args = ap.parse_args()

    if args.profile:
        import cProfile
        import pstats
        rng = random.Random(args.seed)

        def playouts():
            for np_ in args.players:
                for _ in range(args.games):
                    s = GameState(np_)
                    while not s.is_over:
                        s.play(rng.choice(s.legal_actions()))

        prof = cProfile.Profile()
        prof.runcall(playouts)
        pstats.Stats(prof).sort_stats("tottime").print_stats(15)
        return

    print(f"Python {platform.python_version()} on {platform.machine()} "
          f"({platform.system()}), {args.games} games per player count, "
          f"best of {args.reps}")
    header = "".join(f"{str(n) + ' players':>{10 if i == 0 else 12}}"
                     + (" " * 8 if args.compare else "")
                     for i, n in enumerate(args.players))
    print(f"{'':>24}{header}")
    results = {f"{n}p": bench_players(n, args) for n in args.players}
    baseline = None
    if args.compare:
        with open(args.compare) as f:
            saved = json.load(f)
        baseline = saved["results"]
        if saved.get("settings") != {"games": args.games, "seed": args.seed}:
            print("  (note: baseline used different --games/--seed; checksums won't match)")
    print_results(results, baseline)
    if args.save:
        with open(args.save, "w") as f:
            json.dump({"python": platform.python_version(), "machine": platform.machine(),
                       "settings": {"games": args.games, "seed": args.seed},
                       "results": results}, f, indent=2)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    sys.exit(main())
