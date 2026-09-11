# rl_triango

AlphaZero for TrianGO (Ken Knowlton, 1988), following the approach that
worked in `../rl_connect4` (see its `RECAP.md`). Two-player first, but
nothing below assumes two players.

## Game engine

These files are copied unchanged from `../../triango/python` (triango repo
commit `97b4aa2`); keep them in sync with upstream rather than editing
them here. The shared contract (point numbering, action ids, symmetries,
encoding) is documented in that folder's `README.md`.

- `board.py` — geometry, triangles, the 12 board symmetries, 9×9 grid embedding, rendering
- `game.py` — `GameState`: rules, legal actions, `play()`
- `encode.py` — 28-plane network input, legal-action masks, value targets, symmetry transforms
- `play.py` — terminal play (hot seat / random movers)
- `bench.py` — move-generation benchmark with behavior checksums
- `test_game.py`, `test_encode.py` — engine tests

## AlphaZero

- `az/network.py` — policy+value net: hex-masked 3×3 residual trunk on the 9×9
  grid; placement logits per point, capture logits pooled over each triangle;
  value = logits over the 4 seats relative to the player to move (absent
  seats masked), trained with cross-entropy on the winner's seat.
- `az/mcts.py` — PUCT MCTS with per-player value vectors: each node scores its
  edges from its own mover's point of view, so there is no sign flipping and
  3–4 players / skipped turns need no special handling. `run_search` drives
  many trees in lockstep so leaf evaluations are batched.
- `az/evaluators.py` — `NetEvaluator` (batched GPU) and `RolloutEvaluator`
  (uniform priors + one random playout).
- `az/agents.py` — `RandomAgent`, `MCTSAgent`; `net_agent`, `rollout_agent`.
- `az/arena.py` — parallel matches; games share random openings and rotate seats.
- `az/selfplay.py` — batched self-play (root Dirichlet noise, visit-count
  sampling for the first `temp_moves` plies).
- `az/replay.py` — replay buffer; each sample gets a random one of the 12 symmetries.
- `train.py` — training loop; every `Config` field is a flag.
- `eval.py` — head-to-head between any two agents.
- `play_az.py` — play against AlphaZero or rollout MCTS in the terminal.
- `average_ckpts.py` — average several checkpoints' weights (recomputing
  BatchNorm statistics from the replay buffer) into one weights-only file.
- `test_az.py` — tests for the above.

Training is pure self-play. Evaluation (every `--eval-every` iterations)
measures the net against three yardsticks that need no game knowledge:
random, plain rollout MCTS (`--rollout-sims`), and the net as of the
previous evaluation. It also plays the raw policy (the net's top move, no
search) against the previous evaluation's raw policy (`prev_raw`,
`--raw-eval-games`), since the MiniScript game may not afford much search.

## Commands

```bash
python -m unittest                                   # all tests
python train.py                                      # defaults: 100 iters, 64 games/iter, 200 sims
python train.py --resume checkpoints/az.pt --iterations 200
python eval.py checkpoints/az.pt rollout:2000 --games 40
python eval.py checkpoints/az.pt checkpoints/az_0050.pt
python play_az.py --human B --sims 800
python average_ckpts.py checkpoints/az_0{380,390,400}.pt -o checkpoints/az_avg0380-0400.pt
```

Averaging the last three snapshots of a run was worth about +20–30 Elo
over the final snapshot (iterations 380–400: 0.531 at 100 sims and 0.542
raw, over 800 games each). `checkpoints/az_avg0380-0400.pt` is the current
best net.

At the defaults one iteration takes about 50 s on an RTX 4090 (45 s
self-play, 3 s training), plus about 40 s per evaluation. Checkpoints go to
`checkpoints/` (`az.pt` latest, `az_NNNN.pt` every `--snapshot-every`),
with per-iteration metrics in `checkpoints/train_log.jsonl`. The replay
buffer is rewritten to `checkpoints/replay.npz` every iteration (~600 MB
when full), and `--resume` reloads the one in the resumed checkpoint's
directory.

Raw-policy strength (no search) is `eval.py A.pt:1 B.pt:1`.
