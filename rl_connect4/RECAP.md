# Connect-4 RL Recap

Chronological summary of the approaches tried, their results, and why each
next step was taken. Evaluation is win rate over 100–500 games with
alternating first player; head-to-head numbers use temperature 0.1 for the
DQN nets and MCTS-greedy for the AlphaZero net.

## 1. Scaffolding

Small package: `env`, `network`, `replay`, `agent`, `opponents`, `config`,
plus `train.py`, `eval.py`, and a `play.py` harness for playing by hand.
Two-channel board observation from the current player's perspective;
transitions stored per-round (learner move + opponent reply) so both `s`
and `s'` are from the learner's viewpoint and the standard
`r + γ V(s')` bootstrap applies.

## 2. Vanilla DQN vs random

- 3-layer conv Q-net (64 channels), double-DQN, ε-greedy, target net.
- Trained 300k steps.
- **Result: ~0.75–0.80 vs random**, plateauing with 20% loss rate — the
  net was winning when it had the initiative but not reliably *blocking*.

## 3. Bigger net + n-step returns

- Channels 64 → 128, added a 4th conv layer, head bottleneck 128 → 256.
- N-step return with n=3 (per-transition discount stored in the replay
  buffer).
- Trained 300k steps.
- **Result: ~0.80–0.88 vs random** — modest lift, still plateauing.
  Diagnosis: training against random provides poor signal for defense
  because random almost never sets up real threats.

## 4. Curriculum → oneply

Added `--resume` flag so the oneply run could warm-start from the
random-trained net (with ε_start dropped to 0.3 so pre-trained weights
aren't washed out).

- Trained 500k steps against a 1-ply lookahead opponent (win-if-can,
  block-if-must, else random).
- **Result: 0.85 vs random / 0.60 vs oneply.**
- Improvement climbed quickly from 0.16 → 0.50 in the first 180k steps,
  then crept to 0.58 over the remaining 320k — oneply had taught the net
  most of what it could.

## 5. Self-play (first attempt)

Added a `SnapshotPool` and `NetworkOpponent`. Trained against random
snapshots of past selves, warm-started from the oneply checkpoint.

- **Failed:** vs-oneply dropped to 0.10–0.19 within 300k steps; vs-random
  dipped to ~0.78; `ep_ret` ran +0.7 (agent dominating its own snapshots).
- **Root cause:** homogeneous, short-history pool + no fixed-opponent
  anchor → agent overfit to specific patterns in the pool and drifted
  into non-transferable strategies.

## 6. Self-play, take 2 (mixed opponents + anchor)

- 20% random / 20% oneply / 60% pool mix each episode.
- `snapshot_every` 5k → 10k (covers 200k steps of history).
- Warm-start checkpoint kept as a permanent pool anchor.
- Trained 1M steps.
- **Result: 0.90 vs random / 0.42 vs oneply.** vs-random improved; the
  oneply drop wasn't a real regression but the loss of a specialization.
- **Head-to-head vs oneply-trained net (at T=0.1):
  0.57–0.62 win rate** — genuinely stronger overall.

## 7. Head-to-head eval + Boltzmann temperature

Realized 1.00 vs 400 argmax games was really 2 games × 200 repeats. Added
temperature-based (softmax-of-Q) action selection to `Agent`,
`NetworkOpponent`, `eval`, and `play`; default T = 0.1 everywhere. Also
made training use temperature instead of ε-greedy.

- Continued self-play from the previous checkpoint for another 1.5M steps
  with T = 0.1 exploration.
- **Result: 0.75 vs random / 0.22 vs oneply** — a real regression.
  Exploration wasn't the bottleneck; the underlying DQN algorithm had
  hit its ceiling on this game.

## 8. AlphaZero

Ground-up rewrite of the learning algorithm (env, opponents, and play
harness reused):

- `az_network.py` — 2-headed policy/value net (4 residual blocks, 128
  channels).
- `mcts.py` — PUCT MCTS with Dirichlet root noise.
- `az_agent.py` — MCTS-driven greedy player.
- `az_train.py` — self-play with MCTS, supervised training against
  visit-count policies and final outcomes, horizontal-mirror augmentation.

Trained 100 iterations (25 games/iter, 100 sims/move) — about an hour on
an RTX 4090.

- **Result: 1.00 vs random / 0.90–0.97 vs oneply.**
- **Head-to-head vs the strongest DQN net (100 sims/move):
  1.00 win rate.**

## Takeaways

- Vanilla DQN caps out on Connect-4 well below expert play; oneply
  training + self-play squeezed it to ~0.60 vs a 1-ply heuristic.
- Self-play without anchoring to fixed baselines drifts fast — the
  20/20/60 mix was what made it stable.
- Temperature-based exploration was cleaner conceptually but did not
  break the DQN plateau.
- AlphaZero jumped past every DQN checkpoint on the first serious run.
  For games with clean tree structure and a natural symmetry, PUCT + a
  small policy/value net is dramatically more sample-efficient than
  Q-learning.

## Final layout

```
rl_connect4/
├── connect4/
│   ├── env.py               # 2-player Connect-4 environment
│   ├── network.py           # DQN Q-net
│   ├── replay.py            # replay buffer (n-step discounts)
│   ├── agent.py             # DQN agent
│   ├── opponents.py         # random & one-ply
│   ├── selfplay.py          # snapshot pool + network opponent
│   ├── config.py            # DQN hyperparameters
│   ├── az_network.py        # AlphaZero policy+value net
│   ├── mcts.py              # PUCT MCTS
│   └── az_agent.py          # MCTS-driven player
├── train.py                 # DQN training loop
├── az_train.py              # AlphaZero training loop
├── eval.py                  # DQN & AZ eval (incl. head-to-head)
└── play.py                  # interactive play vs any opponent
```
