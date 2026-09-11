"""Batched self-play: many games at once, searched in lockstep."""

from __future__ import annotations

import numpy as np

from encode import encode_points, relative_players
from game import NUM_ACTIONS, GameState, is_capture

from .agents import MCTSAgent


class _Game:
    def __init__(self, num_players: int) -> None:
        self.state = GameState(num_players)
        self.x, self.pi, self.seats = [], [], []
        self.captures = 0


def self_play(agent: MCTSAgent, n_games: int, parallel: int, num_players: int,
              temp_moves: int, rng: np.random.Generator):
    """Play n_games of self-play, `parallel` at a time.

    Moves are sampled from the visit counts for the first temp_moves plies,
    then chosen greedily.  Returns (x, pi, z, stats):
      x   (n, NUM_PLANES, 55) float16 point-form encodings
      pi  (n, NUM_ACTIONS) float32 visit-count policies
      z   (n,) int64 winner's seat relative to the player to move
    """
    started = 0
    games = []
    while started < min(parallel, n_games):
        games.append(_Game(num_players))
        started += 1

    xs, pis, zs = [], [], []
    lengths, captures, first_wins = [], [], 0
    while games:
        results = agent.search([g.state for g in games], add_noise=True)
        still = []
        for g, (actions, visits) in zip(games, results):
            s = g.state
            p = visits / visits.sum()
            pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
            pi[actions] = p
            g.x.append(encode_points(s).astype(np.float16))
            g.pi.append(pi)
            g.seats.append(relative_players(s))
            if s.num_moves < temp_moves:
                a = actions[rng.choice(len(actions), p=p)]
            else:
                a = actions[int(np.argmax(visits))]
            g.captures += is_capture(a)
            s.play(a)
            if not s.is_over:
                still.append(g)
                continue
            xs.extend(g.x)
            pis.extend(g.pi)
            zs.extend(seats.index(s.winner) for seats in g.seats)
            lengths.append(s.num_moves)
            captures.append(g.captures)
            first_wins += s.winner == s.players[0]
            if started < n_games:
                still.append(_Game(num_players))
                started += 1
        games = still

    stats = {
        "positions": len(zs),
        "mean_length": float(np.mean(lengths)),
        "mean_captures": float(np.mean(captures)),
        "first_player_wins": first_wins / n_games,
    }
    return np.stack(xs), np.stack(pis), np.array(zs, dtype=np.int64), stats
