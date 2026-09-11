"""Matches between agents, with all games played in parallel so each
agent's move choices are batched."""

from __future__ import annotations

import random
from collections import defaultdict

from game import GameState


def play_games(agents, n_games: int, opening_moves: int = 2, seed: int = 0):
    """Play n_games among len(agents) agents (one per player, 2-4).

    Games come in groups of len(agents) that share a random opening of
    `opening_moves` plies (so deterministic agents still play varied games)
    and rotate the seats, so every agent plays every seat from each opening.
    Returns the index of the winning agent for each game.
    """
    n = len(agents)
    rng = random.Random(seed)
    games = []  # (state, {player: agent index})
    for g in range(n_games):
        if g % n == 0:
            opening = GameState(n)
            for _ in range(opening_moves):
                if not opening.is_over:
                    opening.play(rng.choice(opening.legal_actions()))
        owners = {p: (j + g) % n for j, p in enumerate(opening.players)}
        games.append((opening.copy(), owners))

    active = [g for g in range(n_games) if not games[g][0].is_over]
    while active:
        by_agent = defaultdict(list)
        for g in active:
            state, owners = games[g]
            by_agent[owners[state.to_move]].append(g)
        for a, gs in by_agent.items():
            for g, action in zip(gs, agents[a].choose([games[g][0] for g in gs])):
                games[g][0].play(action)
        active = [g for g in active if not games[g][0].is_over]
    return [owners[state.winner] for state, owners in games]


def win_rate(agents, n_games: int, opening_moves: int = 2, seed: int = 0) -> float:
    """Fraction of games won by agents[0]."""
    winners = play_games(agents, n_games, opening_moves, seed)
    return sum(w == 0 for w in winners) / len(winners)
