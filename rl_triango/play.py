#!/usr/bin/env python3
"""Play TrianGO in the terminal: hot seat, optionally with random-move players.

Examples:
    python3 play.py                 # two humans
    python3 play.py -n 4 -r BGW     # you are Red; Blue, Green, White move randomly
"""

import argparse
import random

from board import CHAR_TO_PLAYER, PLAYER_NAMES, colorize, render
from game import GameState, IllegalMove, action_str, action_triangle, capture_action, is_capture

HELP = """Commands:
  <point>          place a stone on that point (1-55)
  c                capture your full triangle (when you have just one)
  c <corners...>   capture the full triangle with those corner(s), e.g. "c 4 17"
  m                list your legal moves
  u                undo
  q                quit"""


def name(p, color):
    return colorize(PLAYER_NAMES[p], p, color)


def show(state, color):
    print()
    print(render(state.board, color))
    print()
    for p in state.players:
        arrow = "->" if p == state.to_move else "  "
        print(f"{arrow} {name(p, color)}{' ' * (6 - len(PLAYER_NAMES[p]))}"
              f"hand {state.hand[p]:2}   board {state.stones_on_board(p):2}   "
              f"perimeter {state.perimeter_count(p)}/6   "
              f"captured {state.captured[p]:2}   lost {state.lost[p]:2}")
        tris = state.triangles(p)
        if tris:
            print("         triangles: " +
                  ", ".join(t.name() + ("*" if full else "") for t, full in tris))
    if any(state.triangles(p) for p in state.players):
        print("         (* = full; can be captured)")


def describe(state, player, action, color):
    who = name(player, color)
    if is_capture(action):
        return f"{who} captures triangle {action_triangle(action).name()}."
    return f"{who} places a stone at {action}."


def parse_capture(state, words):
    """Return (action, error) for a capture command."""
    full = state.full_triangles()
    if not full:
        return None, "You have no full triangles to capture."
    try:
        corners = [int(w) for w in words]
    except ValueError:
        return None, "Corners must be point numbers."
    matches = [t for t in full if all(c in t.corners for c in corners)]
    if not matches:
        return None, "None of your full triangles has those corners."
    if len(matches) > 1:
        return None, ("Which one?  Give enough corners to pick one of: " +
                      ", ".join(t.name() for t in matches))
    return capture_action(matches[0]), None


def human_turn(state, history, randoms, color):
    """Handle one command.  Returns the action to play, or None to re-prompt."""
    who = name(state.to_move, color)
    try:
        line = input(f"{who} to move> ").strip().lower()
    except EOFError:
        raise SystemExit
    if not line:
        return None
    words = line.split()
    cmd = words[0]
    if cmd in ("q", "quit", "exit"):
        raise SystemExit
    if cmd in ("h", "help", "?"):
        print(HELP)
    elif cmd == "m":
        print("Legal moves: " + ", ".join(action_str(a) for a in state.legal_actions()))
    elif cmd == "u":
        if not history:
            print("Nothing to undo.")
            return None
        # Undo back to the most recent position where a human was to move.
        prev = history.pop()
        while history and prev.to_move in randoms:
            prev = history.pop()
        state.__dict__.update(prev.__dict__)
        show(state, color)
    elif cmd in ("c", "capture"):
        action, err = parse_capture(state, words[1:])
        if err:
            print(err)
        return action
    elif cmd.isdigit():
        err = state.placement_error(int(cmd))
        if err:
            print(f"Can't place there: {err}.")
            return None
        return int(cmd)
    else:
        print(f"Unknown command: {line!r}.  Type ? for help.")
    return None


def main():
    ap = argparse.ArgumentParser(description="Play TrianGO in the terminal.")
    ap.add_argument("-n", "--players", type=int, default=2, choices=(2, 3, 4),
                    help="number of players (default 2)")
    ap.add_argument("-r", "--random", default="",
                    help="colors played by a random mover, e.g. B or BGW")
    ap.add_argument("-f", "--first", default="R", help="color that moves first (default R)")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = ap.parse_args()

    color = False if args.no_color else None
    rng = random.Random(args.seed)
    try:
        randoms = {CHAR_TO_PLAYER[c] for c in args.random.upper() if c not in ", "}
        first = CHAR_TO_PLAYER[args.first.upper()[:1]]
    except KeyError as e:
        ap.error(f"unknown color {e}; use R, B, G, or W")

    state = GameState(args.players, first)
    history = []
    print(HELP)
    show(state, color)
    while not state.is_over:
        p = state.to_move
        if p in randoms:
            action = rng.choice(state.legal_actions())
        else:
            action = human_turn(state, history, randoms, color)
            if action is None:
                continue
        history.append(state.copy())
        try:
            state.play(action)
        except IllegalMove as e:  # shouldn't happen; the UI checks first
            history.pop()
            print(f"Illegal move: {e}")
            continue
        print()
        print(describe(state, p, action, color))
        for q in state.passed:
            print(f"{name(q, color)} has no legal move and is skipped.")
        show(state, color)
    print()
    print(f"Game over after {state.num_moves} moves.  {name(state.winner, color)} wins!")


if __name__ == "__main__":
    main()
