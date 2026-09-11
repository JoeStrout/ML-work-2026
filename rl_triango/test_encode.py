"""Tests for the neural-network encoding.  Run: python3 -m unittest -v"""

import random
import unittest

import numpy as np

from board import BLUE, GRID_RC, NEIGHBOR, NUM_POINTS, NUM_SYMMETRIES, RED, Board
from encode import (
    EMPTY_TRI, FULL_TRI, HAND, HEX_KERNEL_MASK, LEGAL_PLACEMENT, NUM_PLANES, ON_BOARD,
    PERIMETER_CENTER, PERIMETER_END, SEAT, STONES, encode, encode_points, from_grid,
    legal_action_mask, relative_players, to_grid, transform_grid, transform_points,
    transform_policy, value_target,
)
from game import ACTION_PERM, NUM_ACTIONS, GameState


def random_positions(num_players, games, seed, every=4):
    rng = random.Random(seed)
    result = []
    for _ in range(games):
        s = GameState(num_players)
        while not s.is_over:
            if s.num_moves % every == 0:
                result.append(s.copy())
            s.play(rng.choice(s.legal_actions()))
    return result


def position(stones, to_move=RED, num_players=2):
    board = Board()
    for i, p in stones.items():
        board[i] = p
    return GameState.from_board(board, num_players, to_move)


class GridTests(unittest.TestCase):
    def test_grid_layout(self):
        self.assertEqual(GRID_RC[28], (4, 4))
        self.assertEqual(GRID_RC[1], (0, 5))
        self.assertEqual(len({GRID_RC[i] for i in range(1, NUM_POINTS + 1)}), NUM_POINTS)

    def test_hex_kernel_matches_neighbors(self):
        offsets = set()
        for i in range(1, NUM_POINTS + 1):
            for j in NEIGHBOR[i]:
                if j:
                    dr = GRID_RC[j][0] - GRID_RC[i][0]
                    dc = GRID_RC[j][1] - GRID_RC[i][1]
                    offsets.add((dr, dc))
        self.assertEqual(len(offsets), 6)
        kernel = {(r - 1, c - 1) for r in range(3) for c in range(3)
                  if HEX_KERNEL_MASK[r, c] and (r, c) != (1, 1)}
        self.assertEqual(offsets, kernel)

    def test_grid_round_trip(self):
        x = np.arange(2 * NUM_POINTS, dtype=np.float32).reshape(2, NUM_POINTS)
        g = to_grid(x)
        self.assertEqual(g.shape, (2, 9, 9))
        self.assertEqual(g.sum(), x.sum())
        np.testing.assert_array_equal(from_grid(g), x)


class EncodingTests(unittest.TestCase):
    def test_shape_and_constant_planes(self):
        g = encode(GameState(2))
        self.assertEqual(g.shape, (NUM_PLANES, 9, 9))
        self.assertEqual(g.dtype, np.float32)
        self.assertEqual(g[ON_BOARD].sum(), NUM_POINTS)
        self.assertEqual(g[PERIMETER_END].sum(), 12)
        self.assertEqual(g[PERIMETER_CENTER].sum(), 6)
        self.assertEqual(g[LEGAL_PLACEMENT].sum(), NUM_POINTS)
        np.testing.assert_array_equal(g[HAND:HAND + 4], g[ON_BOARD][None].repeat(4, 0) *
                                      np.array([1, 1, 0, 0])[:, None, None])
        self.assertEqual(g[SEAT + 2].sum(), 0)

    def test_seats_are_relative_to_player_to_move(self):
        s = position({28: RED}, to_move=BLUE)
        self.assertEqual(relative_players(s), (BLUE, RED))
        g = encode(s)
        self.assertEqual(g[STONES].sum(), 0)          # Blue (to move) has no stones
        self.assertEqual(g[STONES + 1, 4, 4], 1)      # Red's stone at 28
        self.assertEqual(g[STONES + 1].sum(), 1)

    def test_triangle_planes(self):
        s = position({4: RED, 17: RED, 19: RED, 11: BLUE, 30: RED, 44: RED, 46: RED})
        x = encode_points(s)
        full = {i + 1 for i in np.flatnonzero(x[FULL_TRI])}
        empty = {i + 1 for i in np.flatnonzero(x[EMPTY_TRI])}
        self.assertEqual(full, {4, 10, 11, 17, 18, 19})
        self.assertEqual(empty, {30, 37, 38, 44, 45, 46})
        self.assertEqual(x[FULL_TRI + 1].sum() + x[EMPTY_TRI + 1].sum(), 0)

    def test_legal_action_mask(self):
        s = position({4: RED, 17: RED, 19: RED, 11: BLUE})
        m = legal_action_mask(s)
        self.assertEqual(m.shape, (NUM_ACTIONS,))
        self.assertFalse(m[0])
        self.assertEqual(sorted(np.flatnonzero(m)), sorted(s.legal_actions()))

    def test_value_target(self):
        s = GameState(3, BLUE)
        np.testing.assert_array_equal(value_target(s, RED), [0, 0, 1, 0])

    def test_numpy_int_actions(self):
        s = GameState(2)
        s.play(np.int64(28))
        self.assertEqual(s.board[28], RED)


class SymmetryEncodingTests(unittest.TestCase):
    def test_encoding_commutes_with_symmetry(self):
        for n in (2, 3, 4):
            for state in random_positions(n, 6, seed=n):
                x = encode_points(state)
                m = legal_action_mask(state)
                for sym in range(NUM_SYMMETRIES):
                    t = state.transformed(sym)
                    np.testing.assert_array_equal(encode_points(t), transform_points(x, sym))
                    np.testing.assert_array_equal(legal_action_mask(t), transform_policy(m, sym))

    def test_transform_helpers_agree(self):
        state = random_positions(2, 1, seed=9)[3]
        g = encode(state)
        policy = np.random.default_rng(0).random(NUM_ACTIONS)
        for sym in range(NUM_SYMMETRIES):
            np.testing.assert_array_equal(transform_grid(g, sym), encode(state.transformed(sym)))
            moved = transform_policy(policy, sym)
            for a in range(NUM_ACTIONS):
                self.assertEqual(moved[ACTION_PERM[sym][a]], policy[a])


if __name__ == "__main__":
    unittest.main()
