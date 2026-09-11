"""Tests for the TrianGO board geometry and rules.  Run: python3 -m unittest -v"""

import itertools
import random
import unittest

from board import (
    BIT, BLUE, EMPTY, GREEN, INVERSE_SYMMETRY, NEIGHBOR, NUM_POINTS,
    NUM_SYMMETRIES, PERIMETER_CONFLICT, PERIMETER_MASK, PERIMETERS, PLAYERS,
    POINT_PERM, RED, STONES_PER_PLAYER, TRIANGLE_BY_CORNERS, TRIANGLE_PERM,
    TRIANGLES, WHITE, Board, mask_to_points, popcount,
)
from game import ACTION_PERM, NUM_ACTIONS, GameState, IllegalMove, capture_action, is_capture

R, B, G, W = RED, BLUE, GREEN, WHITE


def position(stones, num_players=2, to_move=RED, hand=None):
    """stones: {point: player}"""
    board = Board()
    for i, p in stones.items():
        board[i] = p
    return GameState.from_board(board, num_players, to_move, hand)


def line_steps(a, b):
    """If b lies on a board line from a, return (direction, steps), else None."""
    for d in range(6):
        cur, n = a, 0
        while cur:
            cur = NEIGHBOR[cur][d]
            n += 1
            if cur == b:
                return d, n
    return None


class GeometryTests(unittest.TestCase):
    def test_neighbors_match_miniscript(self):
        self.assertEqual(NEIGHBOR[28], [29, 21, 20, 27, 35, 36])
        self.assertEqual(NEIGHBOR[8][1], 0)   # NE
        self.assertEqual(NEIGHBOR[8][2], 3)   # NW
        self.assertEqual(NEIGHBOR[8][5], 15)  # SE

    def test_specific_triangles(self):
        self.assertEqual(TRIANGLE_BY_CORNERS[(4, 17, 19)].points, (4, 10, 11, 17, 18, 19))
        self.assertEqual(TRIANGLE_BY_CORNERS[(1, 3, 13)].points, (1, 2, 3, 6, 7, 13))
        self.assertNotIn((5, 11, 12), TRIANGLE_BY_CORNERS)  # too small (3 points)

    def test_triangle_list_matches_brute_force(self):
        # Independently find every triple of points that are pairwise joined by
        # board lines of equal length >= 2 in three different directions.
        expected = set()
        for tri in itertools.combinations(range(1, NUM_POINTS + 1), 3):
            links = [line_steps(a, b) for a, b in itertools.combinations(tri, 2)]
            if None in links:
                continue
            steps = {n for _, n in links}
            dirs = {d % 3 for d, _ in links}  # a line and its reverse count once
            if len(steps) == 1 and steps.pop() >= 2 and len(dirs) == 3:
                expected.add(tri)
        self.assertEqual(set(TRIANGLE_BY_CORNERS), expected)
        for t in TRIANGLES:
            self.assertEqual(len(t.points), (t.size + 1) * (t.size + 2) // 2)


class PlacementTests(unittest.TestCase):
    def test_occupied(self):
        s = position({28: B})
        self.assertIsNotNone(s.placement_error(28))
        self.assertRaises(IllegalMove, s.play, 28)

    def test_perimeter_center_blocks_ends(self):
        s = position({2: R})
        self.assertIsNotNone(s.placement_error(1, R))
        self.assertIsNotNone(s.placement_error(3, R))
        self.assertIsNone(s.placement_error(1, B))

    def test_perimeter_end_blocks_center(self):
        s = position({40: R})
        self.assertIsNotNone(s.placement_error(32, R))  # 40 is the center of 32-40-47
        s = position({32: R})
        self.assertIsNotNone(s.placement_error(40, R))
        self.assertIsNone(s.placement_error(47, R))       # both ends are fine
        self.assertIsNone(s.placement_error(40, B))

    def test_six_perimeter_stones_max(self):
        s = position({1: R, 3: R, 4: R, 17: R, 9: R, 24: R})
        self.assertEqual(s.perimeter_count(R), 6)
        self.assertIsNotNone(s.placement_error(32, R))
        self.assertIsNone(s.placement_error(28, R))       # interior still fine
        self.assertIsNone(s.placement_error(32, B))

    def test_no_stones_in_hand(self):
        s = position({28: R}, hand={R: 0})
        self.assertIsNotNone(s.placement_error(29, R))
        self.assertEqual(s.legal_placements(R), [])


class TriangleTests(unittest.TestCase):
    def test_empty_triangle_not_capturable(self):
        s = position({4: R, 17: R, 19: R, 11: R})
        self.assertEqual(s.triangles(R), [(TRIANGLE_BY_CORNERS[(4, 17, 19)], False)])
        self.assertEqual(s.full_triangles(R), [])
        tri = TRIANGLE_BY_CORNERS[(4, 17, 19)]
        self.assertRaises(IllegalMove, s.play, capture_action(tri))

    def test_border_stone_makes_full(self):
        s = position({4: R, 17: R, 19: R, 10: B})
        self.assertEqual(s.full_triangles(R), [TRIANGLE_BY_CORNERS[(4, 17, 19)]])

    def test_capture(self):
        # Red triangle 4-17-19 holding Blue 11 and Red 18; Blue 28 is outside.
        s = position({4: R, 17: R, 19: R, 18: R, 11: B, 28: B})
        s.play(capture_action(TRIANGLE_BY_CORNERS[(4, 17, 19)]))
        for i in (4, 10, 11, 17, 18, 19):
            self.assertEqual(s.board[i], EMPTY)
        self.assertEqual(s.board[28], B)
        self.assertEqual(s.hand[R], STONES_PER_PLAYER)      # all 4 red stones back
        self.assertEqual(s.captured[R], 1)
        self.assertEqual(s.lost[B], 1)
        self.assertEqual(s.hand[B], STONES_PER_PLAYER - 2)  # captured stone is gone
        self.assertEqual(s.to_move, B)

    def test_capture_several_opponents(self):
        s = position({4: R, 17: R, 19: R, 11: B, 18: G, 10: W}, num_players=4)
        s.play(capture_action(TRIANGLE_BY_CORNERS[(4, 17, 19)]))
        self.assertEqual(s.captured[R], 3)
        self.assertEqual([s.lost[p] for p in (B, G, W)], [1, 1, 1])

    def test_capture_breaks_other_triangles(self):
        # Blue's triangle 18-20-34 loses its corner 18 when Red captures 4-17-19.
        s = position({4: R, 17: R, 19: R, 18: B, 20: B, 34: B}, to_move=R)
        self.assertEqual(len(s.triangles(B)), 1)
        s.play(capture_action(TRIANGLE_BY_CORNERS[(4, 17, 19)]))
        self.assertEqual(s.triangles(B), [])


class TurnAndEndTests(unittest.TestCase):
    def test_turns_alternate(self):
        s = GameState(3)
        for expected in (R, B, G, R):
            self.assertEqual(s.to_move, expected)
            s.play(s.legal_placements()[0])

    def test_two_player_out_of_stones_loses(self):
        s = position({28: B}, to_move=R, hand={B: 0})
        s.play(1)
        self.assertTrue(s.is_over)
        self.assertEqual(s.winner, R)
        self.assertIsNone(s.to_move)

    def test_forced_capture_keeps_player_alive(self):
        # Blue has no stones in hand but has a full triangle, so must capture.
        s = position({4: B, 17: B, 19: B, 11: R}, to_move=R, hand={B: 0})
        s.play(28)
        self.assertFalse(s.is_over)
        self.assertEqual(s.to_move, B)
        self.assertEqual(s.legal_actions(), [capture_action(TRIANGLE_BY_CORNERS[(4, 17, 19)])])

    def test_stuck_player_is_skipped(self):
        # Green has nothing to place and no full triangle; Red and Blue can move.
        s = position({4: G, 17: G, 19: G}, num_players=3, to_move=B, hand={G: 0})
        s.play(28)
        self.assertEqual(s.to_move, R)
        self.assertEqual(s.passed, [G])
        # Red puts a stone in Green's triangle, so Green gets a turn to capture.
        s.play(11)
        self.assertEqual(s.to_move, B)
        s.play(29)
        self.assertEqual(s.to_move, G)
        self.assertTrue(all(is_capture(a) for a in s.legal_actions()))

    def test_last_player_able_to_move_wins(self):
        # Three players; after Red's move only Red can move.
        s = position({28: B, 29: G}, num_players=3, to_move=R, hand={B: 0, G: 0})
        s.play(1)
        self.assertEqual(s.winner, R)

    def test_nobody_can_move_mover_wins(self):
        s = position({28: B}, to_move=R, hand={R: 1, B: 0})
        s.play(1)
        self.assertEqual(s.winner, R)


def state_key(s):
    return (s.board.to_string(), s.hand, s.captured, s.lost, s.to_move,
            s.winner, s.passed, s.num_moves)


class SymmetryTests(unittest.TestCase):
    def test_twelve_symmetries_form_a_group(self):
        perms = {tuple(p) for p in POINT_PERM}
        self.assertEqual(len(perms), NUM_SYMMETRIES)
        self.assertEqual(POINT_PERM[0], list(range(NUM_POINTS + 1)))
        for s, perm in enumerate(POINT_PERM):
            self.assertEqual(sorted(perm), list(range(NUM_POINTS + 1)))
            self.assertEqual(perm[28], 28)
            inv = POINT_PERM[INVERSE_SYMMETRY[s]]
            self.assertTrue(all(inv[perm[i]] == i for i in range(NUM_POINTS + 1)))
            for other in POINT_PERM:
                self.assertIn(tuple(other[perm[i]] for i in range(NUM_POINTS + 1)), perms)
        # s=1 is a rotation of order 6.
        p = list(range(NUM_POINTS + 1))
        for k in range(1, 7):
            p = [POINT_PERM[1][i] for i in p]
            self.assertEqual(p == list(range(NUM_POINTS + 1)), k == 6)

    def test_symmetries_preserve_board_structure(self):
        edges = {frozenset((i, j)) for i in range(1, NUM_POINTS + 1) for j in NEIGHBOR[i] if j}
        double = set()
        for a, b, c in PERIMETERS:
            double |= {frozenset((a, b)), frozenset((b, c))}
        perims = {(frozenset((a, c)), b) for a, b, c in PERIMETERS}
        for s, perm in enumerate(POINT_PERM):
            image = lambda pts: frozenset(perm[i] for i in pts)
            self.assertEqual({image(e) for e in edges}, edges)
            self.assertEqual({image(e) for e in double}, double)
            self.assertEqual({(image(ends), perm[c]) for ends, c in perims}, perims)
            for t in TRIANGLES:
                u = TRIANGLES[TRIANGLE_PERM[s][t.index]]
                self.assertEqual(set(u.points), image(t.points))
                self.assertEqual(set(u.corners), image(t.corners))

    def test_action_perm(self):
        for s in range(NUM_SYMMETRIES):
            self.assertEqual(sorted(ACTION_PERM[s]), list(range(NUM_ACTIONS)))
            self.assertEqual(ACTION_PERM[s][0], 0)

    def test_play_commutes_with_symmetry(self):
        rng = random.Random(99)
        for num_players in (2, 3, 4):
            for _ in range(5):
                s = GameState(num_players)
                while not s.is_over:
                    legal = s.legal_actions()
                    a = rng.choice(legal)
                    for sym in range(NUM_SYMMETRIES):
                        t = s.transformed(sym)
                        self.assertEqual(sorted(t.legal_actions()),
                                         sorted(ACTION_PERM[sym][x] for x in legal))
                        expected = s.copy()
                        expected.play(a)
                        t.play(ACTION_PERM[sym][a])
                        self.assertEqual(state_key(t), state_key(expected.transformed(sym)))
                    s.play(a)


class RandomPlayoutTests(unittest.TestCase):
    def check_invariants(self, s):
        occupied = 0
        for p in s.players:
            m = s.masks[p]
            self.assertEqual(occupied & m, 0)
            occupied |= m
            self.assertEqual(s.hand[p] + popcount(m) + s.lost[p], STONES_PER_PLAYER)
            self.assertGreaterEqual(s.hand[p], 0)
            self.assertLessEqual(popcount(m & PERIMETER_MASK), 6)
            for i in range(1, NUM_POINTS + 1):
                if m & BIT[i]:
                    self.assertEqual(s.board[i], p)
                    self.assertFalse(m & PERIMETER_CONFLICT[i])
        for i in range(1, NUM_POINTS + 1):
            if not occupied & BIT[i]:
                self.assertEqual(s.board[i], EMPTY)
        self.assertEqual(sum(s.captured), sum(s.lost))
        # Incrementally maintained data must match a from-scratch computation.
        self.assertEqual(s.occ, occupied)
        for p in s.players:
            m = s.masks[p]
            self.assertEqual(list(s.tris[p]),
                             [t for t in TRIANGLES if m & t.corner_mask == t.corner_mask])
            self.assertEqual(mask_to_points(m),
                             [i for i in range(1, NUM_POINTS + 1) if m & BIT[i]])
        if not s.is_over:
            self.assertTrue(s.legal_actions())

    def test_random_games(self):
        rng = random.Random(1234)
        for num_players in (2, 3, 4):
            for _ in range(100):
                s = GameState(num_players, rng.choice(PLAYERS[:num_players]))
                while not s.is_over:
                    s.play(rng.choice(s.legal_actions()))
                    self.check_invariants(s)
                    self.assertLess(s.num_moves, 5000)
                self.assertIn(s.winner, s.players)


if __name__ == "__main__":
    unittest.main()
