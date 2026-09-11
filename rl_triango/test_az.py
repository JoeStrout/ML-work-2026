"""Tests for the AlphaZero code.  Run: python -m unittest -v test_az"""

import os
import random
import tempfile
import unittest

import numpy as np
import torch

from az.agents import MCTSAgent, RandomAgent, net_agent, rollout_agent
from az.arena import play_games
from az.evaluators import NetEvaluator, RolloutEvaluator
from az.mcts import MCTSConfig, Tree, run_search
from az.network import NEG, TrianGONet
from az.replay import ReplayBuffer
from az.selfplay import self_play
from board import BLUE, PERIMETER_POINTS, RED, TRIANGLES, Board
from encode import LEGAL_PLACEMENT, encode, encode_points, from_grid
from game import NUM_ACTIONS, GameState, is_capture

CPU = torch.device("cpu")


def tiny_net():
    torch.manual_seed(0)
    return TrianGONet(channels=16, blocks=1, head_channels=8, value_channels=2).eval()


def winning_capture_position():
    """Red to move.  Blue has no stones in hand and one full triangle
    (4-17-19, holding Red 11), so Blue survives only by capturing it.  Red
    has a full triangle containing Blue's corner 19: capturing it wins at
    once; any placement lets Blue capture and play on."""
    blue = {4, 17, 19}
    for t in TRIANGLES:
        corners = set(t.corners)
        if (19 in t.points and not corners & (blue | {11} | PERIMETER_POINTS)):
            board = Board()
            for p in blue:
                board[p] = BLUE
            for p in corners | {11}:
                board[p] = RED
            return GameState.from_board(board, 2, RED, hand={BLUE: 0})
    raise AssertionError("no suitable triangle")


def random_positions(num_players, n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = GameState(num_players)
        while not s.is_over:
            if rng.random() < 0.2:
                out.append(s.copy())
            s.play(rng.choice(s.legal_actions()))
    return out[:n]


class NetworkTests(unittest.TestCase):
    def test_shapes_and_masks(self):
        net = tiny_net()
        states = [GameState(2), GameState(3), GameState(4)]
        x = torch.from_numpy(np.stack([encode(s) for s in states]))
        logits, v = net(x)
        self.assertEqual(logits.shape, (3, NUM_ACTIONS))
        self.assertEqual(v.shape, (3, 4))
        self.assertTrue((logits[:, 0] == NEG).all())
        probs = torch.softmax(v, 1)
        self.assertTrue(torch.allclose(probs.sum(1), torch.ones(3)))
        self.assertLess(probs[0, 2:].sum().item(), 1e-6)   # 2 players: seats 2,3 absent
        self.assertLess(probs[1, 3].item(), 1e-6)

    def test_net_evaluator_value_is_per_player(self):
        s = GameState(3, BLUE)
        [(actions, priors, v)] = NetEvaluator(tiny_net(), CPU)([s])
        self.assertEqual(actions, s.legal_actions())
        self.assertAlmostEqual(priors.sum(), 1.0, places=5)
        self.assertAlmostEqual(v.sum(), 1.0, places=5)
        self.assertEqual(v[3], 0.0)  # White not playing


class MCTSTests(unittest.TestCase):
    def test_root_visit_count(self):
        s = GameState(2)
        tree = Tree(s, MCTSConfig(sims=50))
        run_search([tree], RolloutEvaluator(0), 50)
        actions, n = tree.root_visits()
        self.assertEqual(n.sum(), 50)
        self.assertEqual(actions, s.legal_actions())
        self.assertEqual(s.num_moves, 0)  # state untouched

    def test_finds_winning_capture(self):
        # Red is far ahead here, so random rollouts call every move a win;
        # use evaluators that know nothing, so only the terminal win stands out.
        def know_nothing(states):
            out = []
            for s in states:
                actions = s.legal_actions()
                v = np.zeros(4)
                v[[p - 1 for p in s.players]] = 1 / len(s.players)
                out.append((actions, np.full(len(actions), 1 / len(actions)), v))
            return out

        # With flat priors and an FPU reduction, search sticks to the first
        # child it tries; fpu_reduction=0 makes it try every child once.
        s = winning_capture_position()
        agents = [MCTSAgent(know_nothing, MCTSConfig(sims=100, fpu_reduction=0.0)),
                  MCTSAgent(NetEvaluator(tiny_net(), CPU), MCTSConfig(sims=100))]
        for agent in agents:
            [a] = agent.choose([s])
            self.assertTrue(is_capture(a))
            t = s.copy()
            t.play(a)
            self.assertEqual(t.winner, RED)

    def test_forced_move_not_searched(self):
        # Red has no stones in hand and one full triangle: capturing is forced.
        board = Board()
        for p in (4, 17, 19):
            board[p] = RED
        board[11] = BLUE
        s = GameState.from_board(board, 2, RED, hand={RED: 0})
        [(actions, n)] = rollout_agent(50).search([s])
        self.assertEqual(len(actions), 1)
        self.assertEqual(n.tolist(), [1])

    def test_many_trees_in_lockstep(self):
        states = random_positions(3, 8, seed=3)
        agent = MCTSAgent(NetEvaluator(tiny_net(), CPU), MCTSConfig(sims=20))
        for s, (actions, n) in zip(states, agent.search(states, add_noise=True)):
            self.assertEqual(sorted(actions), sorted(s.legal_actions()))
            if len(actions) > 1:
                self.assertEqual(n.sum(), 20)


class ArenaTests(unittest.TestCase):
    def test_seats_rotate(self):
        for n in (2, 3, 4):
            winners = play_games([RandomAgent(k) for k in range(n)], 4 * n, seed=n)
            self.assertEqual(len(winners), 4 * n)
            self.assertTrue(all(0 <= w < n for w in winners))

    def test_rollout_beats_random(self):
        winners = play_games([rollout_agent(100, seed=0), RandomAgent(0)], 10)
        self.assertGreaterEqual(sum(w == 0 for w in winners), 8)


class SelfPlayAndReplayTests(unittest.TestCase):
    def test_self_play(self):
        agent = net_agent(tiny_net(), CPU, sims=4)
        for n in (2, 3):
            x, pi, z, stats = self_play(agent, 3, 2, n, 4, np.random.default_rng(0))
            self.assertEqual(len(x), stats["positions"])
            self.assertEqual(x.shape[1:], (28, 55))
            np.testing.assert_allclose(pi.sum(1), 1, rtol=1e-5)
            self.assertTrue(((z >= 0) & (z < n)).all())

    def test_save_load_round_trip(self):
        n = 7
        x = np.arange(n, dtype=np.float16)[:, None, None] * np.ones((1, 28, 55), np.float16)
        pi = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        pi[np.arange(n), np.arange(n) + 1] = 1
        buf = ReplayBuffer(5)
        buf.add(x, pi, np.arange(n))    # wraps: holds samples 2..6
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "replay.npz")
            buf.save(path, iteration=3)
            same, smaller = ReplayBuffer(5), ReplayBuffer(3)
            self.assertEqual(same.load(path), {"iteration": 3})
            smaller.load(path)
        self.assertEqual(same.z.tolist(), [2, 3, 4, 5, 6])   # oldest first
        self.assertEqual(same.x[:, 0, 0].tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(same.pi.argmax(1).tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(smaller.z.tolist(), [4, 5, 6])      # newest kept
        same.add(x[:1], pi[:1], np.array([9]))               # overwrites the oldest
        self.assertEqual(same.z.tolist(), [9, 3, 4, 5, 6])

    def test_samples_are_consistently_symmetrized(self):
        # Store random legal policies; after sampling, placement mass must
        # still sit on the (transformed) legal-placement plane.
        rng = np.random.default_rng(0)
        buf = ReplayBuffer(100)
        states = random_positions(2, 60, seed=5)
        x = np.stack([encode_points(s).astype(np.float16) for s in states])
        pi = np.zeros((len(states), NUM_ACTIONS), dtype=np.float32)
        for k, s in enumerate(states):
            acts = s.legal_actions()
            pi[k, acts] = rng.random(len(acts))
            pi[k] /= pi[k].sum()
        buf.add(x, pi, np.zeros(len(states), dtype=np.int64))
        gx, gpi, _ = buf.sample(500, rng)
        legal = from_grid(gx)[:, LEGAL_PLACEMENT]
        self.assertTrue((gpi[:, 1:56][legal == 0] == 0).all())
        np.testing.assert_allclose(gpi.sum(1), 1, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
