"""Connect-4 environment.

Board is a (ROWS, COLS) int8 array with values in {0, 1, 2}: 0 empty,
1 for player 1, 2 for player 2. Row 0 is the bottom of the board.
Actions are column indices 0..COLS-1.
"""

from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
CONNECT = 4


class Connect4Env:
    def __init__(self) -> None:
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self.winner = 0
        self.done = False

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_player = 1
        self.winner = 0
        self.done = False
        return self.observation()

    def clone(self) -> "Connect4Env":
        c = Connect4Env()
        c.board = self.board.copy()
        c.current_player = self.current_player
        c.winner = self.winner
        c.done = self.done
        return c

    def legal_actions(self) -> np.ndarray:
        """Boolean mask of legal columns (top cell empty)."""
        return self.board[ROWS - 1] == 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Drop a piece for current_player, then flip player.

        Reward is from the perspective of the player who *just moved*:
        +1 win, 0 otherwise. Illegal moves raise ValueError.
        """
        if self.done:
            raise RuntimeError("step() called on finished game")
        if not (0 <= action < COLS) or self.board[ROWS - 1, action] != 0:
            raise ValueError(f"illegal action {action}")

        row = int(np.argmax(self.board[:, action] == 0))
        player = self.current_player
        self.board[row, action] = player

        reward = 0.0
        if self._is_win(row, action, player):
            self.winner = player
            self.done = True
            reward = 1.0
        elif not self.legal_actions().any():
            self.done = True  # draw

        self.current_player = 3 - player
        return self.observation(), reward, self.done, {"winner": self.winner}

    def observation(self) -> np.ndarray:
        """Two-channel view from current player's perspective: (own, opp)."""
        own = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == 3 - self.current_player).astype(np.float32)
        return np.stack([own, opp], axis=0)

    def _is_win(self, r: int, c: int, p: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr, cc] == p:
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= CONNECT:
                return True
        return False

    def render(self) -> str:
        glyphs = {0: ".", 1: "X", 2: "O"}
        lines = []
        for r in range(ROWS - 1, -1, -1):
            lines.append(" ".join(glyphs[int(v)] for v in self.board[r]))
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)
