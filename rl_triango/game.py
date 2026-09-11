"""TrianGO game state and rules.

Rules (from the manual), as implemented here:

  * Each player starts with 14 stones in hand.  Turns go in order
    Red, Blue, Green, White (clockwise), using only as many as are playing.
  * A turn is either placing a stone from your hand on an empty point, or
    capturing one of your full triangles.
  * Perimeter rules, per player: on each perimeter (double line) you may
    occupy one end, both ends, or only the center -- never the center plus
    an end.  And you may have at most 6 stones on perimeter points in total.
  * A triangle is three of your stones at the corners of an equilateral
    triangle whose sides run along board lines, with side length >= 2
    (so it covers at least 6 points).  It is "full" if any opponent stone
    is inside it or on its border, and only full triangles can be captured.
  * Capturing removes every stone inside and on the border of the triangle
    (corners included).  Your own stones go back to your hand; opponent
    stones are out of the game for good.
  * A player with no legal move on their turn is skipped.  The game ends as
    soon as at most one player is able to move; that player wins (or, if
    nobody can move, the player who just moved wins).  In a two-player game
    this means you lose when you cannot move on your turn.

Actions are ints: 1-55 place a stone on that point; CAPTURE_BASE + k
captures TRIANGLES[k].  (0 is unused.)  ACTION_PERM maps actions through
the board symmetries in board.py.
"""

import operator
from operator import attrgetter

from board import (
    ALL_POINTS_MASK, BIT, EMPTY, MAX_PERIMETER_STONES, NUM_POINTS,
    PERIMETER_BITS, PERIMETER_CONFLICT, PERIMETER_MASK, PLAYER_NAMES, PLAYERS,
    POINT_PERM, STONES_PER_PLAYER, TRIANGLE_PERM, TRIANGLES,
    TRIANGLES_AT_CORNER, Board, mask_to_points, popcount, transform_mask,
)

CAPTURE_BASE = NUM_POINTS + 1
NUM_ACTIONS = CAPTURE_BASE + len(TRIANGLES)

# ACTION_PERM[s][a]: what action a becomes under board symmetry s.
ACTION_PERM = [[0] + perm[1:] + [CAPTURE_BASE + k for k in tperm]
               for perm, tperm in zip(POINT_PERM, TRIANGLE_PERM)]


def is_capture(action):
    return action >= CAPTURE_BASE


def capture_action(tri):
    return CAPTURE_BASE + tri.index


def action_triangle(action):
    return TRIANGLES[action - CAPTURE_BASE]


def action_str(action):
    if is_capture(action):
        return "C" + action_triangle(action).name()
    return str(action)


class IllegalMove(Exception):
    pass


_by_index = attrgetter("index")


class GameState:
    def __init__(self, num_players=2, first_player=PLAYERS[0]):
        if not 2 <= num_players <= 4:
            raise ValueError("TrianGO is for 2 to 4 players")
        self.players = PLAYERS[:num_players]
        if first_player not in self.players:
            raise ValueError(f"{first_player} is not playing")
        self.board = Board()
        self.occ = 0               # bitmask of all stones on the board
        # The following are indexed by player (index 0 unused).
        self.masks = [0] * 5       # bitmask of each player's stones on the board
        self.tris = [()] * 5       # each player's triangles (empty or full), by index
        self.hand = [0] * 5        # stones available to place
        self.captured = [0] * 5    # opponent stones this player has captured
        self.lost = [0] * 5        # this player's stones captured by opponents
        for p in self.players:
            self.hand[p] = STONES_PER_PLAYER
        self.to_move = first_player  # None once the game is over
        self.winner = None
        self.last_mover = None
        self.passed = []           # players skipped (unable to move) after the last move
        self.num_moves = 0

    @classmethod
    def from_board(cls, board, num_players=2, to_move=PLAYERS[0], hand=None):
        """Set up a position (for testing).  By default each player's hand
        is 14 minus their stones on the board; pass hand={player: n} to
        override."""
        s = cls(num_players, to_move)
        for i in range(1, NUM_POINTS + 1):
            p = board[i]
            if p == EMPTY:
                continue
            if p not in s.players:
                raise ValueError(f"{PLAYER_NAMES[p]} is not playing")
            s.board.points[i] = p
            s.masks[p] |= BIT[i]
            s.occ |= BIT[i]
            s.hand[p] -= 1
        for p in s.players:
            own = s.masks[p]
            s.tris[p] = tuple(t for t in TRIANGLES if own & t.corner_mask == t.corner_mask)
        if hand:
            for p, n in hand.items():
                s.hand[p] = n
        return s

    @classmethod
    def from_string(cls, board_str, num_players=2, to_move=PLAYERS[0], hand=None):
        return cls.from_board(Board.from_string(board_str), num_players, to_move, hand)

    def copy(self):
        s = GameState.__new__(GameState)
        s.players = self.players
        s.board = self.board.copy()
        s.occ = self.occ
        s.masks = self.masks[:]
        s.tris = self.tris[:]
        s.hand = self.hand[:]
        s.captured = self.captured[:]
        s.lost = self.lost[:]
        s.to_move = self.to_move
        s.winner = self.winner
        s.last_mover = self.last_mover
        s.passed = self.passed[:]
        s.num_moves = self.num_moves
        return s

    def transformed(self, s):
        """A copy of this state with board symmetry s applied."""
        t = self.copy()
        perm = POINT_PERM[s]
        t.board = Board()
        for i in range(1, NUM_POINTS + 1):
            t.board.points[perm[i]] = self.board.points[i]
        t.occ = transform_mask(self.occ, s)
        tperm = TRIANGLE_PERM[s]
        for p in self.players:
            t.masks[p] = transform_mask(self.masks[p], s)
            t.tris[p] = tuple(sorted((TRIANGLES[tperm[x.index]] for x in self.tris[p]),
                                     key=_by_index))
        return t

    # ----- queries -----

    @property
    def is_over(self):
        return self.winner is not None

    def occupied_mask(self):
        return self.occ

    def stones_on_board(self, player):
        return popcount(self.masks[player])

    def perimeter_count(self, player):
        return popcount(self.masks[player] & PERIMETER_MASK)

    def placement_error(self, point, player=None):
        """Return why player may not place at point, or None if they may."""
        p = self.to_move if player is None else player
        try:
            point = operator.index(point)
        except TypeError:
            return f"{point!r} is not a point on the board"
        if not 1 <= point <= NUM_POINTS:
            return f"{point} is not a point on the board"
        if self.board[point] != EMPTY:
            return f"point {point} is occupied"
        if self.hand[p] == 0:
            return f"{PLAYER_NAMES[p]} has no stones left to place"
        if PERIMETER_MASK & BIT[point]:
            if self.masks[p] & PERIMETER_CONFLICT[point]:
                return "a perimeter may not have your stones on both its center and an end"
            if self.perimeter_count(p) >= MAX_PERIMETER_STONES:
                return f"you already have {MAX_PERIMETER_STONES} stones on perimeters"
        return None

    def blocked_mask(self, player):
        """Empty or not, the points where the perimeter rules forbid player to place."""
        own = self.masks[player]
        if popcount(own & PERIMETER_MASK) >= MAX_PERIMETER_STONES:
            return PERIMETER_MASK
        blocked = 0
        for center, ends in PERIMETER_BITS:
            if own & center:
                blocked |= ends
            if own & ends:
                blocked |= center
        return blocked

    def legal_placement_mask(self, player=None):
        p = self.to_move if player is None else player
        if self.hand[p] == 0:
            return 0
        return ALL_POINTS_MASK & ~self.occ & ~self.blocked_mask(p)

    def legal_placements(self, player=None):
        return mask_to_points(self.legal_placement_mask(player))

    def triangles(self, player):
        """All of player's triangles, as a list of (Triangle, is_full)."""
        opp = self.occ & ~self.masks[player]
        return [(t, bool(t.mask & opp)) for t in self.tris[player]]

    def full_triangles(self, player=None):
        p = self.to_move if player is None else player
        opp = self.occ & ~self.masks[p]
        return [t for t in self.tris[p] if t.mask & opp]

    def legal_actions(self, player=None):
        if self.is_over:
            return []
        p = self.to_move if player is None else player
        return self.legal_placements(p) + [capture_action(t) for t in self.full_triangles(p)]

    def has_legal_move(self, player):
        if self.legal_placement_mask(player):
            return True
        opp = self.occ & ~self.masks[player]
        for t in self.tris[player]:
            if t.mask & opp:
                return True
        return False

    # ----- making moves -----

    def play(self, action):
        """Apply action for the player to move (mutates this state)."""
        if self.is_over:
            raise IllegalMove("the game is over")
        try:
            action = operator.index(action)  # accepts numpy ints too
        except TypeError:
            raise IllegalMove(f"not an action: {action!r}") from None
        p = self.to_move
        if is_capture(action):
            if action >= NUM_ACTIONS:
                raise IllegalMove(f"no such action: {action}")
            tri = action_triangle(action)
            if tri not in self.full_triangles(p):
                raise IllegalMove(f"{tri.name()} is not a full triangle of {PLAYER_NAMES[p]}")
            self._capture(p, tri)
        else:
            err = self.placement_error(action, p)
            if err:
                raise IllegalMove(err)
            self._place(p, action)
        self.last_mover = p
        self.num_moves += 1
        self._advance(p)

    def _place(self, player, point):
        b = BIT[point]
        self.board.points[point] = player
        own = self.masks[player] | b
        self.masks[player] = own
        self.occ |= b
        self.hand[player] -= 1
        new = [t for t in TRIANGLES_AT_CORNER[point] if own & t.corner_mask == t.corner_mask]
        if new:
            self.tris[player] = tuple(sorted(self.tris[player] + tuple(new), key=_by_index))

    def _capture(self, player, tri):
        for q in self.players:
            removed = self.masks[q] & tri.mask
            if not removed:
                continue
            n = popcount(removed)
            self.masks[q] &= ~removed
            self.tris[q] = tuple(t for t in self.tris[q] if not t.corner_mask & tri.mask)
            if q == player:
                self.hand[q] += n
            else:
                self.captured[player] += n
                self.lost[q] += n
        self.occ &= ~tri.mask
        for i in tri.points:
            self.board.points[i] = EMPTY

    def _advance(self, mover):
        """Find the next player able to move, or end the game."""
        n = len(self.players)
        i = self.players.index(mover)
        first = None
        skipped = []
        for k in range(1, n + 1):
            q = self.players[(i + k) % n]
            if self.has_legal_move(q):
                if first is not None:  # at least two players can move
                    self.to_move = first
                    self.passed = skipped
                    return
                first = q
            elif first is None:
                skipped.append(q)
        self.winner = mover if first is None else first
        self.to_move = None
        self.passed = []
