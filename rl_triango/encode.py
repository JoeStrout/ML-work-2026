"""Neural-network input encoding for TrianGO positions.

This is shared infrastructure: training and inference code everywhere must
agree on it.  If you change anything here, bump ENCODING_VERSION.

A position is encoded as NUM_PLANES feature planes over the 55 points.  In
"point form" the shape is (NUM_PLANES, 55), with point i in column i-1.
to_grid() scatters that onto the 9x9 grid of board.GRID_RC, shape
(NUM_PLANES, 9, 9), for a convolutional network; off-board cells are 0 in
every plane.  In grid form a point's six neighbors are its 3x3
neighborhood minus the (-1,-1) and (+1,+1) corners (see HEX_KERNEL_MASK).

Players are encoded relative to the player to move: seat 0 is the player to
move, seat 1 the next player in turn order, and so on.  Seats not in the
game are all zeros.  (Only the current position matters in TrianGO, so
there are no history planes.)

Planes (k = seat 0..3):
   0- 3  stones[k]           1 where seat k has a stone
   4     on_board            1 on every point
   5     perimeter_end       1 on the 12 perimeter end points
   6     perimeter_center    1 on the 6 perimeter center points
   7     legal_placement     1 where the player to move may place a stone
   8-11  empty_tri[k]        1 on points inside or on seat k's empty triangles
  12-15  full_tri[k]         1 on points inside or on seat k's full triangles
  16-19  hand[k]             seat k's stones in hand / 14, on every point
  20-23  perimeter_count[k]  seat k's stones on perimeters / 6, on every point
  24-27  seat[k]             1 on every point if seat k is in the game

Policies use game.py's action numbering (NUM_ACTIONS = 192; action 0 is
never legal).  Values are over the 4 seats (one-hot for the winner).
"""

import numpy as np

from board import (
    GRID_COLS, GRID_RC, GRID_ROWS, INVERSE_SYMMETRY, MAX_PERIMETER_STONES,
    NUM_POINTS, NUM_SYMMETRIES, PERIMETERS, POINT_PERM, STONES_PER_PLAYER,
    popcount,
)
from game import ACTION_PERM, NUM_ACTIONS
from board import PERIMETER_MASK

ENCODING_VERSION = 1
MAX_PLAYERS = 4

PLANE_NAMES = (
    [f"stones[{k}]" for k in range(4)] +
    ["on_board", "perimeter_end", "perimeter_center", "legal_placement"] +
    [f"empty_tri[{k}]" for k in range(4)] +
    [f"full_tri[{k}]" for k in range(4)] +
    [f"hand[{k}]" for k in range(4)] +
    [f"perimeter_count[{k}]" for k in range(4)] +
    [f"seat[{k}]" for k in range(4)]
)
NUM_PLANES = len(PLANE_NAMES)

STONES = PLANE_NAMES.index("stones[0]")
ON_BOARD = PLANE_NAMES.index("on_board")
PERIMETER_END = PLANE_NAMES.index("perimeter_end")
PERIMETER_CENTER = PLANE_NAMES.index("perimeter_center")
LEGAL_PLACEMENT = PLANE_NAMES.index("legal_placement")
EMPTY_TRI = PLANE_NAMES.index("empty_tri[0]")
FULL_TRI = PLANE_NAMES.index("full_tri[0]")
HAND = PLANE_NAMES.index("hand[0]")
PERIMETER_COUNT = PLANE_NAMES.index("perimeter_count[0]")
SEAT = PLANE_NAMES.index("seat[0]")

# 3x3 convolution mask selecting a grid cell and its six hex neighbors.
HEX_KERNEL_MASK = np.array([[0, 1, 1],
                            [1, 1, 1],
                            [1, 1, 0]], dtype=np.float32)

GRID_ROW_INDEX = np.array([GRID_RC[i][0] for i in range(1, NUM_POINTS + 1)])
GRID_COL_INDEX = np.array([GRID_RC[i][1] for i in range(1, NUM_POINTS + 1)])


def mask_bits(mask):
    """A bitmask of points as a uint8 array of shape (55,)."""
    raw = np.frombuffer(mask.to_bytes((NUM_POINTS + 8) // 8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[1:NUM_POINTS + 1]


_PERIMETER_END_BITS = mask_bits(sum(1 << p for a, _, c in PERIMETERS for p in (a, c)))
_PERIMETER_CENTER_BITS = mask_bits(sum(1 << b for _, b, _ in PERIMETERS))


def relative_players(state):
    """The players in seat order: the player to move first."""
    if state.to_move is None:
        raise ValueError("the game is over; there is no player to move")
    i = state.players.index(state.to_move)
    return state.players[i:] + state.players[:i]


def encode_points(state):
    """Encode a (non-terminal) state in point form: (NUM_PLANES, 55) float32."""
    x = np.zeros((NUM_PLANES, NUM_POINTS), dtype=np.float32)
    x[ON_BOARD] = 1
    x[PERIMETER_END] = _PERIMETER_END_BITS
    x[PERIMETER_CENTER] = _PERIMETER_CENTER_BITS
    x[LEGAL_PLACEMENT] = mask_bits(state.legal_placement_mask())
    occ = state.occ
    for k, p in enumerate(relative_players(state)):
        own = state.masks[p]
        x[STONES + k] = mask_bits(own)
        opp = occ & ~own
        empty_cover = full_cover = 0
        for t in state.tris[p]:
            if t.mask & opp:
                full_cover |= t.mask
            else:
                empty_cover |= t.mask
        if empty_cover:
            x[EMPTY_TRI + k] = mask_bits(empty_cover)
        if full_cover:
            x[FULL_TRI + k] = mask_bits(full_cover)
        x[HAND + k] = state.hand[p] / STONES_PER_PLAYER
        x[PERIMETER_COUNT + k] = popcount(own & PERIMETER_MASK) / MAX_PERIMETER_STONES
        x[SEAT + k] = 1
    return x


def to_grid(x):
    """Point form (..., 55) to grid form (..., 9, 9)."""
    g = np.zeros(x.shape[:-1] + (GRID_ROWS, GRID_COLS), dtype=x.dtype)
    g[..., GRID_ROW_INDEX, GRID_COL_INDEX] = x
    return g


def from_grid(g):
    """Grid form (..., 9, 9) to point form (..., 55)."""
    return g[..., GRID_ROW_INDEX, GRID_COL_INDEX]


def encode(state):
    """Encode a (non-terminal) state in grid form: (NUM_PLANES, 9, 9) float32."""
    return to_grid(encode_points(state))


def legal_action_mask(state):
    """Boolean array of shape (NUM_ACTIONS,): True for legal actions."""
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[state.legal_actions()] = True
    return m


def value_target(state, winner):
    """One-hot (4,) float32 over seats relative to state's player to move."""
    v = np.zeros(MAX_PLAYERS, dtype=np.float32)
    v[relative_players(state).index(winner)] = 1
    return v


# Gather indexes: transform_points(x, s) == x[..., _POINT_GATHER[s]], i.e.
# destination column j takes source column _POINT_GATHER[s][j].
_POINT_GATHER = np.array([[POINT_PERM[INVERSE_SYMMETRY[s]][j + 1] - 1 for j in range(NUM_POINTS)]
                          for s in range(NUM_SYMMETRIES)])
_ACTION_GATHER = np.array([ACTION_PERM[INVERSE_SYMMETRY[s]] for s in range(NUM_SYMMETRIES)])


def transform_points(x, s):
    """Apply board symmetry s to point-form data (..., 55)."""
    return x[..., _POINT_GATHER[s]]


def transform_grid(g, s):
    """Apply board symmetry s to grid-form data (..., 9, 9)."""
    return to_grid(transform_points(from_grid(g), s))


def transform_policy(p, s):
    """Apply board symmetry s to a policy (or legal mask) of shape (..., NUM_ACTIONS)."""
    return p[..., _ACTION_GATHER[s]]
