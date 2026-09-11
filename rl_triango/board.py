"""TrianGO board geometry, board state, and terminal rendering.

Points are numbered 1-55, row by row from the top (see Figure 4 in the
manual).  Rows alternate being shifted half a step to the left, so each
point has up to six neighbors (E, NE, NW, W, SW, SE).
"""

import os
import sys

if sys.version_info < (3, 10):
    raise RuntimeError("TrianGO requires Python 3.10 or newer")

# ---------------------------------------------------------------------------
# Players / point contents

EMPTY, RED, BLUE, GREEN, WHITE = 0, 1, 2, 3, 4
PLAYERS = (RED, BLUE, GREEN, WHITE)
PLAYER_NAMES = {RED: "Red", BLUE: "Blue", GREEN: "Green", WHITE: "White"}
PLAYER_CHARS = {RED: "R", BLUE: "B", GREEN: "G", WHITE: "W"}
CHAR_TO_PLAYER = {c: p for p, c in PLAYER_CHARS.items()}

STONES_PER_PLAYER = 14
MAX_PERIMETER_STONES = 6

# ---------------------------------------------------------------------------
# Geometry

NUM_POINTS = 55

# Inclusive column range for each of the 9 rows (same scheme as boardUtil.ms).
ROW_COLS = [(2, 4), (1, 6), (0, 6), (0, 7), (0, 6), (0, 7), (0, 6), (1, 6), (2, 4)]
NUM_ROWS = len(ROW_COLS)

# Directions, in the same order as boardUtil.ms.
E, NE, NW, W, SW, SE = range(6)
DIR_NAMES = ("E", "NE", "NW", "W", "SW", "SE")

# Step in (half-x, row) space for each direction.  "Half-x" is the horizontal
# position in units of half the point spacing: hx = 2*col - (row % 2).
_DIR_STEPS = {E: (2, 0), NE: (1, -1), NW: (-1, -1), W: (-2, 0), SW: (-1, 1), SE: (1, 1)}

# ROW_COL[i] = (row, col) of point i; HX[i] = half-x position of point i.
# Index 0 is unused so that point numbers can be used directly as indexes.
ROW_COL = [None]
HX = [None]
_index_at = {}  # (hx, row) -> point index
for _row, (_c0, _c1) in enumerate(ROW_COLS):
    for _col in range(_c0, _c1 + 1):
        _hx = 2 * _col - (_row % 2)
        ROW_COL.append((_row, _col))
        HX.append(_hx)
        _index_at[(_hx, _row)] = len(ROW_COL) - 1
assert len(ROW_COL) == NUM_POINTS + 1

# NEIGHBOR[i][d] = index of the neighbor of point i in direction d, or 0.
NEIGHBOR = [[0] * 6]
for _i in range(1, NUM_POINTS + 1):
    _row = ROW_COL[_i][0]
    NEIGHBOR.append([_index_at.get((HX[_i] + dx, _row + dy), 0)
                     for dx, dy in (_DIR_STEPS[d] for d in range(6))])

# The six perimeters (double lines), each as [end, middle, end].
PERIMETERS = [
    (1, 2, 3),
    (4, 10, 17),
    (9, 16, 24),
    (32, 40, 47),
    (39, 46, 52),
    (53, 54, 55),
]
PERIMETER_POINTS = frozenset(p for per in PERIMETERS for p in per)

# ---------------------------------------------------------------------------
# Bitmasks: a set of points is an int with bit i set for point i.

BIT = [1 << i for i in range(NUM_POINTS + 1)]
PERIMETER_MASK = sum(BIT[p] for p in PERIMETER_POINTS)

# PERIMETER_CONFLICT[i]: points where the same player's stone would prevent
# placing at i (a perimeter's center conflicts with both of its ends).
PERIMETER_CONFLICT = [0] * (NUM_POINTS + 1)
for _a, _b, _c in PERIMETERS:
    PERIMETER_CONFLICT[_a] = PERIMETER_CONFLICT[_c] = BIT[_b]
    PERIMETER_CONFLICT[_b] = BIT[_a] | BIT[_c]

popcount = int.bit_count


ALL_POINTS_MASK = sum(BIT[1:])

# (center bit, ends mask) for each perimeter.
PERIMETER_BITS = [(BIT[b], BIT[a] | BIT[c]) for a, b, c in PERIMETERS]

# _BYTE_POINTS[k][v]: the points whose bits are set in byte k of a mask, if
# that byte's value is v.
_BYTE_POINTS = [[tuple(8 * k + j for j in range(8) if v >> j & 1) for v in range(256)]
                for k in range((NUM_POINTS + 8) // 8)]


def mask_to_points(mask):
    """The point numbers in a bitmask, in increasing order."""
    result = []
    k = 0
    while mask:
        v = mask & 255
        if v:
            result.extend(_BYTE_POINTS[k][v])
        mask >>= 8
        k += 1
    return result


# ---------------------------------------------------------------------------
# Triangles: every equilateral triangle whose sides run along board lines and
# which contains at least 6 points (side length >= 2).


class Triangle:
    __slots__ = ("index", "corners", "points", "mask", "corner_mask", "size", "points_up")

    def __init__(self, index, corners, points, size, points_up):
        self.index = index
        self.corners = tuple(sorted(corners))
        self.points = tuple(sorted(points))
        self.mask = sum(BIT[p] for p in points)
        self.corner_mask = sum(BIT[p] for p in corners)
        self.size = size            # side length, in steps
        self.points_up = points_up  # True if the apex is at the top

    def name(self):
        return "-".join(str(c) for c in self.corners)

    def __repr__(self):
        return f"Triangle({self.name()})"


def _build_triangles():
    tris = []
    for apex in range(1, NUM_POINTS + 1):
        hx, row = HX[apex], ROW_COL[apex][0]
        for dy in (1, -1):  # 1: apex on top, extending down; -1: apex on bottom
            for n in range(2, NUM_ROWS):
                pts = [_index_at.get((hx - k + 2 * j, row + dy * k), 0)
                       for k in range(n + 1) for j in range(k + 1)]
                if all(pts):
                    corners = (apex, pts[-(n + 1)], pts[-1])
                    tris.append(Triangle(len(tris), corners, pts, n, dy == 1))
    return tris


TRIANGLES = _build_triangles()
TRIANGLE_BY_CORNERS = {t.corners: t for t in TRIANGLES}
# TRIANGLES_AT_CORNER[i]: triangles having point i as a corner.
TRIANGLES_AT_CORNER = [[t for t in TRIANGLES if i in t.corners]
                       for i in range(NUM_POINTS + 1)]

# ---------------------------------------------------------------------------
# Hex-grid coordinates, and a 9x9 grid embedding for convolutional networks.
#
# AXIAL[i] = (q, r) with the center point 28 at (0, 0); r is the row offset.
# The six neighbors of (q, r) are (q+-1, r), (q, r+-1), (q+1, r-1), and
# (q-1, r+1).  GRID_RC[i] = (grid row, grid col) shifts these to a 9x9 grid
# in which neighboring points are always adjacent cells (though not every
# pair of adjacent cells are neighbors).  55 of the 81 cells are on the board.

_CX, _CY = HX[28], ROW_COL[28][0]
AXIAL = [None] + [((HX[i] - _CX - (ROW_COL[i][0] - _CY)) // 2, ROW_COL[i][0] - _CY)
                  for i in range(1, NUM_POINTS + 1)]
_axial_index = {AXIAL[i]: i for i in range(1, NUM_POINTS + 1)}
_q0 = min(q for q, _ in AXIAL[1:])
_r0 = min(r for _, r in AXIAL[1:])
GRID_RC = [None] + [(r - _r0, q - _q0) for q, r in AXIAL[1:]]
GRID_ROWS = 1 + max(rc[0] for rc in GRID_RC[1:])
GRID_COLS = 1 + max(rc[1] for rc in GRID_RC[1:])
assert (GRID_ROWS, GRID_COLS) == (9, 9)

# ---------------------------------------------------------------------------
# Symmetries.  The board has the full symmetry of a regular hexagon: six
# rotations about point 28 and six reflections.
#
# POINT_PERM[s][i] is where point i goes under symmetry s (POINT_PERM[s][0]
# is 0).  s = 0 is the identity; s = 1-5 rotate by 60*s degrees; s = 6-11
# reflect, then rotate by 60*(s-6) degrees.  TRIANGLE_PERM[s][k] is the
# index of the triangle that TRIANGLES[k] maps to.

NUM_SYMMETRIES = 12
SYMMETRY_NAMES = ([f"rot{60 * k}" for k in range(6)] +
                  [f"mirror+rot{60 * k}" for k in range(6)])


def _transform_axial(s, q, r):
    x, z = q, r         # cube coordinates (x, y, z) with x + y + z = 0
    y = -x - z
    if s >= 6:
        y, z = z, y     # reflect
    for _ in range(s % 6):
        x, y, z = -z, -x, -y   # rotate 60 degrees
    return (x, z)


POINT_PERM = [[0] + [_axial_index[_transform_axial(s, *AXIAL[i])]
                     for i in range(1, NUM_POINTS + 1)]
              for s in range(NUM_SYMMETRIES)]
TRIANGLE_PERM = [[TRIANGLE_BY_CORNERS[tuple(sorted(perm[c] for c in t.corners))].index
                  for t in TRIANGLES]
                 for perm in POINT_PERM]
# INVERSE_SYMMETRY[s]: the symmetry that undoes s.
INVERSE_SYMMETRY = [next(s2 for s2 in range(NUM_SYMMETRIES)
                         if all(POINT_PERM[s2][POINT_PERM[s][i]] == i
                                for i in range(NUM_POINTS + 1)))
                    for s in range(NUM_SYMMETRIES)]


def transform_mask(mask, s):
    """Apply symmetry s to a bitmask of points."""
    perm = POINT_PERM[s]
    result = 0
    for i in mask_to_points(mask):
        result |= BIT[perm[i]]
    return result

# ---------------------------------------------------------------------------
# Board state


class Board:
    """The contents of the 55 points.  points[0] is unused."""

    def __init__(self):
        self.points = [EMPTY] * (NUM_POINTS + 1)

    def __getitem__(self, idx):
        return self.points[idx]

    def __setitem__(self, idx, value):
        if not 1 <= idx <= NUM_POINTS:
            raise IndexError(f"point {idx} is not on the board")
        self.points[idx] = value

    def copy(self):
        b = Board()
        b.points = self.points[:]
        return b

    def to_string(self):
        """Compact 55-char encoding: '.' for empty, else R/B/G/W."""
        return "".join(PLAYER_CHARS.get(p, ".") for p in self.points[1:])

    @classmethod
    def from_string(cls, s):
        """Inverse of to_string (whitespace is ignored)."""
        s = "".join(s.split())
        if len(s) != NUM_POINTS:
            raise ValueError(f"expected {NUM_POINTS} characters, got {len(s)}")
        b = cls()
        for i, ch in enumerate(s, start=1):
            b.points[i] = CHAR_TO_PLAYER.get(ch.upper(), EMPTY)
        return b

    def __str__(self):
        return render(self, color=False)


# ---------------------------------------------------------------------------
# Rendering

_ANSI = {
    RED: "\033[1;31m",
    BLUE: "\033[1;94m",
    GREEN: "\033[1;32m",
    WHITE: "\033[1;97m",
    "line": "\033[90m",
    "num": "\033[33m",
    "reset": "\033[0m",
}


def _use_color():
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def colorize(text, player, color=None):
    """Wrap text in the ANSI color for the given player (if color is on)."""
    if color is None:
        color = _use_color()
    return f"{_ANSI[player]}{text}{_ANSI['reset']}" if color else text


def render(board, color=None):
    """Return a multi-line string drawing the board.

    Empty points show their number; occupied points show R, B, G, or W.
    Each point occupies two character cells, so labels are right-aligned
    in those cells.  Perimeters are drawn with double lines.
    """
    if color is None:
        color = _use_color()

    # Each point's label occupies columns x-1 and x of text row 3*row,
    # where x = 3*hx + 4.  Diagonal lines go on the two text rows between.
    def xpos(i):
        return 3 * HX[i] + 4

    height = 3 * (NUM_ROWS - 1) + 1
    width = 3 * max(HX[1:]) + 8
    grid = [[" "] * width for _ in range(height)]
    kinds = [[None] * width for _ in range(height)]  # for coloring

    def put(y, x, ch, kind="line"):
        grid[y][x] = ch
        kinds[y][x] = kind

    double = set()
    for a, b, c in PERIMETERS:
        double.add(frozenset((a, b)))
        double.add(frozenset((b, c)))
    center_hx = (min(HX[1:]) + max(HX[1:])) / 2

    for i in range(1, NUM_POINTS + 1):
        y, x = 3 * ROW_COL[i][0], xpos(i)
        # Horizontal line to the east neighbor.
        j = NEIGHBOR[i][E]
        if j:
            ch = "=" if frozenset((i, j)) in double else "-"
            for xx in range(x + 1, xpos(j)):
                put(y, xx, ch)
        # Diagonals down to the SW and SE neighbors.
        j = NEIGHBOR[i][SW]
        if j:
            put(y + 1, x - 1, "/")
            put(y + 2, x - 2, "/")
            if frozenset((i, j)) in double:
                d = -1 if HX[i] < center_hx else 1   # double on the outside
                put(y + 1, x - 1 + d, "/")
                put(y + 2, x - 2 + d, "/")
        j = NEIGHBOR[i][SE]
        if j:
            put(y + 1, x, "\\")
            put(y + 2, x + 1, "\\")
            if frozenset((i, j)) in double:
                d = -1 if HX[i] < center_hx else 1
                put(y + 1, x + d, "\\")
                put(y + 2, x + 1 + d, "\\")

    # Labels go on top of the lines.
    for i in range(1, NUM_POINTS + 1):
        y, x = 3 * ROW_COL[i][0], xpos(i)
        p = board[i]
        if p == EMPTY:
            label = str(i)
            for k, ch in enumerate(reversed(label)):
                put(y, x - k, ch, "num")
        else:
            put(y, x, PLAYER_CHARS[p], p)

    lines = []
    for y in range(height):
        if not color:
            lines.append("".join(grid[y]).rstrip())
            continue
        out, cur = [], None
        for x in range(width):
            kind = kinds[y][x] if grid[y][x] != " " else None
            if kind != cur:
                out.append(_ANSI["reset"] if kind is None else _ANSI[kind])
                cur = kind
            out.append(grid[y][x])
        if cur is not None:
            out.append(_ANSI["reset"])
        lines.append("".join(out).rstrip())
    return "\n".join(lines)


def print_board(board, color=None):
    print(render(board, color))


if __name__ == "__main__":
    b = Board()
    print_board(b)
    print()
    # The position from the MiniScript debug setup.
    b[28], b[53], b[8], b[43] = RED, BLUE, GREEN, WHITE
    print_board(b)
    print(b.to_string())
