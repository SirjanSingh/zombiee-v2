"""Radio + extraction objective (W4): zone geometry and the seeded zone pick.

Pure helpers, no game state. The zone is picked with its own rng
(`random.Random(f"radio|{seed}")`), so turning extraction on never shifts the
episode rng stream: an episode with extraction on is identical to the same
seed with it off up to the radio step.
"""

from __future__ import annotations

import random

from survivecity_v2_env.layout import GRID_COLS, GRID_ROWS, WALL_CELLS

# Name -> corner cell. Order is fixed: it is part of the seeded pick.
CORNERS: dict[str, tuple[int, int]] = {
    "NW": (0, 0),
    "NE": (0, GRID_COLS - 1),
    "SW": (GRID_ROWS - 1, 0),
    "SE": (GRID_ROWS - 1, GRID_COLS - 1),
}


def zone_bounds(name: str, radius: int) -> tuple[int, int, int, int]:
    """(row_lo, row_hi, col_lo, col_hi) of the square around a corner, clipped to the grid."""
    r, c = CORNERS[name]
    return (max(0, r - radius), min(GRID_ROWS - 1, r + radius),
            max(0, c - radius), min(GRID_COLS - 1, c + radius))


def zone_cells(name: str, radius: int) -> list[tuple[int, int]]:
    """Walkable cells within Chebyshev distance `radius` of the corner, row-major."""
    r0, r1, c0, c1 = zone_bounds(name, radius)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in WALL_CELLS]


def pick_zone(seed: int) -> str:
    """The corner announced by the radio for this episode seed."""
    return random.Random(f"radio|{int(seed)}").choice(list(CORNERS))


def describe_zone(name: str, radius: int) -> str:
    r0, r1, c0, c1 = zone_bounds(name, radius)
    return f"the {name} corner (rows {r0}-{r1}, cols {c0}-{c1})"
