"""The five-city road network.

Lengths are the ones written on the roads. Pairs with no road are absent
from the graph; they are not given a finite length. The direct separation
between city 0 and city 2 is one of those absent roads.
"""

from __future__ import annotations

CITIES: tuple[int, ...] = (0, 1, 2, 3, 4)
DEPOT: int = 0
# Second city visited after leaving the depot, so step index 2.
REQUIRED_CITY: int = 2
REQUIRED_STEP: int = 2

# Undirected roads. Each pair is stored once.
EDGE_LENGTH: dict[frozenset[int], float] = {
    frozenset((0, 1)): 3,
    frozenset((0, 3)): 4,
    frozenset((0, 4)): 2,
    frozenset((1, 2)): 3,
    frozenset((1, 3)): 4,
    frozenset((2, 3)): 1,
    frozenset((2, 4)): 6,
}

# Layout follows the drawn map: depot on the left, city 2 on the right,
# city 1 above, cities 3 and 4 below.
POSITIONS: dict[int, tuple[float, float]] = {
    0: (0.00, 0.30),
    1: (1.55, 1.50),
    2: (2.20, 0.10),
    3: (1.25, -1.10),
    4: (-1.10, -1.00),
}


def has_road(i: int, j: int) -> bool:
    if i == j:
        return False
    return frozenset((i, j)) in EDGE_LENGTH


def road_length(i: int, j: int) -> float | None:
    """Length of the road between two cities, or None if no road exists."""
    if i == j:
        return None
    return EDGE_LENGTH.get(frozenset((i, j)))


def max_road_length() -> float:
    return max(EDGE_LENGTH.values())
