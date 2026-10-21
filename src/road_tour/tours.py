"""Closed tours on the road network.

A tour is a permutation of the five cities that starts at the depot.
The return from the last city to the depot is part of the tour and is
not stored as an extra entry.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from road_tour.network import CITIES, DEPOT, REQUIRED_CITY, REQUIRED_STEP, road_length


@dataclass(frozen=True)
class Tour:
    order: tuple[int, ...]
    length: float | None
    missing_hops: int

    @property
    def on_road(self) -> bool:
        return self.missing_hops == 0

    @property
    def meets_visit_constraint(self) -> bool:
        return self.order[REQUIRED_STEP] == REQUIRED_CITY

    def hops(self) -> tuple[tuple[int, int, float | None], ...]:
        cycle = self.order + (self.order[0],)
        return tuple(
            (cycle[t], cycle[t + 1], road_length(cycle[t], cycle[t + 1]))
            for t in range(len(self.order))
        )


def all_visit_orders() -> tuple[Tour, ...]:
    others = [city for city in CITIES if city != DEPOT]
    tours = []
    for perm in itertools.permutations(others):
        order = (DEPOT, *perm)
        tours.append(_evaluate(order))
    return tuple(tours)


def _evaluate(order: tuple[int, ...]) -> Tour:
    cycle = order + (order[0],)
    length = 0.0
    missing = 0
    for a, b in zip(cycle, cycle[1:]):
        hop = road_length(a, b)
        if hop is None:
            missing += 1
        else:
            length += hop
    return Tour(order=order, length=None if missing else length, missing_hops=missing)


def on_road_tours(tours: tuple[Tour, ...] | None = None) -> tuple[Tour, ...]:
    source = all_visit_orders() if tours is None else tours
    return tuple(sorted(
        (tour for tour in source if tour.on_road),
        key=lambda tour: (tour.length, tour.order),
    ))


def constrained_tours(tours: tuple[Tour, ...] | None = None) -> tuple[Tour, ...]:
    """On-road tours in which city 2 is the second city after the depot."""
    source = on_road_tours() if tours is None else tours
    return tuple(
        tour for tour in source
        if tour.on_road and tour.meets_visit_constraint
    )


def shortest(tours: tuple[Tour, ...]) -> Tour | None:
    feasible = [tour for tour in tours if tour.on_road]
    if not feasible:
        return None
    return min(feasible, key=lambda tour: (tour.length, tour.order))


def uses_road(tour: Tour, i: int, j: int) -> bool:
    pair = frozenset((i, j))
    cycle = tour.order + (tour.order[0],)
    return any(frozenset(hop) == pair for hop in zip(cycle, cycle[1:]))


def slot_summary(tours: tuple[Tour, ...] | None = None) -> tuple[dict, ...]:
    """On-road tours grouped by the step at which city 2 is visited."""
    source = on_road_tours() if tours is None else tuple(
        tour for tour in tours if tour.on_road
    )
    rows = []
    n = len(CITIES)
    for step in range(1, n):
        group = [tour for tour in source if tour.order[step] == REQUIRED_CITY]
        rows.append({
            "step": step,
            "count": len(group),
            "min_length": None if not group else min(tour.length for tour in group),
            "orders": tuple(tour.order for tour in group),
        })
    return tuple(rows)


