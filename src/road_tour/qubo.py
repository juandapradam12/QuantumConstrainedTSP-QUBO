"""QUBO encoding of the constrained tour.

Binary variables x_{i,t} equal 1 when city i is visited at step t.
The depot is fixed at step 0 and city 2 is fixed at step 2. Both cities
are visited only once, so every other variable that would place them, or
that would fill those two steps with a different city, is identically zero.
Nine variables remain: cities {1, 3, 4} at steps {1, 3, 4}.

The energy is

    H = H_travel + λ_road H_missing + λ_onehot (H_step + H_city).

H_travel sums road lengths over consecutive occupations. H_missing counts
consecutive occupations that are not a road. H_step and H_city are squared
one-hot defects. On any tour that is a permutation and stays on the roads,
H equals the tour length.

The quadratic form is stored as a symmetric matrix Q such that

    H(x) = x^T Q x + C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from road_tour.network import CITIES, DEPOT, REQUIRED_CITY, REQUIRED_STEP, has_road, road_length

FREE_CITIES: tuple[int, ...] = tuple(
    city for city in CITIES if city not in (DEPOT, REQUIRED_CITY)
)
FREE_STEPS: tuple[int, ...] = tuple(
    step for step in range(len(CITIES)) if step not in (0, REQUIRED_STEP)
)
N_CITIES = len(CITIES)


@dataclass(frozen=True)
class Qubo:
    variables: tuple[tuple[int, int], ...]
    matrix: np.ndarray
    offset: float
    onehot_penalty: float
    road_penalty: float

    def energy(self, state: np.ndarray) -> float:
        x = np.asarray(state, dtype=float)
        return float(x @ self.matrix @ x + self.offset)

    def index(self, city: int, step: int) -> int:
        return self.variables.index((city, step))


def variable_order() -> tuple[tuple[int, int], ...]:
    """Variables grouped by step, then by city.

    The grouping makes each step one-hot constraint a diagonal block.
    """
    return tuple(
        (city, step)
        for step in FREE_STEPS
        for city in FREE_CITIES
    )


def build_qubo(onehot_penalty: float, road_penalty: float) -> Qubo:
    if onehot_penalty < 0 or road_penalty < 0:
        raise ValueError("penalties must be non-negative")

    variables = variable_order()
    index = {pair: k for k, pair in enumerate(variables)}
    size = len(variables)
    matrix = np.zeros((size, size), dtype=float)
    offset = 0.0

    def add_linear(k: int, coefficient: float) -> None:
        matrix[k, k] += coefficient

    def add_product(a: int, b: int, coefficient: float) -> None:
        matrix[a, b] += coefficient / 2.0
        matrix[b, a] += coefficient / 2.0

    for step in FREE_STEPS:
        members = [index[(city, step)] for city in FREE_CITIES]
        offset += _add_onehot(members, onehot_penalty, add_linear, add_product)

    for city in FREE_CITIES:
        members = [index[(city, step)] for step in FREE_STEPS]
        offset += _add_onehot(members, onehot_penalty, add_linear, add_product)

    _add_hops(index, road_penalty, add_linear, add_product)

    return Qubo(
        variables=variables,
        matrix=matrix,
        offset=offset,
        onehot_penalty=onehot_penalty,
        road_penalty=road_penalty,
    )


def _add_onehot(members: list[int], penalty: float, add_linear, add_product) -> float:
    """Add penalty * (sum y - 1)^2. Returns the constant term.

    For binary y, (sum y - 1)^2 = 1 - sum y + 2 sum_{a<b} y_a y_b.
    """
    for k in members:
        add_linear(k, -penalty)
    for a, b in _pairs(members):
        add_product(a, b, 2.0 * penalty)
    return penalty


def _pairs(members: list[int]):
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            yield a, b


def _add_hops(index: dict[tuple[int, int], int], road_penalty: float, add_linear, add_product) -> None:
    """Bilinear travel cost on the five consecutive step pairs.

    Steps 0 and 2 are fixed, so four of the five hops are linear in the
    free variables. The hop from step 3 to step 4 couples two free
    variables and is the only hop that can leave the road network.
    """
    fixed = {
        0: DEPOT,
        REQUIRED_STEP: REQUIRED_CITY,
    }

    for step in range(N_CITIES):
        nxt = (step + 1) % N_CITIES
        origins = _occupants(step, fixed)
        destinations = _occupants(nxt, fixed)
        for city_a, var_a in origins:
            for city_b, var_b in destinations:
                coefficient = _hop_coefficient(city_a, city_b, road_penalty)
                if var_a is None and var_b is None:
                    raise RuntimeError("two consecutive steps are both fixed")
                if var_a is None:
                    add_linear(index[var_b], coefficient)
                elif var_b is None:
                    add_linear(index[var_a], coefficient)
                else:
                    add_product(index[var_a], index[var_b], coefficient)


def _occupants(step: int, fixed: dict[int, int]) -> tuple[tuple[int, tuple[int, int] | None], ...]:
    if step in fixed:
        return ((fixed[step], None),)
    return tuple((city, (city, step)) for city in FREE_CITIES)


def _hop_coefficient(i: int, j: int, road_penalty: float) -> float:
    length = road_length(i, j)
    if length is None:
        return road_penalty
    return length


def assignment_matrix(state: np.ndarray, variables: tuple[tuple[int, int], ...] | None = None) -> np.ndarray:
    """Return the 5 by 5 occupation matrix, including the two fixed cities."""
    variables = variable_order() if variables is None else variables
    x = np.zeros((N_CITIES, N_CITIES), dtype=int)
    x[DEPOT, 0] = 1
    x[REQUIRED_CITY, REQUIRED_STEP] = 1
    for bit, (city, step) in zip(np.asarray(state, dtype=int), variables):
        x[city, step] = int(bit)
    return x


def direct_energy(
    state: np.ndarray,
    onehot_penalty: float,
    road_penalty: float,
) -> float:
    """Energy from the sums, used to check the quadratic form."""
    x = assignment_matrix(state)
    onehot = 0
    for step in range(N_CITIES):
        onehot += (int(x[:, step].sum()) - 1) ** 2
    for city in range(N_CITIES):
        onehot += (int(x[city, :].sum()) - 1) ** 2

    travel = 0.0
    missing = 0
    for step in range(N_CITIES):
        nxt = (step + 1) % N_CITIES
        for i in range(N_CITIES):
            if not x[i, step]:
                continue
            for j in range(N_CITIES):
                if not x[j, nxt]:
                    continue
                length = road_length(i, j)
                if length is None:
                    missing += 1
                else:
                    travel += length
    return travel + road_penalty * missing + onehot_penalty * onehot


def encode(order: tuple[int, ...]) -> np.ndarray | None:
    """Bit string for a visit order that already obeys the two fixed cities."""
    if len(order) != N_CITIES:
        return None
    if order[0] != DEPOT or order[REQUIRED_STEP] != REQUIRED_CITY:
        return None
    if tuple(sorted(order)) != CITIES:
        return None
    variables = variable_order()
    state = np.zeros(len(variables), dtype=int)
    for k, (city, step) in enumerate(variables):
        state[k] = int(order[step] == city)
    return state


def decode(state: np.ndarray) -> tuple[int, ...] | None:
    """Read a visit order, or None if some step is not occupied by one city."""
    x = assignment_matrix(state)
    if any(int(x[:, step].sum()) != 1 for step in range(N_CITIES)):
        return None
    if any(int(x[city, :].sum()) != 1 for city in range(N_CITIES)):
        return None
    return tuple(int(np.flatnonzero(x[:, step])[0]) for step in range(N_CITIES))


def is_legal(state: np.ndarray) -> bool:
    order = decode(state)
    if order is None:
        return False
    cycle = order + (order[0],)
    return all(has_road(a, b) for a, b in zip(cycle, cycle[1:]))


@dataclass(frozen=True)
class GroundState:
    state: np.ndarray
    energy: float
    order: tuple[int, ...] | None
    legal: bool


def exhaustive(qubo: Qubo) -> tuple[GroundState, np.ndarray]:
    """Evaluate every configuration. The instance has 2^9 = 512 states."""
    n = len(qubo.variables)
    energies = np.empty(1 << n, dtype=float)
    best_state = None
    best_energy = np.inf
    for mask in range(1 << n):
        state = _mask_to_state(mask, n)
        value = qubo.energy(state)
        energies[mask] = value
        if value < best_energy:
            best_energy = value
            best_state = state
    assert best_state is not None
    return (
        GroundState(
            state=best_state,
            energy=float(best_energy),
            order=decode(best_state),
            legal=is_legal(best_state),
        ),
        energies,
    )


def _mask_to_state(mask: int, n: int) -> np.ndarray:
    return np.array([(mask >> bit) & 1 for bit in range(n)], dtype=int)
