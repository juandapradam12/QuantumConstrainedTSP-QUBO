"""Simulated annealing on the QUBO energy.

A move flips one binary variable. A downhill move is always accepted. An
uphill move of size ΔE is accepted with probability exp(-ΔE / T). The
temperature falls geometrically from t_start to t_end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from road_tour.qubo import Qubo


@dataclass(frozen=True)
class AnnealRun:
    best_state: np.ndarray
    best_energy: float
    best_trace: np.ndarray
    current_trace: np.ndarray
    temperature: np.ndarray


def simulated_annealing(
    qubo: Qubo,
    steps: int,
    t_start: float,
    t_end: float,
    seed: int,
) -> AnnealRun:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if t_start <= 0 or t_end <= 0:
        raise ValueError("temperatures must be positive")

    rng = np.random.default_rng(seed)
    n = len(qubo.variables)
    state = rng.integers(0, 2, size=n)
    current = qubo.energy(state)
    best_state = state.copy()
    best = current

    best_trace = np.empty(steps, dtype=float)
    current_trace = np.empty(steps, dtype=float)
    temperature = np.geomspace(t_start, t_end, steps)

    for k, temp in enumerate(temperature):
        bit = int(rng.integers(0, n))
        state[bit] = 1 - state[bit]
        proposal = qubo.energy(state)
        delta = proposal - current
        if delta <= 0 or rng.random() < np.exp(-delta / temp):
            current = proposal
            if current < best:
                best = current
                best_state = state.copy()
        else:
            state[bit] = 1 - state[bit]
        best_trace[k] = best
        current_trace[k] = current

    return AnnealRun(
        best_state=best_state.copy(),
        best_energy=float(best),
        best_trace=best_trace,
        current_trace=current_trace,
        temperature=temperature,
    )


def success_rate(
    qubo: Qubo,
    steps: int,
    t_start: float,
    t_end: float,
    seeds: tuple[int, ...],
    target_energy: float,
    atol: float = 1e-8,
) -> float:
    hits = 0
    for seed in seeds:
        run = simulated_annealing(qubo, steps, t_start, t_end, seed)
        if abs(run.best_energy - target_energy) <= atol:
            hits += 1
    return hits / len(seeds)
