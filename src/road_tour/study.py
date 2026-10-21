"""Reproduce the figures and the numerical record of the tour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from road_tour.network import EDGE_LENGTH
from road_tour.plots import (
    ANNEAL_STEPS,
    T_END,
    T_START,
    WORKING_PENALTY,
    annealing_survey,
    penalty_scan,
    write_all,
)
from road_tour.qubo import build_qubo, exhaustive
from road_tour.tours import all_visit_orders, constrained_tours, on_road_tours, shortest

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    tours = all_visit_orders()
    roads = on_road_tours(tours)
    feasible = constrained_tours(roads)
    best = shortest(feasible)
    qubo = build_qubo(WORKING_PENALTY, WORKING_PENALTY)
    ground, _energies = exhaustive(qubo)

    lambdas = np.round(np.arange(0.0, 30.0 + 1e-9, 1.0), 5)
    rows = penalty_scan(lambdas)
    survey = annealing_survey()
    figures = ROOT / "figures"
    spectrum = write_all(figures, rows, survey)

    record = {
        "roads": [
            {"cities": sorted(pair), "length": length}
            for pair, length in sorted(EDGE_LENGTH.items(), key=lambda item: tuple(sorted(item[0])))
        ],
        "visit_orders": len(tours),
        "on_road_tours": [
            {"order": list(tour.order), "length": tour.length, "city_2_step": tour.order.index(2)}
            for tour in roads
        ],
        "feasible_tours": [
            {"order": list(tour.order), "length": tour.length}
            for tour in feasible
        ],
        "optimum_order": list(best.order),
        "optimum_length": best.length,
        "working_penalty": WORKING_PENALTY,
        "qubo_variables": len(qubo.variables),
        "ground_state": {
            "energy": ground.energy,
            "order": list(ground.order) if ground.order else None,
            "legal": ground.legal,
        },
        "penalty_scan": rows,
        "spectrum_lowest": spectrum["levels"],
        "annealing": {
            "temperature_start": T_START,
            "temperature_end": T_END,
            "seeds": 80,
            "steps": list(ANNEAL_STEPS),
            "outcome_share": survey["fractions"],
        },
    }
    out = ROOT / "results" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {figures}")
    print(f"wrote {out}")
    print(
        f"optimum {best.order} length {best.length}; "
        f"ground energy {ground.energy}; "
        f"anneal at {ANNEAL_STEPS[-1]} steps "
        f"{survey['fractions'][-1]['optimum']:.2f}"
    )


if __name__ == "__main__":
    main()
