"""Figures for the constrained tour, in the order the argument uses them."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch

from road_tour.anneal import simulated_annealing
from road_tour.network import CITIES, EDGE_LENGTH, POSITIONS, has_road, road_length
from road_tour.qubo import build_qubo, decode, encode, exhaustive
from road_tour.tours import all_visit_orders, on_road_tours

INK = "#1c1917"
MUTED = "#a8a29e"
PAPER = "#ffffff"
PANEL = "#fafaf9"
OPT = "#c2410c"
OPT_SOFT = "#fdba74"
ALT = "#1d4ed8"
ALT_SOFT = "#93c5fd"
GHOST = "#78716c"
BROKEN = "#d6d3d1"
TEAL = "#0f766e"

# Geometric schedule used throughout the annealing figures.
T_START = 40.0
T_END = 0.05
ANNEAL_SEEDS = tuple(range(80))
ANNEAL_STEPS = (50, 150, 400, 800, 1500, 3000, 5000)
TRACE_SEEDS = (3, 12, 20, 26, 30)
TRACE_STEPS = 5000
WORKING_PENALTY = 25.0


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "medium",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "axes.titleweight": "regular",
        "mathtext.default": "it",
    })


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _node_radius() -> float:
    return 0.20


def _draw_roads(ax, color=MUTED, width=1.6, zorder=1) -> None:
    for pair in EDGE_LENGTH:
        a, b = tuple(pair)
        p, q = POSITIONS[a], POSITIONS[b]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=width, zorder=zorder, solid_capstyle="round")


def _draw_nodes(ax, highlight: int | None = None) -> None:
    radius = _node_radius()
    for city, (x, y) in POSITIONS.items():
        face = TEAL if city == highlight else PAPER
        text = PAPER if city == highlight else INK
        ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=INK, lw=1.4, zorder=3))
        ax.text(x, y, str(city), ha="center", va="center", fontsize=13, color=text, zorder=4)


# (shift along the edge, shift perpendicular to it), in data units.
# The roads 0–3 and 2–4 cross, so those two lengths cannot sit at the midpoint.
_LABEL_SHIFT = {
    frozenset((0, 1)): (0.00, 0.16),
    frozenset((0, 3)): (-0.42, 0.16),
    frozenset((0, 4)): (0.05, 0.15),
    frozenset((1, 2)): (0.00, 0.16),
    frozenset((1, 3)): (0.00, 0.20),
    frozenset((2, 3)): (0.10, -0.16),
    frozenset((2, 4)): (0.62, -0.16),
}


def _edge_label(ax, a: int, b: int, text: str, color=INK) -> None:
    p = np.array(POSITIONS[a], dtype=float)
    q = np.array(POSITIONS[b], dtype=float)
    direction = q - p
    unit = direction / np.linalg.norm(direction)
    normal = np.array([-unit[1], unit[0]])
    along, side = _LABEL_SHIFT[frozenset((a, b))]
    pos = (p + q) / 2 + unit * along + normal * side
    ax.text(
        pos[0], pos[1], text, ha="center", va="center", fontsize=9, color=color, zorder=5,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER, "edgecolor": "none", "alpha": 0.92},
    )


def _arrow(ax, a: int, b: int, color: str, width: float = 2.4) -> None:
    p = np.array(POSITIONS[a], dtype=float)
    q = np.array(POSITIONS[b], dtype=float)
    direction = q - p
    unit = direction / np.linalg.norm(direction)
    start = p + unit * (_node_radius() + 0.02)
    end = q - unit * (_node_radius() + 0.06)
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, lw=width,
        color=color, zorder=2, shrinkA=0, shrinkB=0,
    ))


def _prepare_map(ax) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    xs = [p[0] for p in POSITIONS.values()]
    ys = [p[1] for p in POSITIONS.values()]
    pad = 0.55
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def figure_network(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    _prepare_map(ax)
    _draw_roads(ax)
    for pair, length in EDGE_LENGTH.items():
        a, b = tuple(pair)
        _edge_label(ax, a, b, f"{length:g}")
    _draw_nodes(ax)
    ax.set_title("Eight roads, three missing pairs")

    matrix = np.full((5, 5), np.nan)
    for i in CITIES:
        for j in CITIES:
            length = road_length(i, j)
            if length is not None:
                matrix[i, j] = length
    axm = axes[1]
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad("#f5f5f4")
    image = axm.imshow(matrix, cmap=cmap, vmin=0, vmax=6)
    for i in range(5):
        for j in range(5):
            if np.isnan(matrix[i, j]):
                label, color = "—", MUTED
            else:
                label, color = f"{matrix[i, j]:g}", INK
            axm.text(j, i, label, ha="center", va="center", fontsize=10, color=color)
    axm.set_xticks(range(5))
    axm.set_yticks(range(5))
    axm.set_xlabel("city")
    axm.set_ylabel("city")
    axm.set_title("Road lengths")
    for spine in axm.spines.values():
        spine.set_visible(True)
        spine.set_color("#e7e5e4")
    axm.tick_params(length=0)
    fig.colorbar(image, ax=axm, fraction=0.046, pad=0.04, label="length")
    fig.tight_layout()
    save(fig, path)


def figure_tours(path: Path) -> None:
    tours = all_visit_orders()
    on_road = on_road_tours(tours)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    missing_levels = (0, 1, 2, 3)
    meets, others = [], []
    for missing in missing_levels:
        group = [tour for tour in tours if tour.missing_hops == missing]
        meet = sum(tour.meets_visit_constraint for tour in group)
        meets.append(meet)
        others.append(len(group) - meet)
    x = np.arange(len(missing_levels))
    ax.bar(x, others, color=BROKEN, width=0.72, label="city 2 elsewhere")
    ax.bar(x, meets, bottom=others, color=OPT, width=0.72, label="city 2 at step 2")
    for i, (a, b) in enumerate(zip(others, meets)):
        ax.text(i, a + b + 0.25, str(a + b), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xlabel("missing roads in the visit order")
    ax.set_ylabel("visit orders")
    ax.set_ylim(0, 12)
    ax.set_title("Twenty-four orders, four on the roads")
    ax.legend(loc="upper right")

    ax = axes[1]
    # Pair each cycle with its reverse so the two directions sit together.
    ranked = sorted(on_road, key=lambda tour: (tour.length, 0 if tour.meets_visit_constraint else 1, tour.order))
    labels, lengths, colors = [], [], []
    for tour in ranked:
        route = " → ".join(str(city) for city in tour.order)
        step = tour.order.index(2)
        labels.append(f"{route} → 0")
        lengths.append(tour.length)
        if tour.length == 16 and tour.meets_visit_constraint:
            colors.append(OPT)
        elif tour.length == 16:
            colors.append(OPT_SOFT)
        elif tour.meets_visit_constraint:
            colors.append(ALT)
        else:
            colors.append(ALT_SOFT)
    y = np.arange(len(ranked))[::-1]
    ax.barh(y, lengths, color=colors, height=0.62)
    for yi, length in zip(y, lengths):
        ax.text(length + 0.3, yi, f"{length:g}", va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("length")
    ax.set_xlim(0, 24)
    ax.set_title("Two cycles, each in two directions")
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=OPT, label="length 16, city 2 at step 2"),
            Patch(facecolor=OPT_SOFT, label="length 16, city 2 at step 3"),
            Patch(facecolor=ALT, label="length 19, city 2 at step 2"),
            Patch(facecolor=ALT_SOFT, label="length 19, city 2 at step 3"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.42, -0.18),
        ncol=2,
    )
    fig.tight_layout()
    save(fig, path)


def _monomial_coefficients(matrix: np.ndarray) -> np.ndarray:
    """Coefficient of x_a x_b. Off-diagonal entries of x^T Q x are half of that."""
    view = np.array(matrix, dtype=float, copy=True)
    off = ~np.eye(view.shape[0], dtype=bool)
    view[off] *= 2.0
    return view


def _matrix_panel(ax, matrix: np.ndarray, title: str, diverging: bool) -> None:
    matrix = _monomial_coefficients(matrix)
    if diverging:
        limit = np.max(np.abs(matrix))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = ax.imshow(matrix, cmap="RdBu_r", norm=norm)
    else:
        image = ax.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=max(8, matrix.max()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if abs(value) < 1e-9:
                label = "·"
                color = "#d6d3d1"
            else:
                label = f"{value:g}"
                color = INK if abs(value) < (0.55 * np.max(np.abs(matrix)) if diverging else 5) else PAPER
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    labels = [rf"$x_{{{city},{step}}}$" for city, step in _variables()]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#e7e5e4")
    return image


def _variables():
    from road_tour.qubo import variable_order
    return variable_order()


def figure_qubo(path: Path) -> None:
    travel = build_qubo(0.0, 0.0).matrix
    onehot = build_qubo(1.0, 0.0).matrix - travel
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    image_t = _matrix_panel(axes[0], travel, "Coefficients of the tour length", diverging=False)
    image_h = _matrix_panel(axes[1], onehot, "Step and city penalties, at λ = 1", diverging=True)
    fig.colorbar(image_t, ax=axes[0], fraction=0.046, pad=0.04, label="coefficient")
    fig.colorbar(image_h, ax=axes[1], fraction=0.046, pad=0.04, label="coefficient for λ = 1")
    fig.tight_layout()
    save(fig, path)


def penalty_scan(lambdas: np.ndarray) -> list[dict]:
    ghost = encode((0, 1, 2, 3, 4))
    optimum = encode((0, 4, 2, 3, 1))
    assert ghost is not None and optimum is not None
    rows = []
    for lam in lambdas:
        qubo = build_qubo(float(lam), float(lam))
        ground, energies = exhaustive(qubo)
        at_ground = np.abs(energies - ground.energy) <= 1e-8
        n_ground = int(at_ground.sum())
        classes = set()
        n = len(qubo.variables)
        for mask in np.flatnonzero(at_ground):
            state = np.array([(int(mask) >> bit) & 1 for bit in range(n)], dtype=int)
            visit = decode(state)
            if visit is not None and _on_road(visit):
                classes.add("tour")
            elif visit is not None:
                classes.add("missing-road")
            else:
                classes.add("broken")
        higher = energies[energies > ground.energy + 1e-8]
        gap = float(higher.min() - ground.energy) if len(higher) else 0.0
        rows.append({
            "penalty": float(lam),
            "ground_energy": ground.energy,
            "gap": gap,
            "kind": "+".join(sorted(classes)),
            "order": ground.order,
            "ghost_energy": qubo.energy(ghost),
            "optimum_energy": qubo.energy(optimum),
            "n_ground": n_ground,
        })
    return rows


def _on_road(order: tuple[int, ...]) -> bool:
    cycle = order + (order[0],)
    return all(has_road(a, b) for a, b in zip(cycle, cycle[1:]))


def figure_penalty(path: Path, rows: list[dict]) -> None:
    lam = np.array([row["penalty"] for row in rows])
    ground = np.array([row["ground_energy"] for row in rows])
    ghost = np.array([row["ghost_energy"] for row in rows])
    optimum = np.array([row["optimum_energy"] for row in rows])
    gap = np.array([row["gap"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    ax = axes[0]
    ax.plot(lam, ghost, color=GHOST, lw=1.5, ls="--", label="0 → 1 → 2 → 3 → 4, missing road 3–4")
    ax.plot(lam, optimum, color=OPT, lw=1.5, label="0 → 4 → 2 → 3 → 1, on the roads")
    ax.scatter(
        lam, ground, s=26, c=INK, edgecolors=PAPER, linewidths=0.7, zorder=4,
        label="ground-state energy",
    )
    ax.set_xlabel("penalty λ")
    ax.set_ylabel("energy")
    ax.set_xlim(lam.min(), lam.max())
    ax.set_ylim(-2, 42)
    ax.set_title("The missing road wins until λ passes 7")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(lam, gap, color=INK, lw=1.6, label="gap above the ground state")
    ax.axhline(3, color=OPT, lw=1.1, ls="--", label="the two on-road tours differ by 3")
    ax.set_xlabel("penalty λ")
    ax.set_ylabel("gap")
    ax.set_xlim(lam.min(), lam.max())
    ax.set_ylim(0, 8)
    ax.set_title("Above λ = 10 the gap locks at 3")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, path)


def figure_spectrum(path: Path) -> dict:
    qubo = build_qubo(WORKING_PENALTY, WORKING_PENALTY)
    _ground, energies = exhaustive(qubo)
    order = np.argsort(energies)
    lowest = order[:12]
    labels = []
    colors = []
    records = []
    for rank, mask in enumerate(lowest):
        state = np.array([(int(mask) >> bit) & 1 for bit in range(9)], dtype=int)
        visit = decode(state)
        energy = float(energies[mask])
        if visit == (0, 4, 2, 3, 1):
            color, name = OPT, "shortest tour"
        elif visit == (0, 4, 2, 1, 3):
            color, name = ALT, "other on-road tour"
        elif visit is not None:
            color, name = GHOST, "tour with a missing road"
        else:
            color, name = BROKEN, "broken assignment"
        colors.append(color)
        text = "—" if visit is None else "→".join(str(c) for c in visit)
        labels.append(text)
        records.append({"rank": rank, "energy": energy, "order": visit, "class": name})

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    x = np.arange(len(lowest))
    ax.vlines(x, 0, [energies[m] for m in lowest], color=colors, lw=2)
    ax.scatter(x, [energies[m] for m in lowest], c=colors, s=42, zorder=3)
    for i in (0, 1, 2):
        ax.text(
            i, energies[lowest[i]] + 2.2, labels[i], ha="center", va="bottom", fontsize=8, color=INK,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER, "edgecolor": "none", "alpha": 0.9},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("rank of the configuration")
    ax.set_ylabel("energy")
    ax.set_ylim(0, 80)
    ax.set_title("At λ = 25 the lowest states are the six visit orders")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=OPT, lw=0, label="0 → 4 → 2 → 3 → 1, energy 16"),
        Line2D([0], [0], marker="o", color=ALT, lw=0, label="0 → 4 → 2 → 1 → 3, energy 19"),
        Line2D([0], [0], marker="o", color=GHOST, lw=0, label="visit order using one missing road"),
        Line2D([0], [0], marker="o", color=BROKEN, lw=0, label="not a visit order"),
    ], loc="upper left")
    fig.tight_layout()
    save(fig, path)
    return {"levels": records, "penalty": WORKING_PENALTY}


def _outcome(energy: float) -> str:
    if abs(energy - 16) < 1e-6:
        return "optimum"
    if abs(energy - 19) < 1e-6:
        return "second"
    if energy < 50:
        return "ghost"
    return "broken"


def annealing_survey() -> dict:
    qubo = build_qubo(WORKING_PENALTY, WORKING_PENALTY)
    fractions = []
    for steps in ANNEAL_STEPS:
        counts = {"optimum": 0, "second": 0, "ghost": 0, "broken": 0}
        for seed in ANNEAL_SEEDS:
            run = simulated_annealing(qubo, steps, T_START, T_END, seed)
            counts[_outcome(run.best_energy)] += 1
        total = len(ANNEAL_SEEDS)
        fractions.append({
            "steps": steps,
            **{key: value / total for key, value in counts.items()},
        })
    traces = []
    for seed in TRACE_SEEDS:
        run = simulated_annealing(qubo, TRACE_STEPS, T_START, T_END, seed)
        traces.append({
            "seed": seed,
            "best_trace": run.best_trace,
            "final_energy": run.best_energy,
        })
    return {"fractions": fractions, "traces": traces}


def figure_annealing(path: Path, survey: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    ax = axes[0]
    for trace in survey["traces"]:
        ax.plot(trace["best_trace"], color=OPT, alpha=0.75, lw=1.3)
    from matplotlib.lines import Line2D
    for level, color in ((16, OPT), (19, ALT), (34, GHOST)):
        ax.axhline(level, color=color, lw=0.9, ls="--", alpha=0.95)
    ax.set_xlim(0, 1800)
    ax.set_ylim(10, 70)
    ax.set_xlabel("bit flips")
    ax.set_ylabel("best energy found")
    ax.set_title("Five runs fall through the tour energies")
    ax.legend(handles=[
        Line2D([0], [0], color=OPT, lw=1.4, label="best energy so far"),
        Line2D([0], [0], color=OPT, lw=0.9, ls="--", label="energy 16"),
        Line2D([0], [0], color=ALT, lw=0.9, ls="--", label="energy 19"),
        Line2D([0], [0], color=GHOST, lw=0.9, ls="--", label="energy 34"),
    ], loc="upper right")

    ax = axes[1]
    steps = [row["steps"] for row in survey["fractions"]]
    keys = ("broken", "ghost", "second", "optimum")
    colors = (BROKEN, GHOST, ALT, OPT)
    labels = (
        "not a visit order",
        "visit order with a missing road",
        "on-road tour of length 19",
        "on-road tour of length 16",
    )
    stack = np.vstack([[row[key] for row in survey["fractions"]] for key in keys])
    ax.stackplot(steps, stack, colors=colors, labels=labels)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("bit flips")
    ax.set_ylabel("share of 80 runs")
    ax.set_title("Longer anneals collect on the shortest tour")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    save(fig, path)


def figure_route(path: Path) -> None:
    chosen = (0, 4, 2, 3, 1)
    alternative = (0, 4, 2, 1, 3)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    _prepare_map(ax)
    _draw_roads(ax, color="#e7e5e4", width=4.0)
    cycle = chosen + (chosen[0],)
    for a, b in zip(cycle, cycle[1:]):
        _arrow(ax, a, b, OPT, width=2.6)
        _edge_label(ax, a, b, f"{road_length(a, b):g}", color=OPT)
    _draw_nodes(ax, highlight=2)
    ax.set_title("The tour of length 16")

    ax = axes[1]

    def _cumulative(order: tuple[int, ...]) -> list[float]:
        cities = order + (order[0],)
        total = [0.0]
        for a, b in zip(cities, cities[1:]):
            total.append(total[-1] + road_length(a, b))
        return total

    chosen_length = _cumulative(chosen)
    other_length = _cumulative(alternative)
    ax.plot(range(3), chosen_length[:3], color=INK, lw=2.2, marker="o", label="shared 0 → 4 → 2")
    ax.plot(range(2, 6), chosen_length[2:], color=OPT, lw=2.2, marker="o", label="then 2 → 3 → 1 → 0")
    ax.plot(range(2, 6), other_length[2:], color=ALT, lw=2.2, marker="o", label="then 2 → 1 → 3 → 0")
    ax.text(5.12, chosen_length[-1], "16", color=OPT, va="center", fontsize=10)
    ax.text(5.12, other_length[-1], "19", color=ALT, va="center", fontsize=10)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["0", "1", "2", "3", "4", "return"])
    ax.set_xlabel("step")
    ax.set_ylabel("length so far")
    ax.set_xlim(-0.15, 5.7)
    ax.set_ylim(0, 23)
    ax.set_title("The two returns split after city 2")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, path)


def write_all(directory: Path, rows: list[dict], survey: dict, spectrum: dict | None = None) -> dict:
    apply_style()
    figure_network(directory / "01_network.png")
    figure_tours(directory / "02_tours.png")
    figure_qubo(directory / "03_qubo.png")
    figure_penalty(directory / "04_penalty.png", rows)
    spectrum = figure_spectrum(directory / "05_spectrum.png")
    figure_annealing(directory / "06_annealing.png", survey)
    figure_route(directory / "07_route.png")
    return spectrum
