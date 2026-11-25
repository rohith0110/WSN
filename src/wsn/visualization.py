"""Matplotlib figures built only from recorded simulation data."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from matplotlib.figure import Figure

from wsn.models import NodeId
from wsn.topology import Topology

LOW_ENERGY_FRACTION = 0.3
"""Energy fraction below which a node is drawn as low-energy."""

HEALTHY_COLOR = "#2a9d8f"
LOW_COLOR = "#e9c46a"
DEAD_COLOR = "#9a9a9a"
SINK_COLOR = "#d62828"
PATH_COLOR = "#264653"
ALGORITHM_COLORS: dict[str, str] = {
    "shortest_path": "#6c757d",
    "energy_aware": "#2b7bb9",
    "leach": "#8e5ea2",
    "cluster_adaptive": "#e07a2f",
    "q_learning": "#d1495b",
    "dqn": "#2a9d8f",
}
DROP_COLUMNS: tuple[tuple[str, str], ...] = (
    ("drops_link_failure", "Link failure"),
    ("drops_congestion", "Congestion"),
    ("drops_energy_exhausted", "Energy exhausted"),
    ("drops_no_route", "No route"),
    ("drops_source_dead", "Source dead"),
    ("drops_loop_prevented", "Loop prevented"),
    ("drops_max_hops", "Max hops"),
    ("drops_network_down", "Network down (after stop)"),
)


def _color(algorithm: str) -> str:
    return ALGORITHM_COLORS.get(algorithm, "#555555")


def node_status_colors(energy_j: Sequence[float], initial_energy_j: float) -> list[str]:
    colors: list[str] = []
    for energy in energy_j:
        if energy <= 0.0:
            colors.append(DEAD_COLOR)
        elif energy / initial_energy_j < LOW_ENERGY_FRACTION:
            colors.append(LOW_COLOR)
        else:
            colors.append(HEALTHY_COLOR)
    return colors


def topology_figure(
    topology: Topology,
    energy_j: Sequence[float],
    initial_energy_j: float,
    *,
    title: str,
    path: Sequence[NodeId] = (),
    show_edges: bool = True,
) -> Figure:
    """Nodes coloured healthy / low / dead from recorded energy; optional recorded packet path."""
    figure = Figure(figsize=(6.0, 6.0), layout="constrained")
    axes = figure.add_subplot()
    alive = [energy > 0.0 for energy in energy_j]
    if show_edges:
        for first, second in topology.edge_lengths_m():
            if topology.sink_id in (first, second):
                usable = alive[min(first, second)]
            else:
                usable = alive[first] and alive[second]
            xs = (topology.positions[first][0], topology.positions[second][0])
            ys = (topology.positions[first][1], topology.positions[second][1])
            axes.plot(xs, ys, color="#c8c8c8" if usable else "#eeeeee", linewidth=0.6, zorder=1)
    sensors = topology.positions[:-1]
    axes.scatter(
        [p[0] for p in sensors],
        [p[1] for p in sensors],
        c=node_status_colors(energy_j, initial_energy_j),
        s=46,
        edgecolors="#333333",
        linewidths=0.5,
        zorder=3,
    )
    sink = topology.positions[topology.sink_id]
    axes.scatter([sink[0]], [sink[1]], marker="*", s=320, c=SINK_COLOR, zorder=4, label="Sink")
    if len(path) > 1:
        axes.plot(
            [topology.positions[n][0] for n in path],
            [topology.positions[n][1] for n in path],
            color=PATH_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=4,
            zorder=5,
            label="Recorded packet path",
        )
    for label, color in (
        ("Healthy", HEALTHY_COLOR),
        (f"Low (< {LOW_ENERGY_FRACTION:.0%} energy)", LOW_COLOR),
        ("Dead", DEAD_COLOR),
    ):
        axes.scatter([], [], c=color, s=46, edgecolors="#333333", linewidths=0.5, label=label)
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(title)
    axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return figure


def comparison_line_figure(rounds: pd.DataFrame, column: str, ylabel: str, title: str) -> Figure:
    """Thin lines = individual seeds; thick line = mean across seeds, per algorithm."""
    figure = Figure(figsize=(8.0, 4.5), layout="constrained")
    axes = figure.add_subplot()
    for algorithm, group in rounds.groupby("algorithm", sort=False):
        name = str(algorithm)
        table = group.pivot_table(index="round", columns="seed", values=column)
        for seed in table.columns:
            axes.plot(table.index, table[seed], color=_color(name), alpha=0.18, linewidth=0.8)
        seeds = len(table.columns)
        axes.plot(
            table.index,
            table.mean(axis=1),
            color=_color(name),
            linewidth=2.2,
            label=f"{name} (mean of {seeds} seed{'s' if seeds != 1 else ''})",
        )
    axes.set_xlabel("Round")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    return figure


def drop_reasons_figure(per_run: pd.DataFrame, title: str) -> Figure:
    """Mean number of packets per drop cause, per algorithm (stacked)."""
    figure = Figure(figsize=(8.0, 4.5), layout="constrained")
    axes = figure.add_subplot()
    present = [(col, label) for col, label in DROP_COLUMNS if col in per_run.columns]
    means = per_run.groupby("algorithm", sort=False)[[col for col, _ in present]].mean()
    bottoms = [0.0] * len(means.index)
    labels = [str(name) for name in means.index]
    for col, label in present:
        values = [float(v) for v in means[col]]
        if not any(values):
            continue
        axes.bar(labels, values, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]
    axes.set_ylabel("Packets (mean per run)")
    axes.set_title(title)
    axes.legend(fontsize=8)
    axes.grid(axis="y", alpha=0.3)
    return figure


def training_curve_figure(history: pd.DataFrame) -> Figure:
    figure = Figure(figsize=(8.0, 4.5), layout="constrained")
    reward_axes = figure.add_subplot(1, 2, 1)
    reward_axes.plot(history["episode"], history["mean_reward"], marker="o", color=PATH_COLOR)
    reward_axes.set_xlabel("Episode")
    reward_axes.set_ylabel("Mean reward per attempt")
    reward_axes.set_title("Training reward")
    reward_axes.grid(alpha=0.3)
    pdr_axes = figure.add_subplot(1, 2, 2)
    pdr_axes.plot(
        history["episode"], history["offered_load_pdr_pct"], marker="o", color=HEALTHY_COLOR
    )
    pdr_axes.set_xlabel("Episode")
    pdr_axes.set_ylabel("Offered-load PDR (%)")
    pdr_axes.set_title("Training-episode delivery (training seeds)")
    pdr_axes.grid(alpha=0.3)
    return figure
