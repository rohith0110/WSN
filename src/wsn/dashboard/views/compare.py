"""Compare page: one test condition, every strategy, every metric with its uncertainty."""

from __future__ import annotations

import streamlit as st

from wsn.artifacts import StoredComparison
from wsn.dashboard import data
from wsn.dashboard.catalog import METRICS, algorithm, format_value, variant_name
from wsn.dashboard.charts import (
    IntervalRow,
    RankRow,
    Segment,
    Series,
    StackRow,
    ValueFormat,
    interval_plot,
    line_chart,
    ranking,
    stacked_bars,
)
from wsn.dashboard.components import empty_state, page_head, panel, show
from wsn.dashboard.data import Workspace
from wsn.dashboard.insights import best, paired, scores

CURVES: dict[str, tuple[str, str, ValueFormat]] = {
    "alive": ("alive_nodes", "Alive sensors", ValueFormat.DECIMAL),
    "battery": ("mean_remaining_energy_j", "Battery left per sensor (J)", ValueFormat.JOULES),
    "delivered": ("delivered_packets_cumulative", "Packets delivered so far", ValueFormat.INTEGER),
    "reachable": (
        "reachable_alive_fraction",
        "Share of sensors that can reach the sink",
        ValueFormat.DECIMAL,
    ),
}
CURVE_NAMES = {
    "alive": "Alive sensors",
    "battery": "Battery left",
    "delivered": "Packets delivered",
    "reachable": "Reachable sensors",
}
DROP_CAUSES: tuple[tuple[str, str, str], ...] = (
    ("drops_link_failure", "Radio link failed", "#8E99A1"),
    ("drops_congestion", "Sender hit its per-round limit", "#E3B341"),
    ("drops_energy_exhausted", "Battery ran out mid-route", "#E7718A"),
    ("drops_no_route", "No route to the sink", "#B08CE8"),
    ("drops_source_dead", "Source sensor already dead", "#5A646B"),
    ("drops_loop_prevented", "Route looped back", "#5FC4AE"),
    ("drops_max_hops", "Too many hops", "#E59A5B"),
)


def render(workspace: Workspace) -> None:
    page_head(
        "Compare every strategy, one test at a time",
        "Every strategy ran on the same fields with the same traffic, so differences "
        "come from the routing alone.",
    )
    conditions = workspace.conditions()
    if not conditions:
        empty_state(
            "No comparisons yet",
            "Run a comparison from the command line, then reload this page.",
            "python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json "
            "--scenario standard --output results/comparisons/standard",
        )
        return
    names = {c.folder: c for c in conditions}
    controls = st.columns([1.4, 1, 1.2], gap="small", vertical_alignment="bottom")
    with controls[0]:
        folder = (
            st.segmented_control(
                "Test",
                options=list(names),
                format_func=lambda key: names[key].name,
                default=conditions[0].folder,
                key="compare_condition",
            )
            or conditions[0].folder
        )
    stored = data.comparison(workspace.comparison_path(folder))
    variants = data.variants(stored)
    with controls[1]:
        variant = (
            st.selectbox("Setting", variants, format_func=variant_name, key="compare_variant")
            if len(variants) > 1
            else variants[0]
        )
    with controls[2]:
        metric_key = st.selectbox(
            "Measure",
            list(METRICS),
            format_func=lambda key: METRICS[key].name,
            key="compare_metric",
        )
    st.caption(names[folder].summary)
    _verdict_and_ranking(stored, variant, metric_key)
    _differences(stored, variant, metric_key)
    _curves(stored, variant)
    _drops(stored, variant)
    _downloads(stored)


def _verdict_and_ranking(stored: StoredComparison, variant: str, metric_key: str) -> None:
    metric = METRICS[metric_key]
    items = scores(stored.per_run, variant, metric)
    leader = best(items, metric)
    complete = [s for s in items if s.complete]
    censored = [s for s in items if not s.complete]
    rows = [
        RankRow(
            algorithm(s.algorithm).name,
            algorithm(s.algorithm).color,
            s.mean,
            format_value(metric, s.mean),
        )
        for s in complete
    ]
    title = (
        f"{algorithm(leader.algorithm).name} is best on {metric.name.lower()}"
        if leader
        else metric.name
    )
    note = metric.explanation + ("" if metric.higher_is_better else ". Lower is better")
    body = ranking(rows, higher_is_better=metric.higher_is_better)
    if censored:
        missing = ", ".join(algorithm(s.algorithm).name for s in censored)
        body += (
            f"<p class='rl-caption'>Not reached in every field, so no average is shown: "
            f"{missing}.</p>"
        )
    show(panel(title, body, note=note))


def _differences(stored: StoredComparison, variant: str, metric_key: str) -> None:
    metric = METRICS[metric_key]
    algorithms = data.algorithms(stored)
    reference_default = "shortest_path" if "shortest_path" in algorithms else algorithms[0]
    st.space("small")
    head = st.columns([2, 1], vertical_alignment="bottom")
    with head[1]:
        reference = st.selectbox(
            "Measured against",
            algorithms,
            index=algorithms.index(reference_default),
            format_func=lambda a: algorithm(a).name,
            key="compare_reference",
        )
    diffs = paired(stored.per_run, variant, metric, reference)
    unit = f" {metric.unit}" if metric.unit and metric.unit not in {"%"} else ""
    rows = [
        IntervalRow(
            algorithm(d.algorithm).name,
            algorithm(d.algorithm).color,
            d.mean,
            d.low,
            d.high,
            f"{d.mean:+,.{metric.decimals}f}{unit}  "
            f"[{d.low:+,.{metric.decimals}f}, {d.high:+,.{metric.decimals}f}]",
        )
        for d in sorted(diffs, key=lambda d: d.mean, reverse=metric.higher_is_better)
    ]
    sure = [d for d in diffs if d.certain]
    note = (
        f"Each dot is the average difference over the same {diffs[0].pairs if diffs else 0} "
        "fields, and its bar the 95% confidence interval. A bar that crosses the center line "
        f"means no clear difference. {len(sure)} of {len(diffs)} differences are clear."
    )
    left_note, right_note = ("Worse", "Better") if metric.higher_is_better else ("Better", "Worse")
    with head[0]:
        st.subheader(f"Difference from {algorithm(reference).name}", anchor=False)
    if not rows:
        st.caption("This measure was not reached on every field, so no paired difference is shown.")
        return
    show(
        panel(
            f"{metric.name}, paired by field",
            interval_plot(rows, left_note=left_note, right_note=right_note),
            note=note,
        )
    )


def _curves(stored: StoredComparison, variant: str) -> None:
    st.space("small")
    choice = (
        st.segmented_control(
            "Show over time",
            options=list(CURVES),
            format_func=lambda key: CURVE_NAMES[key],
            default="alive",
            key="compare_curve",
        )
        or "alive"
    )
    column, label, style = CURVES[choice]
    table = data.mean_by_round(stored, variant, column)
    horizon = data.display_horizon(table)
    table = table[table.index <= horizon]
    series = [
        Series(
            key=str(name),
            name=algorithm(str(name)).name,
            color=algorithm(str(name)).color,
            xs=tuple(float(x) for x in table.index),
            ys=tuple(float(y) for y in table[name]),
        )
        for name in table.columns
    ]
    show(
        panel(
            label,
            line_chart(
                series,
                y_label=label,
                value_format=style,
                animate=False,
                y_max=_curve_ceiling(stored, variant, choice),
            ),
            note=(
                "Average across fields, up to the round the last sensor lost its path to the sink."
            ),
        )
    )


def _curve_ceiling(stored: StoredComparison, variant: str, choice: str) -> float | None:
    """Natural top of the axis: the sensor count for alive sensors, 1 for a share."""
    if choice == "alive":
        rows = stored.per_run[stored.per_run["variant"] == variant]
        return float(rows["node_count"].max())
    if choice == "reachable":
        return 1.0
    return None


def _drops(stored: StoredComparison, variant: str) -> None:
    runs = stored.per_run[stored.per_run["variant"] == variant]
    present = [c for c in DROP_CAUSES if c[0] in runs.columns]
    means = runs.groupby("algorithm", sort=False)[[c[0] for c in present]].mean()
    rows = [
        StackRow(
            algorithm(str(name)).name,
            algorithm(str(name)).color,
            tuple(
                Segment(label, color, float(means.loc[name, col])) for col, label, color in present
            ),
        )
        for name in means.index
    ]
    st.space("small")
    show(
        panel(
            "Why packets were lost while the network was running",
            stacked_bars(rows),
            note="Average lost packets per field, by cause. Packets offered after no sensor "
            "could reach the sink are not included.",
        )
    )


def _downloads(stored: StoredComparison) -> None:
    st.space("small")
    with st.expander("Raw numbers and downloads"):
        st.dataframe(stored.per_run, hide_index=True, width="stretch")
        cols = st.columns(3)
        for column, name in zip(cols, ("per_run.csv", "aggregate.csv", "summary.md"), strict=True):
            path = stored.directory / name
            if path.is_file():
                column.download_button(
                    f"Download {name}",
                    path.read_bytes(),
                    file_name=name,
                    key=f"download_{stored.directory.name}_{name}",
                    width="stretch",
                )
