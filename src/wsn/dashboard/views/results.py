"""Home page: the answer first, then the evidence behind it."""

from __future__ import annotations

from html import escape

import streamlit as st

from wsn.artifacts import StoredComparison
from wsn.dashboard import data
from wsn.dashboard.catalog import METRICS, algorithm, format_value, variant_name
from wsn.dashboard.charts import RankRow, Series, ValueFormat, line_chart, ranking
from wsn.dashboard.components import Stat, battery, empty_state, page_head, panel, show, stats
from wsn.dashboard.data import Workspace
from wsn.dashboard.insights import best, lifetime_headline, score_for, scores, times_better
from wsn.dashboard.navigation import PAGES

BASELINE = "shortest_path"
ICONS = {
    "replay": ":material/play_circle:",
    "compare": ":material/compare_arrows:",
    "run": ":material/tune:",
}


def render(workspace: Workspace) -> None:
    conditions = {c.folder: c for c in workspace.conditions()}
    main = conditions.get("standard") or next(iter(conditions.values()), None)
    if main is None:
        page_head(
            "Which routing strategy keeps a sensor field alive the longest?",
            "Run the standard comparison and the answer appears here.",
        )
        empty_state(
            "No results yet",
            "It compares all six strategies on 10 fields and takes about 2 minutes.",
            "python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json "
            "--scenario standard --output results/comparisons/standard",
        )
        return
    stored = data.comparison(workspace.comparison_path(main.folder))
    variant = data.variants(stored)[0]
    headline = lifetime_headline(stored.per_run, variant)
    leader = headline.leader if headline else BASELINE
    page_head(
        "Which routing strategy keeps a sensor field alive the longest?",
        (headline.sentence if headline else "Not enough runs to name a leader") + ".",
        answer_color=algorithm(leader).color,
    )
    _key_figures(stored, variant, leader)
    _lifetime_chart(stored, variant, leader)
    _rankings(stored, variant)
    _consistency(workspace, leader)
    _next_steps()


def _key_figures(stored: StoredComparison, variant: str, leader: str) -> None:
    tiles: list[Stat] = []
    for key, change_word in (
        ("fnd_round", "vs shortest path"),
        ("delivered_packets", "vs shortest path"),
        ("energy_used_j", "vs shortest path"),
    ):
        metric = METRICS[key]
        items = scores(stored.per_run, variant, metric)
        mine = score_for(items, leader)
        base = score_for(items, BASELINE)
        if mine is None:
            continue
        change, strong, worse = "", "", False
        if base is not None and leader != BASELINE and base.mean > 0:
            delta = (mine.mean - base.mean) / base.mean
            worse = delta < 0 if metric.higher_is_better else delta > 0
            strong = f"{delta:+.0%}"
            change = change_word
        tiles.append(
            Stat(
                label=metric.name,
                figure=format_value(metric, mine.mean).replace(" J", ""),
                unit="J" if metric.unit == "J" else ("rounds" if metric.unit == "round" else ""),
                change=change,
                change_strong=strong,
                worse=worse,
            )
        )
    seeds = data.seeds(stored, variant)
    leader_rows = stored.per_run[
        (stored.per_run["variant"] == variant) & (stored.per_run["algorithm"] == leader)
    ]
    alive_share = float(leader_rows["final_alive_nodes"].mean()) / float(
        leader_rows["node_count"].iloc[0]
    )
    tiles.append(
        Stat(
            label="Sensors alive when the run ended",
            figure=f"{alive_share:.0%}",
            change=f"average of {len(seeds)} fields",
            glyph=battery(alive_share),
        )
    )
    show(stats(tiles))
    st.space("small")


def _lifetime_chart(stored: StoredComparison, variant: str, leader: str) -> None:
    alive = data.mean_by_round(stored, variant, "alive_nodes")
    horizon = data.display_horizon(alive)
    alive = alive[alive.index <= horizon]
    order = sorted(alive.columns, key=lambda name: (str(name) != leader, str(name) != BASELINE))
    series = [
        Series(
            key=str(name),
            name=algorithm(str(name)).name,
            color=algorithm(str(name)).color,
            xs=tuple(float(x) for x in alive.index),
            ys=tuple(float(y) for y in alive[name]),
            emphasis=str(name) in {leader, BASELINE},
        )
        for name in order
    ]
    seeds = len(data.seeds(stored, variant))
    show(
        panel(
            "Sensors still alive, round by round",
            line_chart(
                series,
                y_label="Alive sensors",
                value_format=ValueFormat.DECIMAL,
                y_max=float(stored.per_run["node_count"].max()),
                caption=(
                    f"Average of {seeds} fields per strategy. A line goes flat once no sensor "
                    "can reach the sink. The sensors still alive are cut off with charge left."
                ),
            ),
            note="Move across the chart to read exact values. Select a name to show or hide it.",
        )
    )


def _rankings(stored: StoredComparison, variant: str) -> None:
    left, right = st.columns(2, gap="medium")
    for column, key in ((left, "delivered_packets"), (right, "first_disconnection_round")):
        metric = METRICS[key]
        items = [s for s in scores(stored.per_run, variant, metric) if s.complete]
        rows = [
            RankRow(
                algorithm(s.algorithm).name,
                algorithm(s.algorithm).color,
                s.mean,
                format_value(metric, s.mean),
            )
            for s in items
        ]
        with column:
            show(
                panel(
                    metric.name,
                    ranking(rows, higher_is_better=metric.higher_is_better),
                    note=metric.explanation,
                )
            )


def _consistency(workspace: Workspace, leader: str) -> None:
    fnd = METRICS["fnd_round"]
    rows: list[str] = []
    leads = 0
    seen: set[str] = set()
    for condition in workspace.conditions():
        if condition.folder == "smoke":
            continue
        stored = data.comparison(workspace.comparison_path(condition.folder))
        variants = data.variants(stored)
        fingerprints = data.variant_fingerprints(stored)
        for variant in variants:
            fingerprint = fingerprints.get(variant, f"{condition.folder}/{variant}")
            if fingerprint in seen:
                continue  # the same settings already appear under another condition
            seen.add(fingerprint)
            items = scores(stored.per_run, variant, fnd)
            top = best(items, fnd)
            if top is None:
                continue
            leads += top.algorithm == leader
            base = score_for(items, BASELINE)
            gain = times_better(top.mean, base.mean) if base and base.complete else None
            label = condition.name if len(variants) == 1 else variant_name(variant)
            info = algorithm(top.algorithm)
            rows.append(
                f"<li><span>{escape(label)}</span>"
                f"<span class='rl-rank-name'><i class='rl-dot' style='--c:{info.color}'></i>"
                f"<span>{escape(info.name)}</span></span>"
                f"<span class='rl-cond-gain'>{escape(gain + ' longer') if gain else ''}</span></li>"
            )
    if len(rows) < 2:
        return
    show(
        panel(
            f"{algorithm(leader).name} leads in {leads} of {len(rows)} tests",
            "<ul class='rl-cond'>" + "".join(rows) + "</ul>",
            note=(
                "The strategy that keeps the first sensor alive longest in each test, and how "
                "much longer than always taking the shortest path."
            ),
        )
    )


def _next_steps() -> None:
    st.space("small")
    st.subheader("Go deeper", anchor=False)
    cols = st.columns(3, gap="small")
    links = (
        ("Watch a replay", "See two strategies drain the same field side by side", "replay"),
        ("Compare in detail", "Every measure with its uncertainty, test by test", "compare"),
        ("Run your own field", "Choose size, battery and strategy, then watch it run", "run"),
    )
    for column, (title, text, key) in zip(cols, links, strict=True):
        with column, st.container(border=True):
            st.page_link(PAGES.get(key), label=title, icon=ICONS[key])
            st.caption(text)
