"""Plain-language findings computed from stored per-run results.

Every sentence the dashboard states about which strategy is better is produced here from the
numbers on disk, so a different result produces a different sentence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from wsn.dashboard.catalog import METRICS, MetricInfo, algorithm
from wsn.metrics import describe

PER_RUN_COLUMNS: dict[str, str] = {
    "delivered_packets": "delivered_packets",
    "fnd_round": "fnd_round",
    "first_disconnection_round": "first_disconnection_round",
    "rounds_executed": "rounds_executed",
    "energy_used_j": "energy_used_j",
    "operational_pdr_pct": "operational_pdr_pct",
    "mean_delivered_hops": "mean_delivered_hops",
    "policy_compute_time_s": "policy_compute_time_s",
}


@dataclass(frozen=True, slots=True)
class Score:
    algorithm: str
    mean: float
    low: float
    high: float
    runs: int
    observed: int

    @property
    def complete(self) -> bool:
        return self.observed == self.runs


def scores(per_run: pd.DataFrame, variant: str, metric: MetricInfo) -> list[Score]:
    """Mean and bootstrap CI per algorithm; lifetime events count only fully observed runs."""
    rows = per_run[per_run["variant"] == variant]
    result: list[Score] = []
    for name, group in rows.groupby("algorithm", sort=False):
        raw = [float(v) for v in group[PER_RUN_COLUMNS[metric.key]]]
        values = [v for v in raw if not math.isnan(v)]
        if not values:
            continue
        stats = describe(values)
        result.append(
            Score(str(name), stats.mean, stats.ci_low, stats.ci_high, len(raw), len(values))
        )
    return result


def best(items: list[Score], metric: MetricInfo) -> Score | None:
    usable = [s for s in items if s.complete]
    if not usable:
        return None
    return (max if metric.higher_is_better else min)(usable, key=lambda s: s.mean)


def score_for(items: list[Score], name: str) -> Score | None:
    return next((s for s in items if s.algorithm == name), None)


@dataclass(frozen=True, slots=True)
class PairedDifference:
    algorithm: str
    mean: float
    low: float
    high: float
    pairs: int

    @property
    def certain(self) -> bool:
        """The 95% interval excludes zero."""
        return self.low > 0 or self.high < 0


def paired(
    per_run: pd.DataFrame, variant: str, metric: MetricInfo, reference: str
) -> list[PairedDifference]:
    """Per-field differences (algorithm - reference) with a bootstrap CI over fields."""
    rows = per_run[per_run["variant"] == variant]
    column = PER_RUN_COLUMNS[metric.key]
    table = rows.pivot_table(index="seed", columns="algorithm", values=column, aggfunc="first")
    if reference not in table.columns:
        return []
    result: list[PairedDifference] = []
    for name in table.columns:
        if name == reference:
            continue
        pairs = table[[name, reference]].dropna()
        if pairs.empty or len(pairs) < len(table):
            continue
        diffs = [float(a) - float(b) for a, b in zip(pairs[name], pairs[reference], strict=True)]
        stats = describe(diffs)
        result.append(
            PairedDifference(str(name), stats.mean, stats.ci_low, stats.ci_high, len(diffs))
        )
    return result


def times_better(winner: float, other: float) -> str | None:
    if other <= 0:
        return None
    ratio = winner / other
    return f"{ratio:.1f}\u00d7" if ratio >= 1.5 else f"{(ratio - 1) * 100:.0f}%"


@dataclass(frozen=True, slots=True)
class Headline:
    leader: str
    sentence: str


def lifetime_headline(per_run: pd.DataFrame, variant: str) -> Headline | None:
    """Who keeps the first sensor alive longest, and how that compares to shortest path."""
    fnd = METRICS["fnd_round"]
    items = scores(per_run, variant, fnd)
    leader = best(items, fnd)
    if leader is None:
        return None
    name = algorithm(leader.algorithm).name
    sentence = f"{name} keeps every sensor alive for {leader.mean:.0f} rounds on average"
    baseline = score_for(items, "shortest_path")
    if baseline is not None and baseline.complete and leader.algorithm != "shortest_path":
        gain = times_better(leader.mean, baseline.mean)
        if gain is not None:
            sentence += f", {gain} longer than always taking the shortest path"
    delivered = METRICS["delivered_packets"]
    top = best(scores(per_run, variant, delivered), delivered)
    if top is not None and top.algorithm == leader.algorithm:
        sentence += ". It also delivers the most packets"
    elif top is not None:
        sentence += f". {algorithm(top.algorithm).name} delivers the most packets"
    return Headline(leader.algorithm, sentence)


def wins(per_run: pd.DataFrame, metric: MetricInfo) -> dict[str, int]:
    """How many variants each algorithm leads on ``metric``."""
    counts: dict[str, int] = {}
    for variant in dict.fromkeys(str(v) for v in per_run["variant"]):
        leader = best(scores(per_run, variant, metric), metric)
        if leader is not None:
            counts[leader.algorithm] = counts.get(leader.algorithm, 0) + 1
    return counts
