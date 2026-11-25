"""Human names, colors and descriptions for algorithms, test conditions and metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlgorithmInfo:
    key: str
    name: str
    summary: str
    color: str
    family: str


ALGORITHM_INFO: dict[str, AlgorithmInfo] = {
    info.key: info
    for info in (
        AlgorithmInfo(
            "shortest_path",
            "Shortest path",
            "Always takes the fewest hops to the sink",
            "#8E99A1",
            "Baseline",
        ),
        AlgorithmInfo(
            "energy_aware",
            "Energy-aware",
            "Weighs battery, link quality and traffic before every hop",
            "#6CB4F0",
            "Heuristic",
        ),
        AlgorithmInfo(
            "leach",
            "LEACH clusters",
            "Sensors take random turns as cluster heads",
            "#B08CE8",
            "Clustering",
        ),
        AlgorithmInfo(
            "cluster_adaptive",
            "Adaptive clusters",
            "Chooses cluster heads by battery, coverage and load",
            "#E59A5B",
            "Clustering",
        ),
        AlgorithmInfo(
            "q_learning",
            "Q-Learning",
            "Learns which neighbor to pick from a table of past outcomes",
            "#E7718A",
            "Learned",
        ),
        AlgorithmInfo(
            "dqn",
            "Deep Q-network",
            "A small neural network trained on the same choice",
            "#5FC4AE",
            "Learned",
        ),
    )
}


def algorithm(key: str) -> AlgorithmInfo:
    """Known algorithm, or a neutral placeholder for an unknown key from older results."""
    return ALGORITHM_INFO.get(key, AlgorithmInfo(key, key.replace("_", " "), "", "#8E99A1", ""))


@dataclass(frozen=True, slots=True)
class ConditionInfo:
    folder: str
    name: str
    summary: str


CONDITIONS: tuple[ConditionInfo, ...] = (
    ConditionInfo("standard", "Standard field", "50 sensors, 10 random fields, steady traffic"),
    ConditionInfo("stress", "Harsh conditions", "Bursty traffic, weak links or a sink on the edge"),
    ConditionInfo("scalability", "Network size", "The same test with 25, 50 and 100 sensors"),
    ConditionInfo("smoke", "Quick check", "One small field, for a quick correctness check"),
)

VARIANT_NAMES: dict[str, str] = {
    "n50": "50 sensors",
    "n25": "25 sensors",
    "n100": "100 sensors",
    "bursty": "Bursty traffic",
    "degraded_links": "Weak radio links",
    "edge_sink": "Sink at the edge",
    "base": "Base settings",
}


def variant_name(key: str) -> str:
    return VARIANT_NAMES.get(key, key.replace("_", " ").capitalize())


@dataclass(frozen=True, slots=True)
class MetricInfo:
    key: str
    name: str
    unit: str
    higher_is_better: bool
    explanation: str
    decimals: int = 0


METRICS: dict[str, MetricInfo] = {
    info.key: info
    for info in (
        MetricInfo(
            "delivered_packets",
            "Packets delivered",
            "",
            True,
            "Packets that reached the sink before no sensor could reach it",
        ),
        MetricInfo(
            "fnd_round",
            "First sensor dies",
            "round",
            True,
            "Round when the first battery runs out. Later is better",
        ),
        MetricInfo(
            "first_disconnection_round",
            "First sensor cut off",
            "round",
            True,
            "When a live sensor first loses every path to the sink",
        ),
        MetricInfo(
            "rounds_executed",
            "Rounds until cut off",
            "rounds",
            True,
            "Rounds until no live sensor can reach the sink any more",
        ),
        MetricInfo(
            "energy_used_j",
            "Energy used",
            "J",
            False,
            "Battery drained across all sensors",
            2,
        ),
        MetricInfo(
            "operational_pdr_pct",
            "Delivery rate",
            "%",
            True,
            "Share of packets delivered while sensors could still reach the sink",
            1,
        ),
        MetricInfo(
            "mean_delivered_hops",
            "Hops per packet",
            "",
            False,
            "Average hops a delivered packet took to reach the sink",
            2,
        ),
        MetricInfo(
            "policy_compute_time_s",
            "Decision time",
            "s",
            False,
            "Computer time spent choosing routes in one run. Not charged to sensor batteries",
            2,
        ),
    )
}

LIFETIME_METRICS = frozenset({"fnd_round", "first_disconnection_round", "hnd_round", "lnd_round"})


def format_value(metric: MetricInfo, value: float) -> str:
    text = f"{value:,.{metric.decimals}f}"
    if metric.unit == "%":
        return f"{text}%"
    if metric.unit in {"J", "s"}:
        return f"{text} {metric.unit}"
    return text
