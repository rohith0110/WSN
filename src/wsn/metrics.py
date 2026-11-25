"""Metric records, lifetime events with censoring, and summary statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from wsn.config import to_plain_data
from wsn.models import PacketOutcome, PlainData

BOOTSTRAP_RESAMPLES = 5000
CONFIDENCE_LEVEL = 0.95

PDR_DEFINITIONS: dict[str, str] = {
    "offered_load_pdr_pct": (
        "100 * delivered / offered, where offered counts every scheduled sensing event over the "
        "configured horizon, including events from dead or disconnected sensors and events after "
        "the run stopped early"
    ),
    "active_source_pdr_pct": (
        "100 * delivered / packets offered by sources that were alive when the packet was offered "
        "(supplementary; null when the denominator is 0)"
    ),
    "operational_pdr_pct": (
        "100 * delivered / packets offered during simulated rounds, i.e. while at least one "
        "alive sensor could still reach the sink (supplementary)"
    ),
}

UNITS: dict[str, str] = {
    "energy": "joules (J)",
    "time": "simulation rounds (no conversion to seconds is modelled)",
    "delay": "hop count (abstract units, not milliseconds)",
    "pdr": "percent",
    "policy_compute_time_s": "wall-clock seconds spent inside router decisions",
}


@dataclass(frozen=True, slots=True)
class LifetimeEvent:
    """Round an event occurred, or ``None`` with ``censored=True`` if the run ended first."""

    round_number: int | None

    @property
    def censored(self) -> bool:
        return self.round_number is None

    def to_data(self) -> dict[str, PlainData]:
        return {"round": self.round_number, "censored": self.censored}


@dataclass(frozen=True, slots=True)
class RoundMetrics:
    round: int
    simulated: bool
    """False for rounds after an early stop (frozen network, traffic still offered)."""
    algorithm: str
    seed: int
    alive_nodes: int
    dead_nodes: int
    mean_remaining_energy_j: float
    total_remaining_energy_j: float
    cumulative_energy_used_j: float
    offered_packets_round: int
    delivered_packets_round: int
    offered_packets_cumulative: int
    delivered_packets_cumulative: int
    cumulative_offered_pdr: float
    round_offered_pdr: float
    reachable_alive_fraction: float
    link_failure_drops: int
    congestion_drops: int
    energy_drops: int
    no_route_drops: int
    source_dead_drops: int
    loop_prevented_drops: int
    max_hops_drops: int
    network_down_drops: int
    hops_attempted_round: int

    def to_data(self) -> dict[str, PlainData]:
        data = to_plain_data(asdict(self))
        if not isinstance(data, dict):
            raise TypeError("RoundMetrics must serialise to a mapping")
        return data


@dataclass(frozen=True, slots=True)
class RunSummary:
    algorithm: str
    seed: int
    node_count: int
    config_fingerprint: str
    model_fingerprint: str | None
    rounds_executed: int
    max_rounds: int
    stop_reason: str
    offered_packets: int
    offered_packets_simulated: int
    delivered_packets: int
    packets_from_alive_sources: int
    total_hops_attempted: int
    delivered_hops_total: int
    fnd: LifetimeEvent
    hnd: LifetimeEvent
    lnd: LifetimeEvent
    first_disconnection: LifetimeEvent
    final_alive_nodes: int
    final_reachable_fraction: float
    initial_energy_total_j: float
    remaining_energy_total_j: float
    consumed_energy_total_j: float
    drop_counts: dict[str, int]
    policy_decisions: int
    policy_compute_time_s: float
    timestamp_utc: str

    @property
    def offered_load_pdr_pct(self) -> float:
        return percentage(self.delivered_packets, self.offered_packets) or 0.0

    @property
    def operational_pdr_pct(self) -> float | None:
        """Delivered / offered during simulated rounds only (before any early stop)."""
        return percentage(self.delivered_packets, self.offered_packets_simulated)

    @property
    def active_source_pdr_pct(self) -> float | None:
        return percentage(self.delivered_packets, self.packets_from_alive_sources)

    @property
    def mean_delivered_hops(self) -> float | None:
        if self.delivered_packets == 0:
            return None
        return self.delivered_hops_total / self.delivered_packets

    @property
    def mean_attempted_hops_per_offered(self) -> float:
        return self.total_hops_attempted / self.offered_packets if self.offered_packets else 0.0

    @property
    def throughput_packets_per_round(self) -> float:
        return self.delivered_packets / self.rounds_executed if self.rounds_executed else 0.0

    @property
    def energy_ledger_error_j(self) -> float:
        return self.initial_energy_total_j - (
            self.remaining_energy_total_j + self.consumed_energy_total_j
        )

    def scientific_data(self) -> dict[str, PlainData]:
        """Everything except wall-clock measurements, for reproducibility comparisons."""
        lifetime: dict[str, PlainData] = {
            "fnd": self.fnd.to_data(),
            "hnd": self.hnd.to_data(),
            "lnd": self.lnd.to_data(),
            "first_disconnection": self.first_disconnection.to_data(),
        }
        drops: dict[str, PlainData] = dict(self.drop_counts.items())
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "node_count": self.node_count,
            "config_fingerprint": self.config_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "rounds_executed": self.rounds_executed,
            "max_rounds": self.max_rounds,
            "stop_reason": self.stop_reason,
            "packets": {
                "offered": self.offered_packets,
                "offered_simulated_rounds": self.offered_packets_simulated,
                "delivered": self.delivered_packets,
                "offered_by_alive_sources": self.packets_from_alive_sources,
                "drops": drops,
            },
            "metrics": {
                "offered_load_pdr_pct": self.offered_load_pdr_pct,
                "active_source_pdr_pct": self.active_source_pdr_pct,
                "operational_pdr_pct": self.operational_pdr_pct,
                "mean_delivered_hops": self.mean_delivered_hops,
                "mean_attempted_hops_per_offered": self.mean_attempted_hops_per_offered,
                "throughput_packets_per_round": self.throughput_packets_per_round,
                "final_alive_nodes": self.final_alive_nodes,
                "final_reachable_fraction": self.final_reachable_fraction,
            },
            "lifetime_events": lifetime,
            "energy": {
                "initial_total_j": self.initial_energy_total_j,
                "remaining_total_j": self.remaining_energy_total_j,
                "consumed_total_j": self.consumed_energy_total_j,
                "ledger_error_j": self.energy_ledger_error_j,
            },
            "pdr_definitions": dict(PDR_DEFINITIONS.items()),
            "units": dict(UNITS.items()),
        }

    def to_data(self) -> dict[str, PlainData]:
        data = self.scientific_data()
        data["policy"] = {
            "decisions": self.policy_decisions,
            "compute_time_s": self.policy_compute_time_s,
        }
        data["timestamp_utc"] = self.timestamp_utc
        return data

    def flat_record(self) -> dict[str, PlainData]:
        """One row for per-run CSV tables."""
        record: dict[str, PlainData] = {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "node_count": self.node_count,
            "rounds_executed": self.rounds_executed,
            "stop_reason": self.stop_reason,
            "offered_packets": self.offered_packets,
            "delivered_packets": self.delivered_packets,
            "offered_load_pdr_pct": self.offered_load_pdr_pct,
            "active_source_pdr_pct": self.active_source_pdr_pct,
            "operational_pdr_pct": self.operational_pdr_pct,
            "mean_delivered_hops": self.mean_delivered_hops,
            "throughput_packets_per_round": self.throughput_packets_per_round,
            "energy_used_j": self.consumed_energy_total_j,
            "final_alive_nodes": self.final_alive_nodes,
            "fnd_round": self.fnd.round_number,
            "hnd_round": self.hnd.round_number,
            "lnd_round": self.lnd.round_number,
            "first_disconnection_round": self.first_disconnection.round_number,
            "policy_compute_time_s": self.policy_compute_time_s,
        }
        for outcome in PacketOutcome:
            if outcome is not PacketOutcome.DELIVERED:
                record[f"drops_{outcome.value}"] = self.drop_counts.get(outcome.value, 0)
        return record


def percentage(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator > 0 else None


@dataclass(frozen=True, slots=True)
class SampleStatistics:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def bootstrap_mean_ci(values: Sequence[float], seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean (degenerate for n = 1)."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 1:
        single = float(data[0])
        return single, single
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(BOOTSTRAP_RESAMPLES, data.size), replace=True)
    means = samples.mean(axis=1)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def describe(values: Sequence[float], seed: int = 0) -> SampleStatistics:
    if not values:
        raise ValueError("Cannot describe an empty sample")
    mean = math.fsum(values) / len(values)
    std = (
        math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1
        else 0.0
    )
    low, high = bootstrap_mean_ci(values, seed)
    return SampleStatistics(len(values), mean, std, low, high)
