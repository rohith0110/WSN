from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wsn.config import SimulationConfig
from wsn.experiments import ComparisonRun, aggregate_runs, paired_differences
from wsn.metrics import bootstrap_mean_ci, describe, percentage
from wsn.network import build_scenario
from wsn.reporting import format_cell, write_run_artifacts
from wsn.routing.energy_aware import EnergyAwareRouter
from wsn.routing.shortest_path import ShortestPathRouter
from wsn.simulator import run_simulation

ROUND_COLUMNS = {
    "round",
    "algorithm",
    "seed",
    "alive_nodes",
    "dead_nodes",
    "mean_remaining_energy_j",
    "total_remaining_energy_j",
    "cumulative_energy_used_j",
    "offered_packets_round",
    "delivered_packets_round",
    "offered_packets_cumulative",
    "delivered_packets_cumulative",
    "cumulative_offered_pdr",
    "reachable_alive_fraction",
    "link_failure_drops",
    "congestion_drops",
    "energy_drops",
    "no_route_drops",
}


def test_percentage_handles_zero_denominator() -> None:
    assert percentage(1, 4) == 25.0
    assert percentage(0, 0) is None


def test_describe_and_bootstrap() -> None:
    stats = describe([1.0, 2.0, 3.0, 4.0])
    assert stats.mean == 2.5
    assert stats.std == pytest.approx(1.2909944)
    assert stats.ci_low <= stats.mean <= stats.ci_high
    assert bootstrap_mean_ci([5.0]) == (5.0, 5.0)
    assert bootstrap_mean_ci([1.0, 9.0], seed=1) == bootstrap_mean_ci([1.0, 9.0], seed=1)
    with pytest.raises(ValueError, match="empty"):
        describe([])


def test_missing_values_render_as_not_available() -> None:
    assert format_cell(None) == "N/A"
    assert format_cell(float("nan")) == "N/A"
    assert format_cell(0.123456) == "0.1235"


def test_run_artifacts_parse_with_expected_schema(
    smoke_config: SimulationConfig, tmp_path: Path
) -> None:
    result = run_simulation(build_scenario(smoke_config, 2000), ShortestPathRouter())
    out = write_run_artifacts(result, tmp_path / "run")
    rounds = pd.read_csv(out / "round_metrics.csv")
    assert set(rounds.columns) >= ROUND_COLUMNS
    assert len(rounds) == smoke_config.scenario.max_rounds
    assert int(rounds["delivered_packets_cumulative"].iloc[-1]) == result.summary.delivered_packets
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["algorithm"] == "shortest_path"
    assert summary["packets"]["delivered"] == result.summary.delivered_packets
    assert set(summary["lifetime_events"]) == {"fnd", "hnd", "lnd", "first_disconnection"}
    assert summary["lifetime_events"]["lnd"] == {"round": None, "censored": True}
    assert "offered_load_pdr_pct" in summary["pdr_definitions"]
    nodes = pd.read_csv(out / "nodes_initial.csv")
    assert len(nodes) == 26
    assert nodes["is_sink"].sum() == 1
    traces = (out / "sampled_packet_traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert traces
    assert all("outcome" in json.loads(line) for line in traces)
    assert (out / "topology.png").stat().st_size > 0
    scenario = json.loads((out / "scenario.json").read_text(encoding="utf-8"))
    assert scenario["sink_id"] == "sink"


def test_aggregate_withholds_censored_means(smoke_config: SimulationConfig) -> None:
    runs = []
    for seed in (2000, 2001):
        scenario = build_scenario(smoke_config, seed)
        runs.append(ComparisonRun("v", run_simulation(scenario, ShortestPathRouter())))
        runs.append(
            ComparisonRun(
                "v", run_simulation(scenario, EnergyAwareRouter(smoke_config.routing.heuristic))
            )
        )
    rows = {(r["algorithm"], r["metric"]): r for r in aggregate_runs(runs)}
    lnd = rows[("shortest_path", "lnd_round")]
    assert lnd["n_observed"] == 0
    assert lnd["mean"] is None
    assert lnd["note"] == "not reached in any run"
    delivered = rows[("shortest_path", "delivered_packets")]
    summaries = [r.result.summary for r in runs if r.result.algorithm == "shortest_path"]
    assert delivered["mean"] == pytest.approx(
        sum(s.delivered_packets for s in summaries) / len(summaries)
    )
    diffs = paired_differences(runs)
    contrast = next(
        d
        for d in diffs
        if d["contrast"] == "energy_aware - shortest_path" and d["metric"] == "delivered_packets"
    )
    assert contrast["n_complete_pairs"] == 2
