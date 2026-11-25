from __future__ import annotations

from dataclasses import replace

import pytest

from wsn.config import ClusteringConfig, SimulationConfig
from wsn.models import PacketContext
from wsn.network import NetworkState, NetworkView, build_scenario
from wsn.routing.clustering import AdaptiveClusterRouter, LeachRouter
from wsn.simulator import run_simulation

CONTEXT = PacketContext(packet_id=0, source=0, round_number=1, hops_attempted=0)


def _view(config: SimulationConfig, seed: int = 2000) -> tuple[NetworkView, NetworkState]:
    scenario = build_scenario(config, seed)
    state = NetworkState(scenario)
    state.round_number = 1
    return NetworkView(scenario, state), state


def test_leach_threshold_matches_formula() -> None:
    router = LeachRouter(ClusteringConfig(leach_ch_fraction=0.1), 1)
    assert router.threshold(0) == pytest.approx(0.1)
    assert router.threshold(5) == pytest.approx(0.1 / (1 - 0.1 * 5))
    assert router.threshold(9) == pytest.approx(1.0)
    assert router.threshold(10) == pytest.approx(0.1)


def test_leach_sensor_leads_at_most_once_per_cycle(smoke_config: SimulationConfig) -> None:
    view, state = _view(smoke_config)
    router = LeachRouter(smoke_config.clustering, 2000)
    rotation = smoke_config.clustering.rotation_rounds
    for epoch in range(10):
        state.round_number = epoch * rotation + 1
        router.select_next_hop(0, view.sink_id, {0}, view, CONTEXT)
    led = [node for heads in router.head_history for node in heads]
    assert len(led) == len(set(led))
    assert len(router.head_history) == 10


def test_members_join_an_alive_head_in_range(smoke_config: SimulationConfig) -> None:
    view, _ = _view(smoke_config)
    for router in (
        LeachRouter(smoke_config.clustering, 2000),
        AdaptiveClusterRouter(smoke_config.clustering, 2000),
    ):
        router.select_next_hop(0, view.sink_id, {0}, view, CONTEXT)
        for member, head in router.head_of.items():
            assert head in router.heads
            assert head in view.neighbors(member)
            assert member not in router.heads


def test_adaptive_heads_cover_every_eligible_sensor(smoke_config: SimulationConfig) -> None:
    view, state = _view(smoke_config)
    state.energy_j[3] = 0.0
    state.energy_j[4] = smoke_config.scenario.initial_energy_j * 0.05
    router = AdaptiveClusterRouter(smoke_config.clustering, 2000)
    router.select_next_hop(0, view.sink_id, {0}, view, CONTEXT)
    assert 3 not in router.heads
    assert 4 not in router.heads  # below min_head_energy_fraction
    for node in range(view.node_count):
        if node in {3, 4}:
            continue
        assert node in router.heads or any(h in router.heads for h in view.neighbors(node))


def test_adaptive_prefers_high_energy_heads(smoke_config: SimulationConfig) -> None:
    view, state = _view(smoke_config)
    energy_only = replace(
        smoke_config.clustering,
        adaptive_coverage=0.0,
        adaptive_sink_proximity=0.0,
        adaptive_load=0.0,
    )
    first = AdaptiveClusterRouter(energy_only, 2000)
    first.select_next_hop(0, view.sink_id, {0}, view, CONTEXT)
    top = min(first.heads)
    state.energy_j[top] = smoke_config.scenario.initial_energy_j * 0.2
    second = AdaptiveClusterRouter(energy_only, 2000)
    second.select_next_hop(0, view.sink_id, {0}, view, CONTEXT)
    assert top not in second.heads


def test_source_sends_to_its_head_first(smoke_config: SimulationConfig) -> None:
    view, _ = _view(smoke_config)
    router = AdaptiveClusterRouter(smoke_config.clustering, 2000)
    router.select_next_hop(0, view.sink_id, {0}, view, CONTEXT)
    member, head = next(iter(router.head_of.items()))
    context = PacketContext(packet_id=1, source=member, round_number=1, hops_attempted=0)
    assert router.select_next_hop(member, view.sink_id, {member}, view, context) == head


@pytest.mark.parametrize("router_class", [LeachRouter, AdaptiveClusterRouter])
def test_cluster_runs_are_valid_and_deterministic(
    smoke_config: SimulationConfig, router_class: type[LeachRouter] | type[AdaptiveClusterRouter]
) -> None:
    scenario = build_scenario(smoke_config, 2000)
    first = run_simulation(scenario, router_class(smoke_config.clustering, 2000))
    second = run_simulation(
        build_scenario(smoke_config, 2000), router_class(smoke_config.clustering, 2000)
    )
    assert first.summary.scientific_data() == second.summary.scientific_data()
    assert first.summary.delivered_packets > 0
    assert abs(first.summary.energy_ledger_error_j) < 1e-9
