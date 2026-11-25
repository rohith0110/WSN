from __future__ import annotations

from dataclasses import replace

import pytest

from wsn.config import SimulationConfig
from wsn.models import PacketOutcome
from wsn.network import build_scenario
from wsn.radio import bytes_to_bits
from wsn.routing.shortest_path import ShortestPathRouter
from wsn.simulator import RoutingContractError, Simulation, run_simulation

from .support import LINE, FixedRouter, tiny_config, tiny_scenario

BITS = bytes_to_bits(512)
E_ELEC = 5e-8
RX_J = BITS * E_ELEC


def tx_j(distance_m: float) -> float:
    return BITS * E_ELEC + BITS * 1e-11 * distance_m**2


def test_energy_ledger_and_nonnegative_energy(smoke_config: SimulationConfig) -> None:
    result = run_simulation(build_scenario(smoke_config, 2000), ShortestPathRouter())
    summary = result.summary
    assert abs(summary.energy_ledger_error_j) < 1e-9
    assert summary.remaining_energy_total_j >= 0
    assert all(min(s.energy_j) >= 0.0 for s in result.energy_snapshots)
    assert all(row.total_remaining_energy_j >= 0 for row in result.round_metrics)


def test_every_offered_packet_has_exactly_one_outcome(smoke_config: SimulationConfig) -> None:
    result = run_simulation(build_scenario(smoke_config, 2000), ShortestPathRouter())
    for row in result.round_metrics:
        drops = (
            row.link_failure_drops
            + row.congestion_drops
            + row.energy_drops
            + row.no_route_drops
            + row.source_dead_drops
            + row.loop_prevented_drops
            + row.max_hops_drops
            + row.network_down_drops
        )
        assert row.delivered_packets_round + drops == row.offered_packets_round
    summary = result.summary
    assert summary.delivered_packets + sum(summary.drop_counts.values()) == summary.offered_packets
    assert summary.offered_packets == 25 * smoke_config.scenario.max_rounds


def test_smoke_run_exercises_delivery_and_failures(smoke_config: SimulationConfig) -> None:
    summary = run_simulation(build_scenario(smoke_config, 2000), ShortestPathRouter()).summary
    assert summary.delivered_packets > 0
    assert summary.drop_counts.get("link_failure", 0) > 0
    assert summary.fnd.round_number is not None


def test_successful_two_hop_delivery_charges_exact_energy() -> None:
    scenario = tiny_scenario(LINE, tiny_config(success_probability=0.99))
    sim = Simulation(scenario, ShortestPathRouter())
    sim.state.energy_j[0] = 0.0  # node 0 dead: only node 1 offers, and it has no other route
    assert sim.run().summary.drop_counts == {"no_route": 1, "source_dead": 1}

    sim = Simulation(scenario, ShortestPathRouter())
    result = sim.run()
    consumed = sim.state.consumed_j
    delivered = result.summary.delivered_packets
    # Each delivered packet: node0 sends 10 m to sink; node1 relays via node0 (10 m hops).
    assert delivered >= 1
    assert consumed[0] > 0
    assert consumed[1] > 0
    assert sum(consumed) == pytest.approx(sum(0.2 - e for e in sim.state.energy_j))


def test_sink_battery_is_never_charged() -> None:
    scenario = tiny_scenario(((10.0, 0.0),), tiny_config(success_probability=0.99))
    sim = Simulation(scenario, ShortestPathRouter())
    sim.run()
    assert sim.state.consumed_j == [pytest.approx(tx_j(10.0))]


def test_failed_hop_still_charges_tx_and_rx() -> None:
    scenario = tiny_scenario(LINE, tiny_config(success_probability=0.01))
    sim = Simulation(scenario, FixedRouter({1: 0, 0: scenario.topology.sink_id}))
    result = sim.run()
    assert result.summary.drop_counts.get("link_failure", 0) >= 1
    # Node 1's single attempt to node 0 failed or succeeded; either way both were charged.
    assert sim.state.consumed_j[1] == pytest.approx(tx_j(10.0))
    assert sim.state.consumed_j[0] >= RX_J + tx_j(10.0)


def test_sender_that_cannot_afford_tx_is_exhausted_without_transmitting() -> None:
    scenario = tiny_scenario(LINE, tiny_config(success_probability=0.99))
    sim = Simulation(scenario, FixedRouter({1: 0, 0: scenario.topology.sink_id}))
    sim.state.energy_j[1] = tx_j(10.0) - 1e-12  # just below the budget
    result = sim.run()
    assert sim.state.energy_j[1] == 0.0
    assert sim.state.death_round[1] == 1
    assert result.summary.drop_counts.get("energy_exhausted") == 1
    # Node 0 only paid for its own packet: no receive energy, because nothing was sent to it.
    assert sim.state.consumed_j[0] == pytest.approx(tx_j(10.0))


def test_receiver_that_cannot_afford_rx_is_exhausted() -> None:
    scenario = tiny_scenario(LINE, tiny_config(success_probability=0.99))
    sim = Simulation(scenario, FixedRouter({1: 0, 0: None}))
    sim.state.energy_j[0] = RX_J / 2
    sim.run()
    assert sim.state.energy_j[0] == 0.0
    assert sim.state.consumed_j[1] == pytest.approx(tx_j(10.0))


def test_dead_node_never_relays() -> None:
    scenario = tiny_scenario(LINE, tiny_config(success_probability=0.99))
    sim = Simulation(scenario, FixedRouter({1: 0}))
    sim.state.energy_j[0] = 0.0
    with pytest.raises(RoutingContractError, match="dead node 0"):
        sim.run()


def test_out_of_range_choice_is_rejected() -> None:
    scenario = tiny_scenario(LINE, tiny_config())
    sim = Simulation(scenario, FixedRouter({1: scenario.topology.sink_id}))
    sim.state.energy_j[0] = 0.0
    with pytest.raises(RoutingContractError, match="out of range"):
        sim.run()


def test_revisiting_a_node_is_prevented() -> None:
    scenario = tiny_scenario(LINE, tiny_config(success_probability=0.99))
    result = Simulation(scenario, FixedRouter({1: 0, 0: 1})).run()
    assert result.summary.drop_counts.get("loop_prevented", 0) >= 1


def test_paths_terminate_within_hop_limit(smoke_config: SimulationConfig) -> None:
    result = run_simulation(build_scenario(smoke_config, 2001), ShortestPathRouter())
    limit = smoke_config.scenario.max_hops
    assert all(trace.hops_attempted <= limit for trace in result.traces)
    assert all(len(set(trace.path)) == len(trace.path) for trace in result.traces)


def test_no_path_records_no_route_instead_of_crashing() -> None:
    config = tiny_config(require_connected=False)
    scenario = tiny_scenario(((10.0, 0.0), (90.0, 90.0)), config)
    result = Simulation(scenario, ShortestPathRouter()).run()
    assert result.summary.drop_counts.get("no_route") == 1


def test_single_node_next_to_sink_delivers() -> None:
    scenario = tiny_scenario(((5.0, 0.0),), tiny_config(success_probability=0.99))
    summary = Simulation(scenario, ShortestPathRouter()).run().summary
    assert summary.delivered_packets == 1
    assert summary.mean_delivered_hops == 1.0


def _relay_scenario(capacity: int) -> Simulation:
    # node1 (10,0) is the only relay to the sink for node0 (20,0) and node2 (10,10).
    sensors = ((20.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    scenario = tiny_scenario(sensors, tiny_config(capacity=capacity, success_probability=0.99))
    return Simulation(scenario, ShortestPathRouter())


def test_congestion_depends_on_configured_capacity() -> None:
    constrained = _relay_scenario(1).run().summary.drop_counts.get("congestion", 0)
    relaxed = _relay_scenario(3).run().summary.drop_counts.get("congestion", 0)
    assert constrained == 2
    assert relaxed == 0


def test_zero_deliveries_produce_null_hop_mean() -> None:
    scenario = tiny_scenario(((10.0, 0.0),), tiny_config())
    sim = Simulation(scenario, FixedRouter({}))
    summary = sim.run().summary
    assert summary.delivered_packets == 0
    assert summary.mean_delivered_hops is None
    assert summary.drop_counts == {"no_route": 1}


def test_lifetime_events_are_censored_not_clamped() -> None:
    scenario = tiny_scenario(((10.0, 0.0),), tiny_config(max_rounds=3, success_probability=0.99))
    summary = Simulation(scenario, ShortestPathRouter()).run().summary
    assert summary.fnd.round_number is None
    assert summary.fnd.censored
    assert summary.lnd.to_data() == {"round": None, "censored": True}


def test_early_stop_keeps_offered_load_for_remaining_rounds() -> None:
    config = tiny_config(max_rounds=10, success_probability=0.99, initial_energy_j=1e-4)
    scenario = tiny_scenario(((10.0, 0.0),), config)
    result = Simulation(scenario, ShortestPathRouter()).run()
    summary = result.summary
    assert summary.stop_reason == "no_sensor_can_reach_sink"
    assert summary.rounds_executed < 10
    assert summary.offered_packets == 10
    assert len(result.round_metrics) == 10
    assert not result.round_metrics[-1].simulated
    assert summary.drop_counts["network_down"] == 10 - summary.rounds_executed
    assert summary.lnd.round_number is not None


def test_rounds_override_limits_run(smoke_config: SimulationConfig) -> None:
    config = replace(smoke_config, scenario=replace(smoke_config.scenario, initial_energy_j=1.0))
    result = run_simulation(build_scenario(config, 2000), ShortestPathRouter(), max_rounds=5)
    assert result.summary.rounds_executed == 5
    assert result.summary.offered_packets == 5 * 25
    assert result.summary.stop_reason == "max_rounds"


def test_traces_only_contain_real_outcomes(smoke_config: SimulationConfig) -> None:
    result = run_simulation(build_scenario(smoke_config, 2000), ShortestPathRouter())
    sink = result.scenario.topology.sink_id
    for trace in result.traces:
        if trace.outcome is PacketOutcome.DELIVERED:
            assert trace.path[-1] == sink
            assert trace.hops_attempted == len(trace.path) - 1
        for first, second in zip(trace.path, trace.path[1:], strict=False):
            assert second in result.scenario.topology.neighbors[first]
