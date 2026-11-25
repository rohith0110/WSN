from __future__ import annotations

import networkx as nx

from wsn.config import HeuristicWeights, SimulationConfig
from wsn.models import NodeId, PacketContext
from wsn.network import NetworkState, NetworkView, Scenario, build_scenario
from wsn.routing.energy_aware import EnergyAwareRouter, edge_cost
from wsn.routing.shortest_path import ShortestPathRouter

from .support import tiny_config, tiny_scenario

CONTEXT = PacketContext(packet_id=0, source=0, round_number=1, hops_attempted=0)

# node0 (20,0) can reach the sink (0,0) through node1 (10,4) or node2 (10,-4) only.
DIAMOND = ((20.0, 0.0), (10.0, 4.0), (10.0, -4.0))


def _view(scenario: Scenario) -> tuple[NetworkView, NetworkState]:
    state = NetworkState(scenario)
    state.round_number = 1
    return NetworkView(scenario, state), state


class ReliabilityView(NetworkView):
    """View with a hand-set link reliability for selected links (other links unchanged)."""

    def __init__(
        self, scenario: Scenario, state: NetworkState, overrides: dict[tuple[int, int], float]
    ) -> None:
        super().__init__(scenario, state)
        self._overrides = overrides

    def link_success_probability(self, first: NodeId, second: NodeId) -> float:
        key = (min(first, second), max(first, second))
        if key in self._overrides:
            return self._overrides[key]
        return super().link_success_probability(first, second)


def test_shortest_path_picks_a_valid_minimum_hop_neighbor() -> None:
    scenario = tiny_scenario(DIAMOND, tiny_config())
    view, _ = _view(scenario)
    choice = ShortestPathRouter().select_next_hop(0, scenario.topology.sink_id, {0}, view, CONTEXT)
    assert choice == 1  # tie between relays 1 and 2 broken by lowest id


def test_shortest_path_routes_around_dead_and_visited_nodes() -> None:
    scenario = tiny_scenario(DIAMOND, tiny_config())
    view, state = _view(scenario)
    sink = scenario.topology.sink_id
    state.energy_j[1] = 0.0
    assert ShortestPathRouter().select_next_hop(0, sink, {0}, view, CONTEXT) == 2
    assert ShortestPathRouter().select_next_hop(0, sink, {0, 2}, view, CONTEXT) is None


def test_shortest_path_hop_count_matches_networkx(smoke_config: SimulationConfig) -> None:
    scenario = build_scenario(smoke_config, 2003)
    view, _ = _view(scenario)
    sink = scenario.topology.sink_id
    expected = nx.single_source_shortest_path_length(scenario.topology.graph(), sink)
    router = ShortestPathRouter()
    for node in range(scenario.topology.node_count):
        hops, current, visited = 0, node, {node}
        while current != sink:
            nxt = router.select_next_hop(current, sink, visited, view, CONTEXT)
            assert nxt is not None
            visited.add(nxt)
            current, hops = nxt, hops + 1
        assert hops == expected[node]


def test_energy_aware_avoids_low_energy_relay() -> None:
    scenario = tiny_scenario(DIAMOND, tiny_config())
    view, state = _view(scenario)
    router = EnergyAwareRouter(HeuristicWeights())
    sink = scenario.topology.sink_id
    state.energy_j[1] = 0.01
    assert router.select_next_hop(0, sink, {0}, view, CONTEXT) == 2
    state.energy_j[1], state.energy_j[2] = 0.2, 0.01
    assert router.select_next_hop(0, sink, {0}, view, CONTEXT) == 1


def test_energy_aware_avoids_unreliable_link() -> None:
    scenario = tiny_scenario(DIAMOND, tiny_config())
    state = NetworkState(scenario)
    state.round_number = 1
    router = EnergyAwareRouter(HeuristicWeights())
    sink = scenario.topology.sink_id
    weak_first = ReliabilityView(scenario, state, {(0, 1): 0.5, (0, 2): 0.95})
    weak_second = ReliabilityView(scenario, state, {(0, 1): 0.95, (0, 2): 0.5})
    assert router.select_next_hop(0, sink, {0}, weak_first, CONTEXT) == 2
    assert router.select_next_hop(0, sink, {0}, weak_second, CONTEXT) == 1


def test_energy_aware_avoids_busy_relay() -> None:
    scenario = tiny_scenario(DIAMOND, tiny_config())
    view, state = _view(scenario)
    state.tx_attempts[1] = 10
    router = EnergyAwareRouter(HeuristicWeights())
    assert router.select_next_hop(0, scenario.topology.sink_id, {0}, view, CONTEXT) == 2


def test_edge_costs_are_positive_and_sink_has_no_receiver_penalty(
    smoke_config: SimulationConfig,
) -> None:
    scenario = build_scenario(smoke_config, 2000)
    view, state = _view(scenario)
    weights = HeuristicWeights()
    state.energy_j = [0.0001] * scenario.topology.node_count
    state.energy_j[0] = 0.0
    for node, adjacent in enumerate(scenario.topology.neighbors):
        for other in adjacent:
            assert edge_cost(view, node, other, weights) >= weights.base
    sink = scenario.topology.sink_id
    near_sink = scenario.topology.neighbors[sink][0]
    expected = (
        weights.base
        + weights.energy_cost * view.tx_energy_j(near_sink, sink) / view.reference_energy_j
        + weights.link_unreliability * (1 - view.link_success_probability(near_sink, sink))
    )
    assert edge_cost(view, near_sink, sink, weights) == expected


def test_routers_never_choose_visited_or_dead_nodes(smoke_config: SimulationConfig) -> None:
    scenario = build_scenario(smoke_config, 2004)
    view, state = _view(scenario)
    sink = scenario.topology.sink_id
    for node in range(0, scenario.topology.node_count, 3):
        state.energy_j[node] = 0.0
    for router in (ShortestPathRouter(), EnergyAwareRouter(HeuristicWeights())):
        for current in range(1, scenario.topology.node_count, 3):
            visited = {current, *scenario.topology.neighbors[current][:1]}
            choice = router.select_next_hop(current, sink, visited, view, CONTEXT)
            if choice is not None:
                assert choice not in visited
                assert view.is_alive(choice)
                assert choice in scenario.topology.neighbors[current]
