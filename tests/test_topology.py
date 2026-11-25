from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest

from wsn.config import ScenarioConfig
from wsn.topology import TopologyError, build_topology, generate_topology, reachable_from_sink


def test_links_are_symmetric_and_within_range() -> None:
    topology = generate_topology(ScenarioConfig(), 2000)
    for node, adjacent in enumerate(topology.neighbors):
        assert node not in adjacent
        for other in adjacent:
            assert node in topology.neighbors[other]
            assert topology.distance_m(node, other) <= topology.communication_range_m


def test_link_exactly_at_range_exists_and_beyond_does_not() -> None:
    topology = build_topology([(30.0, 0.0), (60.0001, 0.0)], (0.0, 0.0), 30.0)
    assert topology.sink_id in topology.neighbors[0]
    assert 0 not in topology.neighbors[1]


def test_graph_matches_neighbor_lists() -> None:
    topology = generate_topology(ScenarioConfig(node_count=30), 2001)
    graph = topology.graph()
    for node, adjacent in enumerate(topology.neighbors):
        assert set(graph.neighbors(node)) == set(adjacent)


def test_generation_is_seeded() -> None:
    config = ScenarioConfig()
    assert generate_topology(config, 2000).positions == generate_topology(config, 2000).positions
    assert generate_topology(config, 2000).positions != generate_topology(config, 2001).positions


def test_generated_topology_connects_every_sensor_to_sink() -> None:
    topology = generate_topology(ScenarioConfig(), 2002)
    assert nx.is_connected(topology.graph())
    assert topology.sink_id == ScenarioConfig().node_count


def test_impossible_topology_fails_clearly() -> None:
    config = replace(ScenarioConfig(), communication_range_m=1.0, topology_max_tries=3)
    with pytest.raises(TopologyError, match="after 3 tries"):
        generate_topology(config, 2000)


def test_disconnected_topology_allowed_when_requirement_disabled() -> None:
    config = replace(ScenarioConfig(), communication_range_m=1.0, topology_require_connected=False)
    topology = generate_topology(config, 2000)
    assert topology.generation_attempts == 1
    assert not topology.sensors_connected_to_sink()


def test_reachability_ignores_dead_relays() -> None:
    topology = build_topology([(10.0, 0.0), (20.0, 0.0)], (0.0, 0.0), 12.0)
    assert reachable_from_sink(topology, [True, True]) == {0, 1}
    assert reachable_from_sink(topology, [False, True]) == set()
    assert reachable_from_sink(topology, [True, False]) == {0}
