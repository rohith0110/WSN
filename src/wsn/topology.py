"""Static 2D topology: seeded node placement, potential links and sink reachability."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from wsn.config import ScenarioConfig
from wsn.models import NodeId
from wsn.radio import edge_key
from wsn.random_utils import stream_rng

logger = logging.getLogger(__name__)

Position = tuple[float, float]


class TopologyError(RuntimeError):
    """Raised when no acceptable topology can be generated with the given parameters."""


@dataclass(frozen=True, slots=True)
class Topology:
    """Sensors ``0..N-1`` plus the sink, whose id is ``N`` (stored last in ``positions``)."""

    positions: tuple[Position, ...]
    communication_range_m: float
    neighbors: tuple[tuple[NodeId, ...], ...]
    generation_attempts: int = 1

    @property
    def node_count(self) -> int:
        return len(self.positions) - 1

    @property
    def sink_id(self) -> NodeId:
        return self.node_count

    def distance_m(self, first: NodeId, second: NodeId) -> float:
        return math.dist(self.positions[first], self.positions[second])

    def edge_lengths_m(self) -> dict[tuple[NodeId, NodeId], float]:
        return {
            edge_key(node, other): self.distance_m(node, other)
            for node, adjacent in enumerate(self.neighbors)
            for other in adjacent
            if node < other
        }

    def graph(self) -> nx.Graph[NodeId]:
        """NetworkX graph of all potential links (sink included)."""
        graph: nx.Graph[NodeId] = nx.Graph()
        graph.add_nodes_from(range(len(self.positions)))
        for (first, second), length in self.edge_lengths_m().items():
            graph.add_edge(first, second, distance_m=length)
        return graph

    def sensors_connected_to_sink(self) -> bool:
        graph = self.graph()
        return len(nx.node_connected_component(graph, self.sink_id)) == len(self.positions)


def build_topology(
    sensor_positions: Sequence[Position],
    sink_position: Position,
    communication_range_m: float,
    generation_attempts: int = 1,
) -> Topology:
    """Potential link between any two devices with distance ``<= communication_range_m``."""
    positions = (*tuple(sensor_positions), sink_position)
    neighbors = tuple(
        tuple(
            other
            for other in range(len(positions))
            if other != node
            and math.dist(positions[node], positions[other]) <= communication_range_m
        )
        for node in range(len(positions))
    )
    return Topology(positions, communication_range_m, neighbors, generation_attempts)


def generate_topology(config: ScenarioConfig, seed: int) -> Topology:
    """Uniform random placement; regenerate (logged) until every sensor can reach the sink."""
    rng = stream_rng(seed, "topology")
    sink = (config.sink_x_m, config.sink_y_m)
    for attempt in range(1, config.topology_max_tries + 1):
        xs = rng.uniform(0.0, config.area_width_m, config.node_count)
        ys = rng.uniform(0.0, config.area_height_m, config.node_count)
        sensors = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
        topology = build_topology(sensors, sink, config.communication_range_m, attempt)
        if not config.topology_require_connected or topology.sensors_connected_to_sink():
            if attempt > 1:
                logger.info("Seed %d: connected topology found on attempt %d", seed, attempt)
            return topology
        logger.info(
            "Seed %d: topology attempt %d not connected to sink; regenerating", seed, attempt
        )
    raise TopologyError(
        f"No sink-connected topology for seed {seed} after {config.topology_max_tries} tries "
        f"({config.node_count} nodes, range {config.communication_range_m} m, "
        f"area {config.area_width_m} x {config.area_height_m} m). "
        "Increase the radio range or node count, or disable topology_require_connected."
    )


def reachable_from_sink(topology: Topology, alive: Sequence[bool]) -> set[NodeId]:
    """Alive sensors with a multi-hop path of alive sensors to the sink (sink excluded)."""
    sink = topology.sink_id
    seen = {sink}
    queue = deque([sink])
    while queue:
        node = queue.popleft()
        for other in topology.neighbors[node]:
            if other not in seen and other != sink and alive[other]:
                seen.add(other)
                queue.append(other)
    seen.discard(sink)
    return seen
