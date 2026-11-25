"""Scenario construction, mutable network state, and the read-only view routers receive."""

from __future__ import annotations

from dataclasses import dataclass

from wsn.config import SimulationConfig
from wsn.models import NodeId
from wsn.radio import LinkQualityModel, RadioEnergyModel
from wsn.topology import Topology, generate_topology
from wsn.traffic import TrafficGenerator


@dataclass(frozen=True, slots=True)
class Scenario:
    """Everything fixed by (config, seed): shared unchanged by every algorithm on that seed."""

    config: SimulationConfig
    seed: int
    topology: Topology
    radio: RadioEnergyModel
    links: LinkQualityModel
    traffic: TrafficGenerator

    @property
    def packet_bits(self) -> int:
        return self.config.scenario.packet_bits

    @property
    def reference_energy_j(self) -> float:
        return self.radio.reference_energy_j(self.packet_bits)


def scenario_from_topology(config: SimulationConfig, seed: int, topology: Topology) -> Scenario:
    return Scenario(
        config=config,
        seed=seed,
        topology=topology,
        radio=RadioEnergyModel.from_config(config.radio),
        links=LinkQualityModel(
            config.radio, topology.communication_range_m, seed, topology.edge_lengths_m()
        ),
        traffic=TrafficGenerator(config.traffic, topology.node_count, seed),
    )


def build_scenario(config: SimulationConfig, seed: int) -> Scenario:
    return scenario_from_topology(config, seed, generate_topology(config.scenario, seed))


class NetworkState:
    """Per-run mutable state. Only the simulation engine changes it."""

    def __init__(self, scenario: Scenario) -> None:
        count = scenario.topology.node_count
        initial = scenario.config.scenario.initial_energy_j
        self.initial_energy_j = initial
        self.energy_j = [initial] * count
        self.consumed_j = [0.0] * count
        self.tx_attempts = [0] * count
        self.death_round: list[int | None] = [None] * count
        self.round_number = 0

    def alive_flags(self) -> list[bool]:
        return [energy > 0.0 for energy in self.energy_j]

    def alive_count(self) -> int:
        return sum(1 for energy in self.energy_j if energy > 0.0)


class NetworkView:
    """Observable, current information only: no future link draws, no mutation methods."""

    def __init__(self, scenario: Scenario, state: NetworkState) -> None:
        self._scenario = scenario
        self._state = state
        self._topology = scenario.topology
        self._capacity = scenario.config.traffic.tx_capacity_per_round
        self._density_reference = scenario.config.routing.density_reference

    @property
    def sink_id(self) -> NodeId:
        return self._topology.sink_id

    @property
    def node_count(self) -> int:
        return self._topology.node_count

    @property
    def round_number(self) -> int:
        return self._state.round_number

    @property
    def communication_range_m(self) -> float:
        return self._topology.communication_range_m

    @property
    def field_diagonal_m(self) -> float:
        return self._scenario.config.scenario.field_diagonal_m

    @property
    def reference_energy_j(self) -> float:
        return self._scenario.reference_energy_j

    def neighbors(self, node: NodeId) -> tuple[NodeId, ...]:
        """Potential (in-range) neighbors, alive or not."""
        return self._topology.neighbors[node]

    def distance_m(self, first: NodeId, second: NodeId) -> float:
        return self._topology.distance_m(first, second)

    def distance_to_sink_m(self, node: NodeId) -> float:
        return self._topology.distance_m(node, self.sink_id)

    def is_alive(self, node: NodeId) -> bool:
        return node == self.sink_id or self._state.energy_j[node] > 0.0

    def energy_fraction(self, node: NodeId) -> float:
        """Residual energy fraction; the unlimited-energy sink reports 1."""
        if node == self.sink_id:
            return 1.0
        return self._state.energy_j[node] / self._state.initial_energy_j

    def utilization(self, node: NodeId) -> float:
        """Used transmission attempts this round / capacity, in [0, 1]. Sink reports 0."""
        if node == self.sink_id:
            return 0.0
        return min(self._state.tx_attempts[node] / self._capacity, 1.0)

    def alive_degree(self, node: NodeId) -> int:
        return sum(1 for other in self._topology.neighbors[node] if self.is_alive(other))

    def density(self, node: NodeId) -> float:
        """``min(alive_degree / density_reference, 1)``."""
        return min(self.alive_degree(node) / self._density_reference, 1.0)

    def link_success_probability(self, first: NodeId, second: NodeId) -> float:
        """Current estimated reliability of the link (the draw itself is not observable)."""
        return self._scenario.links.success_probability(first, second, self._state.round_number)

    def tx_energy_j(self, sender: NodeId, receiver: NodeId) -> float:
        return self._scenario.radio.tx_energy_j(
            self._scenario.packet_bits, self.distance_m(sender, receiver)
        )

    def rx_energy_j(self, receiver: NodeId) -> float:
        """Receiver electronics energy; zero for the sink, whose battery is not modelled."""
        if receiver == self.sink_id:
            return 0.0
        return self._scenario.radio.rx_energy_j(self._scenario.packet_bits)
