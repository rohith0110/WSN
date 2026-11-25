"""Test helpers: small hand-built topologies, fast configurations and a fixed router."""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import replace
from pathlib import Path

from wsn.config import (
    RadioConfig,
    ScenarioConfig,
    SimulationConfig,
    TrafficConfig,
)
from wsn.models import NodeId, PacketContext
from wsn.network import NetworkView, Scenario, scenario_from_topology
from wsn.routing.base import Router
from wsn.topology import Position, build_topology

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_PATH = PROJECT_ROOT / "configs" / "smoke.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def tiny_config(
    *,
    capacity: int = 12,
    initial_energy_j: float = 0.2,
    success_probability: float | None = None,
    max_rounds: int = 1,
    require_connected: bool = True,
) -> SimulationConfig:
    """Config for hand-built topologies; optional fixed link success probability."""
    radio = RadioConfig()
    if success_probability is not None:
        radio = replace(
            radio,
            min_success_probability=success_probability,
            max_success_probability=success_probability,
            base_link_quality_min=success_probability,
            base_link_quality_max=success_probability,
        )
    return SimulationConfig(
        scenario=ScenarioConfig(
            initial_energy_j=initial_energy_j,
            max_rounds=max_rounds,
            topology_require_connected=require_connected,
        ),
        radio=radio,
        traffic=TrafficConfig(tx_capacity_per_round=capacity),
    )


def tiny_scenario(
    sensors: Sequence[Position],
    config: SimulationConfig,
    *,
    sink: Position = (0.0, 0.0),
    communication_range_m: float = 12.0,
    seed: int = 1,
) -> Scenario:
    adjusted = replace(
        config,
        scenario=replace(
            config.scenario, node_count=len(sensors), communication_range_m=communication_range_m
        ),
    )
    topology = build_topology(sensors, sink, communication_range_m)
    return scenario_from_topology(adjusted, seed, topology)


LINE = ((10.0, 0.0), (20.0, 0.0))
"""sink(0,0) <- node0(10,0) <- node1(20,0); range 12 m gives a 2-hop line."""


class FixedRouter(Router):
    """Test router that always returns a preset choice per node."""

    name = "fixed"

    def __init__(self, choices: dict[NodeId, NodeId | None]) -> None:
        self.choices = choices

    def select_next_hop(
        self,
        current_node_id: NodeId,
        sink_id: NodeId,
        visited_node_ids: Set[NodeId],
        network_view: NetworkView,
        packet_context: PacketContext,
    ) -> NodeId | None:
        return self.choices.get(current_node_id)
