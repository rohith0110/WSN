"""Shared value types: node identifiers, packet outcomes, hop feedback and traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

NodeId: TypeAlias = int

PlainData: TypeAlias = "bool | int | float | str | list[PlainData] | dict[str, PlainData] | None"
"""JSON/YAML-compatible data: the shape of everything read from config or written as artifacts."""

SINK_LABEL = "sink"


class PacketOutcome(StrEnum):
    """Mutually exclusive final outcome of one offered packet."""

    DELIVERED = "delivered"
    SOURCE_DEAD = "source_dead"
    NO_ROUTE = "no_route"
    LINK_FAILURE = "link_failure"
    ENERGY_EXHAUSTED = "energy_exhausted"
    CONGESTION = "congestion"
    LOOP_PREVENTED = "loop_prevented"
    MAX_HOPS = "max_hops"
    NETWORK_DOWN = "network_down"
    """Offered after the run stopped because no alive sensor could reach the sink."""


DROP_OUTCOMES: tuple[PacketOutcome, ...] = tuple(
    outcome for outcome in PacketOutcome if outcome is not PacketOutcome.DELIVERED
)


@dataclass(frozen=True, slots=True)
class OfferedPacket:
    """One exogenous sensing event, fixed by the scenario and identical for every algorithm."""

    packet_id: int
    source: NodeId
    round_number: int


@dataclass(frozen=True, slots=True)
class PacketContext:
    """What a router may know about the packet it is forwarding."""

    packet_id: int
    source: NodeId
    round_number: int
    hops_attempted: int


@dataclass(frozen=True, slots=True)
class HopOutcome:
    """Engine feedback about one routing decision, after energy charging and the link draw."""

    sender: NodeId
    receiver: NodeId
    transmitted: bool
    """False when the sender could not afford the transmission and nothing was sent."""
    received: bool
    """True when the link draw succeeded and the receiver decoded the packet."""
    delivered: bool
    terminal_outcome: PacketOutcome | None
    """Set when this hop ended the packet (delivery or failure); None when the packet continues."""
    charged_energy_j: float
    newly_exhausted_nodes: int


@dataclass(frozen=True, slots=True)
class PacketTrace:
    """Recorded path of a sampled packet; every listed hop really happened in the simulation."""

    packet_id: int
    round_number: int
    source: NodeId
    path: tuple[NodeId, ...]
    outcome: PacketOutcome
    hops_attempted: int
