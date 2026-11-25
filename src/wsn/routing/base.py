"""Router contract shared by every routing strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Set

from wsn.models import HopOutcome, NodeId, PacketContext, PacketOutcome
from wsn.network import NetworkView


class Router(ABC):
    """Chooses next hops from observable state. The engine owns energy, links and outcomes."""

    name: str

    @abstractmethod
    def select_next_hop(
        self,
        current_node_id: NodeId,
        sink_id: NodeId,
        visited_node_ids: Set[NodeId],
        network_view: NetworkView,
        packet_context: PacketContext,
    ) -> NodeId | None:
        """Return an alive, in-range, unvisited neighbor, or ``None`` when no route exists."""


class FeedbackRouter(Router):
    """A router that learns from what happened after its decisions (Q-Learning)."""

    @abstractmethod
    def on_hop_result(
        self, network_view: NetworkView, packet_context: PacketContext, outcome: HopOutcome
    ) -> None:
        """Feedback after the engine executed the selected hop."""

    @abstractmethod
    def on_packet_end(self, packet_context: PacketContext, outcome: PacketOutcome) -> None:
        """Called once when a packet terminates, whatever the reason."""


def eligible_neighbors(view: NetworkView, current: NodeId, visited: Set[NodeId]) -> list[NodeId]:
    """Alive, in-range neighbors that the packet has not visited yet."""
    return [node for node in view.neighbors(current) if node not in visited and view.is_alive(node)]
