"""Baseline A: minimum hop count to the sink over the currently alive graph."""

from __future__ import annotations

from collections import deque
from collections.abc import Set

from wsn.models import NodeId, PacketContext
from wsn.network import NetworkView
from wsn.routing.base import Router, eligible_neighbors


def hop_distances_to_sink(
    view: NetworkView, sink_id: NodeId, excluded: Set[NodeId]
) -> dict[NodeId, int]:
    """BFS from the sink through alive sensors that are not in ``excluded``."""
    distances = {sink_id: 0}
    queue = deque([sink_id])
    while queue:
        node = queue.popleft()
        for other in view.neighbors(node):
            if other in distances or other in excluded or not view.is_alive(other):
                continue
            distances[other] = distances[node] + 1
            queue.append(other)
    return distances


class ShortestPathRouter(Router):
    """Recomputes the hop-count tree on every decision, so deaths and visited nodes are honoured.

    The current node and all previously visited nodes are excluded from the search, so the
    returned hop is the first hop of a shortest loop-free continuation. Ties: lowest node id.
    """

    name = "shortest_path"

    def select_next_hop(
        self,
        current_node_id: NodeId,
        sink_id: NodeId,
        visited_node_ids: Set[NodeId],
        network_view: NetworkView,
        packet_context: PacketContext,
    ) -> NodeId | None:
        distances = hop_distances_to_sink(network_view, sink_id, visited_node_ids)
        reachable = [
            node
            for node in eligible_neighbors(network_view, current_node_id, visited_node_ids)
            if node in distances
        ]
        if not reachable:
            return None
        return min(reachable, key=lambda node: (distances[node], node))
