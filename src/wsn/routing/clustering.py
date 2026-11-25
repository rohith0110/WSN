"""Cluster-head routing: LEACH-style baseline and an adaptive, coverage-aware rotation.

Both routers share one data path:

1. Every ``rotation_rounds`` rounds the router elects cluster heads from the current
   observable state, and each alive non-head sensor joins the nearest alive head in range.
2. A packet at its source is sent to the source's cluster head (one hop).
3. Cluster heads, gateways and sensors without a head in range forward along a backbone route:
   Dijkstra from the sink where entering a cluster head or the sink costs 1 and entering any
   other sensor costs ``1 + gateway_penalty``, so routes prefer the head backbone.

Data aggregation is **not** modelled: a cluster head forwards every member packet individually,
so offered load, packet accounting and energy costs stay identical to the other algorithms.
"""

from __future__ import annotations

import heapq
from abc import abstractmethod
from collections.abc import Set

from wsn.config import ClusteringConfig
from wsn.models import NodeId, PacketContext
from wsn.network import NetworkView
from wsn.random_utils import keyed_uniform
from wsn.routing.base import Router, eligible_neighbors


class ClusterRouter(Router):
    def __init__(self, config: ClusteringConfig, scenario_seed: int) -> None:
        self._config = config
        self._seed = scenario_seed
        self._epoch = -1
        self.heads: frozenset[NodeId] = frozenset()
        self.head_of: dict[NodeId, NodeId] = {}
        self.head_history: list[frozenset[NodeId]] = []
        self._relayed: dict[NodeId, int] = {}

    @abstractmethod
    def elect_heads(self, view: NetworkView, epoch: int) -> frozenset[NodeId]:
        """Cluster heads for this epoch, chosen from alive sensors only."""

    def _alive_sensors(self, view: NetworkView) -> list[NodeId]:
        return [node for node in range(view.node_count) if view.is_alive(node)]

    def _rotate_if_due(self, view: NetworkView) -> None:
        epoch = max(view.round_number - 1, 0) // self._config.rotation_rounds
        if epoch == self._epoch:
            return
        self._epoch = epoch
        self.heads = self.elect_heads(view, epoch)
        self.head_history.append(self.heads)
        self._relayed = {}
        self.head_of = {}
        for node in self._alive_sensors(view):
            if node in self.heads:
                continue
            in_range = [h for h in view.neighbors(node) if h in self.heads and view.is_alive(h)]
            if in_range:
                self.head_of[node] = min(in_range, key=lambda h: (view.distance_m(node, h), h))

    def _backbone_costs(
        self, view: NetworkView, sink_id: NodeId, excluded: Set[NodeId]
    ) -> dict[NodeId, float]:
        """Cost-to-sink from every node, entering a head or the sink costs 1, others more."""
        penalty = self._config.gateway_penalty
        best = {sink_id: 0.0}
        heap: list[tuple[float, NodeId]] = [(0.0, sink_id)]
        settled: set[NodeId] = set()
        while heap:
            cost_here, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            step = 1.0 if node == sink_id or node in self.heads else 1.0 + penalty
            for sender in view.neighbors(node):
                if sender in settled or sender in excluded or not view.is_alive(sender):
                    continue
                candidate = cost_here + step
                if candidate < best.get(sender, float("inf")):
                    best[sender] = candidate
                    heapq.heappush(heap, (candidate, sender))
        return best

    def _entry_cost(self, node: NodeId, sink_id: NodeId) -> float:
        return 1.0 if node == sink_id or node in self.heads else 1.0 + self._config.gateway_penalty

    def select_next_hop(
        self,
        current_node_id: NodeId,
        sink_id: NodeId,
        visited_node_ids: Set[NodeId],
        network_view: NetworkView,
        packet_context: PacketContext,
    ) -> NodeId | None:
        self._rotate_if_due(network_view)
        choice = self._member_hop(current_node_id, visited_node_ids, network_view, packet_context)
        if choice is None:
            choice = self._backbone_hop(current_node_id, sink_id, visited_node_ids, network_view)
        if choice is not None:
            self._relayed[choice] = self._relayed.get(choice, 0) + 1
        return choice

    def _member_hop(
        self,
        current: NodeId,
        visited: Set[NodeId],
        view: NetworkView,
        packet: PacketContext,
    ) -> NodeId | None:
        if current != packet.source or current in self.heads:
            return None
        head = self.head_of.get(current)
        if head is None or head in visited or not view.is_alive(head):
            return None
        return head

    def _backbone_hop(
        self, current: NodeId, sink_id: NodeId, visited: Set[NodeId], view: NetworkView
    ) -> NodeId | None:
        remaining = self._backbone_costs(view, sink_id, visited)
        options = [
            (self._entry_cost(node, sink_id) + remaining[node], node)
            for node in eligible_neighbors(view, current, visited)
            if node in remaining
        ]
        return min(options)[1] if options else None

    def relay_load(self, node: NodeId) -> int:
        """How often this router chose ``node`` as next hop in the current epoch."""
        return self._relayed.get(node, 0)


class LeachRouter(ClusterRouter):
    """LEACH-style election: threshold ``p / (1 - p * (e mod 1/p))``, once per cycle.

    Each alive sensor that has not been a head in the current cycle of ``round(1/p)`` epochs
    becomes a head when its keyed random draw falls below the threshold, so roughly a fraction
    ``p`` of sensors lead each epoch and every sensor leads once per cycle.
    """

    name = "leach"

    def __init__(self, config: ClusteringConfig, scenario_seed: int) -> None:
        super().__init__(config, scenario_seed)
        self._cycle_length = max(1, round(1.0 / config.leach_ch_fraction))
        self._led_in_cycle: dict[NodeId, int] = {}

    def threshold(self, epoch: int) -> float:
        p = self._config.leach_ch_fraction
        position = epoch % self._cycle_length
        denominator = 1.0 - p * position
        return 1.0 if denominator <= 0 else min(p / denominator, 1.0)

    def elect_heads(self, view: NetworkView, epoch: int) -> frozenset[NodeId]:
        cycle = epoch // self._cycle_length
        limit = self.threshold(epoch)
        heads = frozenset(
            node
            for node in self._alive_sensors(view)
            if self._led_in_cycle.get(node) != cycle
            and keyed_uniform(self._seed, "leach", epoch, node) < limit
        )
        for node in heads:
            self._led_in_cycle[node] = cycle
        return heads


class AdaptiveClusterRouter(ClusterRouter):
    """Heads chosen greedily by score until every alive sensor is a head or a head's neighbor.

    ``score = energy*e + coverage*density + sink_proximity*(1 - d_sink/diagonal)
    - load*(relayed_last_epoch / max_relayed)``. Sensors below ``min_head_energy_fraction``
    are never elected. Greedy coverage keeps every member within one hop of its head.
    """

    name = "cluster_adaptive"

    def score(self, view: NetworkView, node: NodeId, max_load: int) -> float:
        c = self._config
        load = self.relay_load(node) / max_load if max_load else 0.0
        return (
            c.adaptive_energy * view.energy_fraction(node)
            + c.adaptive_coverage * view.density(node)
            + c.adaptive_sink_proximity
            * (1.0 - view.distance_to_sink_m(node) / view.field_diagonal_m)
            - c.adaptive_load * load
        )

    def elect_heads(self, view: NetworkView, epoch: int) -> frozenset[NodeId]:
        alive = self._alive_sensors(view)
        max_load = max((self.relay_load(node) for node in alive), default=0)
        eligible = [
            node
            for node in alive
            if view.energy_fraction(node) >= self._config.min_head_energy_fraction
        ]
        ranked = sorted(eligible, key=lambda node: (-self.score(view, node, max_load), node))
        covered: set[NodeId] = set()
        heads: set[NodeId] = set()
        for node in ranked:
            if node in covered:
                continue
            heads.add(node)
            covered.add(node)
            covered.update(other for other in view.neighbors(node) if view.is_alive(other))
        return frozenset(heads)
