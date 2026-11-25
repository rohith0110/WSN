"""Baseline B: weighted shortest path costed by energy, reliability, depletion, load, density."""

from __future__ import annotations

import heapq
from collections.abc import Set

from wsn.config import HeuristicWeights
from wsn.models import NodeId, PacketContext
from wsn.network import NetworkView
from wsn.routing.base import Router, eligible_neighbors


def receiver_cost(view: NetworkView, receiver: NodeId, w: HeuristicWeights) -> float:
    """Receiver-state part of the edge cost; zero for the unlimited-energy sink."""
    if receiver == view.sink_id:
        return 0.0
    return (
        w.energy_cost * view.rx_energy_j(receiver) / view.reference_energy_j
        + w.depletion * (1.0 - view.energy_fraction(receiver))
        + w.utilization * view.utilization(receiver)
        + w.sparsity * (1.0 - view.density(receiver))
    )


def link_cost(view: NetworkView, sender: NodeId, receiver: NodeId, w: HeuristicWeights) -> float:
    """Link part of the edge cost: base + transmit energy + unreliability."""
    return (
        w.base
        + w.energy_cost * view.tx_energy_j(sender, receiver) / view.reference_energy_j
        + w.link_unreliability * (1.0 - view.link_success_probability(sender, receiver))
    )


def edge_cost(view: NetworkView, sender: NodeId, receiver: NodeId, w: HeuristicWeights) -> float:
    """Positive cost of forwarding ``sender -> receiver`` given current observable state.

    ``cost = base + energy*(E_tx + E_rx)/E_ref + link*(1 - p) + depletion*(1 - e(v))
    + utilization*u(v) + sparsity*(1 - density(v))``, with ``E_ref = packet_bits * E_elec`` (J).
    For the sink the receive-energy, depletion, congestion and sparsity terms are zero.
    """
    return link_cost(view, sender, receiver, w) + receiver_cost(view, receiver, w)


def costs_to_sink(
    view: NetworkView, sink_id: NodeId, excluded: Set[NodeId], weights: HeuristicWeights
) -> dict[NodeId, float]:
    """Dijkstra from the sink over directed costs ``cost(u -> v)``, skipping excluded/dead nodes.

    Costs are rebuilt from the current view on every call; the receiver term of each node is
    computed once per call because it does not depend on the sender.
    """
    best = {sink_id: 0.0}
    heap: list[tuple[float, NodeId]] = [(0.0, sink_id)]
    settled: set[NodeId] = set()
    while heap:
        cost_here, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        through_node = cost_here + receiver_cost(view, node, weights)
        for sender in view.neighbors(node):
            if sender in settled or sender in excluded or not view.is_alive(sender):
                continue
            candidate = through_node + link_cost(view, sender, node, weights)
            if candidate < best.get(sender, float("inf")):
                best[sender] = candidate
                heapq.heappush(heap, (candidate, sender))
    return best


class EnergyAwareRouter(Router):
    """Weights are rebuilt from live energy/load on every decision; no stale cached paths."""

    name = "energy_aware"

    def __init__(self, weights: HeuristicWeights) -> None:
        self._weights = weights

    def select_next_hop(
        self,
        current_node_id: NodeId,
        sink_id: NodeId,
        visited_node_ids: Set[NodeId],
        network_view: NetworkView,
        packet_context: PacketContext,
    ) -> NodeId | None:
        remaining = costs_to_sink(network_view, sink_id, visited_node_ids, self._weights)
        options = [
            (edge_cost(network_view, current_node_id, node, self._weights) + remaining[node], node)
            for node in eligible_neighbors(network_view, current_node_id, visited_node_ids)
            if node in remaining
        ]
        if not options:
            return None
        return min(options)[1]
