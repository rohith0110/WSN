"""Ranked-candidate actions and the discretised local observation used by Q-Learning."""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass

from wsn.config import CandidateWeights, StateBins
from wsn.models import NodeId
from wsn.network import NetworkView
from wsn.routing.base import eligible_neighbors

State = tuple[int, int, int, int, int, int]
"""(energy, sink_distance, load, degree, best_link, best_neighbor_energy) bins."""

NO_CANDIDATE_BIN = 3
"""Sentinel bin for the best-neighbor dimensions when there is no eligible neighbor."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """Observable features of one eligible next hop and its ranking score."""

    node_id: NodeId
    energy_fraction: float
    link_probability: float
    progress_m: float
    utilization: float
    density: float
    tx_cost_ratio: float
    score: float


def _clip_unit(value: float) -> float:
    return min(max(value, -1.0), 1.0)


def describe_candidate(
    view: NetworkView, current: NodeId, node: NodeId, weights: CandidateWeights
) -> Candidate:
    """Score = +energy +link +progress -utilization +density -tx_cost (weights are magnitudes)."""
    is_sink = node == view.sink_id
    energy = view.energy_fraction(node)
    link = view.link_success_probability(current, node)
    progress = view.distance_to_sink_m(current) - view.distance_to_sink_m(node)
    utilization = view.utilization(node)
    density = 1.0 if is_sink else view.density(node)
    tx_cost = view.tx_energy_j(current, node) / view.reference_energy_j
    score = (
        weights.energy * energy
        + weights.link_quality * link
        + weights.progress * _clip_unit(progress / view.communication_range_m)
        - weights.utilization * utilization
        + weights.density * density
        - weights.tx_cost * tx_cost
    )
    return Candidate(node, energy, link, progress, utilization, density, tx_cost, score)


def rank_candidates(
    view: NetworkView,
    current: NodeId,
    visited: Set[NodeId],
    weights: CandidateWeights,
    limit: int,
) -> tuple[Candidate, ...]:
    """Top ``limit`` eligible neighbors by score (ties: lower node id).

    A directly reachable sink always occupies a slot: if it ranks below the cut it replaces the
    last retained candidate. Slot ``i`` therefore means "the i-th ranked candidate".
    """
    described = sorted(
        (
            describe_candidate(view, current, node, weights)
            for node in eligible_neighbors(view, current, visited)
        ),
        key=lambda candidate: (-candidate.score, candidate.node_id),
    )
    retained = described[:limit]
    sink = next((c for c in described if c.node_id == view.sink_id), None)
    if sink is not None and sink not in retained:
        retained = [*retained[: limit - 1], sink]
    return tuple(retained)


def to_bin(value: float, bounds: tuple[float, float]) -> int:
    """0 = low (< first bound), 1 = medium, 2 = high (>= second bound)."""
    if value < bounds[0]:
        return 0
    return 1 if value < bounds[1] else 2


def observe_state(
    view: NetworkView, current: NodeId, candidates: Sequence[Candidate], bins: StateBins
) -> State:
    """Approximate local observation; per-neighbor detail lives in the candidate ranking."""
    if candidates:
        best_link = to_bin(max(c.link_probability for c in candidates), bins.link_quality)
        best_energy = to_bin(max(c.energy_fraction for c in candidates), bins.neighbor_energy)
    else:
        best_link = best_energy = NO_CANDIDATE_BIN
    return (
        to_bin(view.energy_fraction(current), bins.energy),
        to_bin(view.distance_to_sink_m(current) / view.field_diagonal_m, bins.sink_distance),
        to_bin(view.utilization(current), bins.load),
        to_bin(view.density(current), bins.degree),
        best_link,
        best_energy,
    )
