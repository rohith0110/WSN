"""Shared acting/learning loop for routers that choose among ranked candidate slots.

Tabular Q-Learning and the DQN differ only in how they observe a decision point, estimate
Q-values and learn from a transition. Everything else (candidate ranking, epsilon-greedy over
valid slots, reward computation, deferred bootstrapping, frozen evaluation) lives here once.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence, Set
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

import numpy as np

from wsn.config import CandidateWeights, RewardWeights
from wsn.models import HopOutcome, NodeId, PacketContext, PacketOutcome
from wsn.network import NetworkView
from wsn.random_utils import stream_rng
from wsn.rl.reward import attempt_reward
from wsn.rl.state import Candidate, rank_candidates
from wsn.routing.base import FeedbackRouter

ObservationT = TypeVar("ObservationT")


class RouterMode(StrEnum):
    TRAIN = "train"
    EVALUATE = "evaluate"


def greedy_slot(q_values: Sequence[float], valid_count: int) -> int:
    """Highest Q among valid slots; deterministic tie-break: lowest slot (best-ranked)."""
    valid = list(q_values[:valid_count])
    return valid.index(max(valid))


@dataclass(frozen=True, slots=True)
class NextObservation(Generic[ObservationT]):
    """The state a continuing packet reached, and how many slots are valid there."""

    observation: ObservationT
    valid_count: int


@dataclass(frozen=True, slots=True)
class _Decision(Generic[ObservationT]):
    observation: ObservationT
    action: int
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class _Pending(Generic[ObservationT]):
    """A non-terminal transition waiting for the next state to bootstrap from."""

    observation: ObservationT
    action: int
    reward: float


@dataclass(slots=True)
class LearningStats:
    interactions: int = 0
    updates: int = 0
    reward_total: float = 0.0
    exploratory_actions: int = 0

    @property
    def mean_reward(self) -> float | None:
        return self.reward_total / self.interactions if self.interactions else None


class SlotPolicyRouter(FeedbackRouter, Generic[ObservationT]):
    """Epsilon-greedy over valid ranked slots in training; greedy and frozen in evaluation."""

    def __init__(
        self,
        candidate_weights: CandidateWeights,
        candidate_limit: int,
        reward_weights: RewardWeights,
        *,
        mode: RouterMode,
        epsilon: float,
        seed: int,
    ) -> None:
        if mode is RouterMode.EVALUATE and epsilon != 0.0:
            raise ValueError("Evaluation must use epsilon = 0")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self._candidate_weights = candidate_weights
        self._candidate_limit = candidate_limit
        self._reward_weights = reward_weights
        self.mode = mode
        self.epsilon = epsilon
        self._rng: np.random.Generator = stream_rng(seed, "slot-agent", self.name)
        self._decision: _Decision[ObservationT] | None = None
        self._pending: _Pending[ObservationT] | None = None
        self.stats = LearningStats()

    @property
    def learning(self) -> bool:
        return self.mode is RouterMode.TRAIN

    # ----------------------------------------------------------------- model hooks

    @abstractmethod
    def observe(
        self, view: NetworkView, current: NodeId, candidates: Sequence[Candidate]
    ) -> ObservationT:
        """Observation of this decision point (computed after candidate ranking)."""

    @abstractmethod
    def q_values(self, observation: ObservationT) -> Sequence[float]:
        """Estimated Q-value per slot (all ``candidate_limit`` slots)."""

    @abstractmethod
    def learn(
        self,
        observation: ObservationT,
        action: int,
        reward: float,
        next_observation: NextObservation[ObservationT] | None,
    ) -> None:
        """Learn from one transition; ``None`` next observation means terminal."""

    # ----------------------------------------------------------------- acting

    def select_next_hop(
        self,
        current_node_id: NodeId,
        sink_id: NodeId,
        visited_node_ids: Set[NodeId],
        network_view: NetworkView,
        packet_context: PacketContext,
    ) -> NodeId | None:
        candidates = rank_candidates(
            network_view,
            current_node_id,
            visited_node_ids,
            self._candidate_weights,
            self._candidate_limit,
        )
        observation = self.observe(network_view, current_node_id, candidates)
        if self._pending is not None and candidates:
            pending = self._pending
            self._pending = None
            self._learn(
                pending.observation,
                pending.action,
                pending.reward,
                NextObservation(observation, len(candidates)),
            )
        if not candidates:
            return None
        action = self._choose(observation, len(candidates))
        self._decision = _Decision(observation, action, candidates[action])
        return candidates[action].node_id

    def _choose(self, observation: ObservationT, valid_count: int) -> int:
        q_values = self.q_values(observation)
        if not self.learning:
            return greedy_slot(q_values, valid_count)
        if self._rng.random() < self.epsilon:
            self.stats.exploratory_actions += 1
            return int(self._rng.integers(valid_count))
        valid = list(q_values[:valid_count])
        best = max(valid)
        ties = [slot for slot, value in enumerate(valid) if value == best]
        return ties[0] if len(ties) == 1 else ties[int(self._rng.integers(len(ties)))]

    # ----------------------------------------------------------------- feedback

    def on_hop_result(
        self, network_view: NetworkView, packet_context: PacketContext, outcome: HopOutcome
    ) -> None:
        decision = self._decision
        self._decision = None
        if decision is None or not self.learning:
            return
        reward = attempt_reward(
            self._reward_weights,
            decision.candidate,
            outcome,
            network_view.reference_energy_j,
            network_view.communication_range_m,
        )
        self.stats.interactions += 1
        self.stats.reward_total += reward
        if outcome.terminal_outcome is not None:
            self._learn(decision.observation, decision.action, reward, None)
        else:
            self._pending = _Pending(decision.observation, decision.action, reward)

    def on_packet_end(self, packet_context: PacketContext, outcome: PacketOutcome) -> None:
        """A packet that continued but then failed before its next attempt gets the penalty."""
        pending = self._pending
        self._pending = None
        self._decision = None
        if pending is None or not self.learning:
            return
        penalty = (
            0.0 if outcome is PacketOutcome.DELIVERED else self._reward_weights.failure_penalty
        )
        self.stats.reward_total -= penalty
        self._learn(pending.observation, pending.action, pending.reward - penalty, None)

    def _learn(
        self,
        observation: ObservationT,
        action: int,
        reward: float,
        next_observation: NextObservation[ObservationT] | None,
    ) -> None:
        if not self.learning:
            return
        self.learn(observation, action, reward, next_observation)
        self.stats.updates += 1
