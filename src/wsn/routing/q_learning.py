"""Algorithm C: tabular Q-Learning over ranked candidate slots."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from wsn.config import CandidateWeights, RewardWeights, SimulationConfig, StateBins
from wsn.models import NodeId
from wsn.network import NetworkView
from wsn.rl.state import Candidate, State, observe_state
from wsn.routing.slot_policy import (
    NextObservation,
    RouterMode,
    SlotPolicyRouter,
    greedy_slot,
)

__all__ = [
    "QLearningRouter",
    "QPolicy",
    "QTable",
    "RouterMode",
    "greedy_slot",
    "td_target",
    "td_update",
]


class QTable:
    """Q-values per discrete state; unseen states read as zeros and are not stored."""

    def __init__(self, action_count: int, values: dict[State, list[float]] | None = None) -> None:
        if action_count < 1:
            raise ValueError("action_count must be at least 1")
        self.action_count = action_count
        self._values: dict[State, list[float]] = {}
        for state, row in (values or {}).items():
            if len(row) != action_count:
                raise ValueError(f"Q row for {state} has {len(row)} values, need {action_count}")
            self._values[state] = list(row)

    def __len__(self) -> int:
        return len(self._values)

    def items(self) -> Iterator[tuple[State, tuple[float, ...]]]:
        for state, row in self._values.items():
            yield state, tuple(row)

    def values_for(self, state: State) -> tuple[float, ...]:
        row = self._values.get(state)
        return tuple(row) if row is not None else (0.0,) * self.action_count

    def set_value(self, state: State, action: int, value: float) -> None:
        row = self._values.setdefault(state, [0.0] * self.action_count)
        row[action] = value

    def max_valid(self, state: State, valid_count: int) -> float:
        """Max over valid slots only (``valid_count`` must be >= 1)."""
        return max(self.values_for(state)[:valid_count])


def td_target(reward: float, gamma: float, next_max: float | None) -> float:
    """``r + gamma * max_a' Q(s', a')``; a terminal transition (``None``) bootstraps 0."""
    return reward if next_max is None else reward + gamma * next_max


def td_update(current: float, target: float, alpha: float) -> float:
    """``Q <- Q + alpha * (target - Q)``."""
    return current + alpha * (target - current)


@dataclass(slots=True)
class QPolicy:
    """Everything needed to act: the table plus the ranking/state/reward definitions."""

    table: QTable
    candidate_weights: CandidateWeights
    state_bins: StateBins
    reward_weights: RewardWeights
    alpha: float
    gamma: float

    @property
    def candidate_limit(self) -> int:
        return self.table.action_count

    @classmethod
    def empty(cls, config: SimulationConfig) -> QPolicy:
        return cls(
            table=QTable(config.routing.rl_candidate_limit),
            candidate_weights=config.routing.candidate,
            state_bins=config.q_learning.state_bins,
            reward_weights=config.q_learning.reward,
            alpha=config.q_learning.alpha,
            gamma=config.q_learning.gamma,
        )


class QLearningRouter(SlotPolicyRouter[State]):
    """Tabular Q-Learning: one Q-row per discretised local state."""

    name = "q_learning"

    def __init__(
        self, policy: QPolicy, mode: RouterMode, epsilon: float = 0.0, seed: int = 0
    ) -> None:
        super().__init__(
            policy.candidate_weights,
            policy.candidate_limit,
            policy.reward_weights,
            mode=mode,
            epsilon=epsilon,
            seed=seed,
        )
        self.policy = policy

    @classmethod
    def frozen(cls, policy: QPolicy) -> QLearningRouter:
        return cls(policy, RouterMode.EVALUATE)

    def observe(self, view: NetworkView, current: NodeId, candidates: Sequence[Candidate]) -> State:
        return observe_state(view, current, candidates, self.policy.state_bins)

    def q_values(self, observation: State) -> tuple[float, ...]:
        return self.policy.table.values_for(observation)

    def learn(
        self,
        observation: State,
        action: int,
        reward: float,
        next_observation: NextObservation[State] | None,
    ) -> None:
        table = self.policy.table
        next_max = (
            None
            if next_observation is None
            else table.max_valid(next_observation.observation, next_observation.valid_count)
        )
        current = table.values_for(observation)[action]
        target = td_target(reward, self.policy.gamma, next_max)
        table.set_value(observation, action, td_update(current, target, self.policy.alpha))
