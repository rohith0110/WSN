"""Optional extension B: deep Q-network over the same ranked candidate slots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wsn.config import CandidateWeights, RewardWeights, SimulationConfig
from wsn.models import NodeId
from wsn.network import NetworkView
from wsn.rl.dqn import MLP, DQNLearner, FloatArray, encode_observation, input_size
from wsn.rl.state import Candidate
from wsn.routing.slot_policy import NextObservation, RouterMode, SlotPolicyRouter


@dataclass(slots=True)
class DQNPolicy:
    network: MLP
    candidate_weights: CandidateWeights
    reward_weights: RewardWeights
    gamma: float

    @property
    def candidate_limit(self) -> int:
        return self.network.output_size

    @classmethod
    def initialise(cls, config: SimulationConfig) -> DQNPolicy:
        limit = config.routing.rl_candidate_limit
        sizes = (input_size(limit), *config.dqn.hidden_units, limit)
        return cls(
            network=MLP.initialise(sizes, np.random.default_rng(config.dqn.seed)),
            candidate_weights=config.routing.candidate,
            reward_weights=config.q_learning.reward,
            gamma=config.q_learning.gamma,
        )


class DQNRouter(SlotPolicyRouter[FloatArray]):
    """Learns through a ``DQNLearner`` in training; frozen greedy network in evaluation."""

    name = "dqn"

    def __init__(
        self,
        policy: DQNPolicy,
        mode: RouterMode,
        epsilon: float = 0.0,
        seed: int = 0,
        learner: DQNLearner | None = None,
    ) -> None:
        super().__init__(
            policy.candidate_weights,
            policy.candidate_limit,
            policy.reward_weights,
            mode=mode,
            epsilon=epsilon,
            seed=seed,
        )
        if mode is RouterMode.TRAIN and learner is None:
            raise ValueError("Training a DQN router needs a DQNLearner")
        if learner is not None and learner.online is not policy.network:
            raise ValueError("The learner must train the policy's own network")
        self.policy = policy
        self.learner = learner

    @classmethod
    def frozen(cls, policy: DQNPolicy) -> DQNRouter:
        return cls(policy, RouterMode.EVALUATE)

    def observe(
        self, view: NetworkView, current: NodeId, candidates: Sequence[Candidate]
    ) -> FloatArray:
        return encode_observation(view, current, candidates, self.policy.candidate_limit)

    def q_values(self, observation: FloatArray) -> tuple[float, ...]:
        return tuple(float(v) for v in self.policy.network.forward(observation[None, :])[0])

    def learn(
        self,
        observation: FloatArray,
        action: int,
        reward: float,
        next_observation: NextObservation[FloatArray] | None,
    ) -> None:
        if self.learner is None:
            return
        if next_observation is None:
            self.learner.record(observation, action, reward, None, 0)
        else:
            self.learner.record(
                observation,
                action,
                reward,
                next_observation.observation,
                next_observation.valid_count,
            )
