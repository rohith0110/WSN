"""Training on training-only seeds, using the same simulator as evaluation.

One loop serves both learned policies: tabular Q-Learning and the DQN. Each episode is a fresh
topology (one training seed), epsilon decays linearly per episode, and the same policy object
keeps learning across episodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from wsn.config import SimulationConfig
from wsn.network import build_scenario
from wsn.rl.dqn import DQNLearner
from wsn.rl.serialization import DQNModel, QModel, new_dqn_model, new_model
from wsn.routing.dqn import DQNPolicy, DQNRouter
from wsn.routing.q_learning import QLearningRouter, QPolicy
from wsn.routing.slot_policy import RouterMode
from wsn.simulator import run_simulation

EpisodeCallback = Callable[[int, int], None]
"""Called with (completed_episodes, total_episodes)."""


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode: int
    seed: int
    epsilon: float
    rounds: int
    packets_offered: int
    packets_delivered: int
    offered_load_pdr_pct: float
    interactions: int
    mean_reward: float | None
    exploratory_actions: int
    alive_nodes_end: int
    model_size: int
    """Visited Q-table states (tabular) or stored replay transitions (DQN)."""


ModelT = TypeVar("ModelT", QModel, DQNModel)


@dataclass(frozen=True, slots=True)
class TrainingResult(Generic[ModelT]):
    algorithm: str
    model: ModelT
    history: list[EpisodeRecord]


def epsilon_for_episode(episode: int, episodes: int, start: float, end: float) -> float:
    """Linear decay from ``start`` (first episode) to ``end`` (last episode)."""
    if episodes == 1:
        return end
    return start + (end - start) * episode / (episodes - 1)


def _training_seeds(config: SimulationConfig, episodes: int | None) -> tuple[int, ...]:
    seeds = config.q_learning.training_seeds(episodes)
    overlap = set(seeds) & set(config.evaluation.seeds)
    if overlap:
        raise ValueError(f"Training seeds overlap evaluation seeds: {sorted(overlap)}")
    return seeds


class _EpisodeRunner:
    """Runs training episodes for any slot-policy router and records the history."""

    def __init__(
        self, config: SimulationConfig, seeds: tuple[int, ...], progress: EpisodeCallback | None
    ) -> None:
        self.config = config
        self.seeds = seeds
        self.progress = progress
        self.history: list[EpisodeRecord] = []

    def run(
        self,
        make_router: Callable[[int, float], QLearningRouter | DQNRouter],
        model_size: Callable[[], int],
    ) -> None:
        q_config = self.config.q_learning
        for episode, seed in enumerate(self.seeds):
            epsilon = epsilon_for_episode(
                episode, len(self.seeds), q_config.epsilon_start, q_config.epsilon_end
            )
            router = make_router(episode, epsilon)
            result = run_simulation(
                build_scenario(self.config, seed), router, max_rounds=q_config.train_max_rounds
            )
            summary = result.summary
            self.history.append(
                EpisodeRecord(
                    episode=episode + 1,
                    seed=seed,
                    epsilon=epsilon,
                    rounds=summary.rounds_executed,
                    packets_offered=summary.offered_packets,
                    packets_delivered=summary.delivered_packets,
                    offered_load_pdr_pct=summary.offered_load_pdr_pct,
                    interactions=router.stats.interactions,
                    mean_reward=router.stats.mean_reward,
                    exploratory_actions=router.stats.exploratory_actions,
                    alive_nodes_end=summary.final_alive_nodes,
                    model_size=model_size(),
                )
            )
            if self.progress is not None:
                self.progress(episode + 1, len(self.seeds))


def train(
    config: SimulationConfig,
    episodes: int | None = None,
    progress: EpisodeCallback | None = None,
) -> TrainingResult[QModel]:
    """Tabular Q-Learning: one shared ranked-slot Q-table across training topologies."""
    seeds = _training_seeds(config, episodes)
    policy = QPolicy.empty(config)
    agent_seed = config.q_learning.agent_seed
    runner = _EpisodeRunner(config, seeds, progress)
    runner.run(
        lambda episode, epsilon: QLearningRouter(
            policy, RouterMode.TRAIN, epsilon=epsilon, seed=agent_seed + episode
        ),
        lambda: len(policy.table),
    )
    return TrainingResult("q_learning", new_model(policy, config, seeds), runner.history)


def train_dqn(
    config: SimulationConfig,
    episodes: int | None = None,
    progress: EpisodeCallback | None = None,
) -> TrainingResult[DQNModel]:
    """DQN: one network and replay buffer shared across training topologies."""
    seeds = _training_seeds(config, episodes)
    policy = DQNPolicy.initialise(config)
    learner = DQNLearner(policy.network, config.dqn, policy.gamma, config.dqn.seed)
    agent_seed = config.q_learning.agent_seed
    runner = _EpisodeRunner(config, seeds, progress)
    runner.run(
        lambda episode, epsilon: DQNRouter(
            policy, RouterMode.TRAIN, epsilon=epsilon, seed=agent_seed + episode, learner=learner
        ),
        lambda: learner.replay.size,
    )
    return TrainingResult("dqn", new_dqn_model(policy, config, seeds), runner.history)
