from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsn.config import DQNConfig, SimulationConfig
from wsn.network import NetworkState, NetworkView, build_scenario
from wsn.rl.dqn import (
    MLP,
    AdamState,
    DQNLearner,
    OutputGradient,
    encode_observation,
    input_size,
    masked_max,
)
from wsn.rl.serialization import (
    ModelError,
    check_dqn_compatible,
    dqn_model_to_data,
    load_dqn_model,
    save_dqn_model,
)
from wsn.rl.state import rank_candidates
from wsn.rl.trainer import train_dqn
from wsn.routing.dqn import DQNPolicy, DQNRouter
from wsn.routing.slot_policy import RouterMode
from wsn.simulator import run_simulation


def test_observation_layout_and_mask(smoke_config: SimulationConfig) -> None:
    scenario = build_scenario(smoke_config, 2000)
    state = NetworkState(scenario)
    state.round_number = 1
    view = NetworkView(scenario, state)
    candidates = rank_candidates(view, 0, {0}, smoke_config.routing.candidate, 5)
    features = encode_observation(view, 0, candidates, 5)
    assert features.shape == (input_size(5),)
    mask = features[-5:]
    assert list(mask) == [1.0] * len(candidates) + [0.0] * (5 - len(candidates))
    assert features[0] == 1.0  # full battery


def test_masked_max_ignores_invalid_slots_and_terminal_rows() -> None:
    q = np.array([[1.0, 9.0, 3.0], [4.0, 2.0, 8.0]])
    result = masked_max(q, np.array([1, 0], dtype=np.int64))
    assert list(result) == [1.0, 0.0]
    assert list(masked_max(q, np.array([3, 2], dtype=np.int64))) == [9.0, 4.0]


def test_backprop_matches_finite_differences() -> None:
    rng = np.random.default_rng(0)
    network = MLP.initialise((4, 6, 3), rng)
    inputs = rng.normal(size=(5, 4))
    actions = np.array([0, 2, 1, 1, 0], dtype=np.int64)
    targets = rng.normal(size=5)
    loss_fn = OutputGradient(actions, targets, delta=10.0)
    grads, _ = network.gradients(inputs, loss_fn)

    def loss() -> float:
        probe = OutputGradient(actions, targets, delta=10.0)
        probe(network.forward(inputs))
        return probe.loss

    weight = network.layers[0].weight
    epsilon = 1e-6
    for index in [(0, 0), (2, 3), (3, 5)]:
        original = weight[index]
        weight[index] = original + epsilon
        upper = loss()
        weight[index] = original - epsilon
        lower = loss()
        weight[index] = original
        assert grads[0].weight[index] == pytest.approx((upper - lower) / (2 * epsilon), rel=1e-4)


def test_adam_fits_a_simple_target() -> None:
    rng = np.random.default_rng(1)
    network = MLP.initialise((2, 16, 2), rng)
    inputs = rng.normal(size=(64, 2))
    actions = np.zeros(64, dtype=np.int64)
    targets = inputs[:, 0] * 2.0 - inputs[:, 1]
    optimizer = AdamState(learning_rate=0.01)
    losses = []
    for _ in range(300):
        gradient = OutputGradient(actions, targets, delta=1.0)
        grads, _ = network.gradients(inputs, gradient)
        optimizer.apply(network, grads)
        losses.append(gradient.loss)
    assert losses[-1] < losses[0] * 0.05


def test_learner_trains_after_warmup_and_syncs_target() -> None:
    config = DQNConfig(batch_size=4, warmup_transitions=8, train_every=1, target_sync_steps=3)
    network = MLP.initialise((input_size(2), 8, 2), np.random.default_rng(2))
    learner = DQNLearner(network, config, gamma=0.9, seed=3)
    observation = np.ones(input_size(2))
    for step in range(7):
        learner.record(observation, step % 2, 1.0, observation, 2)
    assert learner.gradient_steps == 0
    for _ in range(6):
        learner.record(observation, 0, -1.0, None, 0)
    assert learner.gradient_steps == 6
    assert np.array_equal(learner.target.layers[0].weight, learner.online.layers[0].weight)


def test_training_mode_requires_learner(smoke_config: SimulationConfig) -> None:
    with pytest.raises(ValueError, match="DQNLearner"):
        DQNRouter(DQNPolicy.initialise(smoke_config), RouterMode.TRAIN, epsilon=0.5)


@pytest.fixture
def trained_dqn(smoke_config: SimulationConfig, tmp_path: Path) -> Path:
    result = train_dqn(smoke_config, episodes=2)
    path = tmp_path / "dqn.json"
    save_dqn_model(result.model, path)
    return path


def test_dqn_roundtrip_and_frozen_evaluation(
    trained_dqn: Path, smoke_config: SimulationConfig
) -> None:
    model = load_dqn_model(trained_dqn)
    weights = [layer.weight.copy() for layer in model.policy.network.layers]
    first = run_simulation(build_scenario(smoke_config, 2000), DQNRouter.frozen(model.policy))
    reloaded = load_dqn_model(trained_dqn)
    second = run_simulation(build_scenario(smoke_config, 2000), DQNRouter.frozen(reloaded.policy))
    assert first.summary.scientific_data() == second.summary.scientific_data()
    assert reloaded.fingerprint == model.fingerprint
    for before, layer in zip(weights, model.policy.network.layers, strict=True):
        assert np.array_equal(before, layer.weight)


def test_dqn_training_is_deterministic(smoke_config: SimulationConfig) -> None:
    first = train_dqn(smoke_config, episodes=2)
    second = train_dqn(smoke_config, episodes=2)
    assert first.model.fingerprint == second.model.fingerprint
    assert first.history == second.history


def test_dqn_model_validation(
    trained_dqn: Path, smoke_config: SimulationConfig, tmp_path: Path
) -> None:
    model = load_dqn_model(trained_dqn)
    with pytest.raises(ModelError, match="were used for training"):
        check_dqn_compatible(model, smoke_config, (1000,))
    data = dqn_model_to_data(model)
    data["format_version"] = 7
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelError, match="format_version"):
        load_dqn_model(bad)
    q_table = tmp_path / "q.json"
    q_table.write_text(json.dumps({"format": "wsn-ranked-slot-q-table"}), encoding="utf-8")
    with pytest.raises(ModelError, match="Not a wsn-ranked-slot-dqn"):
        load_dqn_model(q_table)
