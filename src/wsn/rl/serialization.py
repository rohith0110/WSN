"""Portable JSON model artifacts for the ranked-slot Q-Learning and DQN policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wsn.config import (
    CandidateWeights,
    RewardWeights,
    SimulationConfig,
    StateBins,
    as_mapping,
    config_fingerprint,
    to_plain_data,
)
from wsn.models import PlainData
from wsn.rl.dqn import MLP, DenseLayer, FloatArray
from wsn.rl.state import State
from wsn.routing.dqn import DQNPolicy
from wsn.routing.q_learning import QPolicy, QTable

MODEL_FORMAT = "wsn-ranked-slot-q-table"
MODEL_FORMAT_VERSION = 1
STATE_DIMENSIONS = 6


class ModelError(ValueError):
    """Missing, malformed or incompatible model artifact."""


@dataclass(slots=True)
class QModel:
    policy: QPolicy
    config_fingerprint: str
    training_seeds: tuple[int, ...]
    episodes: int
    trained_at_utc: str

    @property
    def fingerprint(self) -> str:
        """Hash of the policy content (table + definitions), independent of timestamps."""
        text = json.dumps(_policy_data(self.policy), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def new_model(policy: QPolicy, config: SimulationConfig, training_seeds: tuple[int, ...]) -> QModel:
    return QModel(
        policy=policy,
        config_fingerprint=config_fingerprint(config),
        training_seeds=training_seeds,
        episodes=len(training_seeds),
        trained_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _policy_data(policy: QPolicy) -> dict[str, PlainData]:
    table: list[PlainData] = [
        {"state": list(state), "q": list(values)} for state, values in sorted(policy.table.items())
    ]
    return {
        "candidate_limit": policy.candidate_limit,
        "alpha": policy.alpha,
        "gamma": policy.gamma,
        "candidate_weights": as_mapping(to_plain_data(asdict(policy.candidate_weights)), "w"),
        "state_bins": as_mapping(to_plain_data(asdict(policy.state_bins)), "bins"),
        "reward_weights": as_mapping(to_plain_data(asdict(policy.reward_weights)), "reward"),
        "q_table": table,
    }


def model_to_data(model: QModel) -> dict[str, PlainData]:
    return {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "policy_status": "trained/frozen",
        "trained_at_utc": model.trained_at_utc,
        "config_fingerprint": model.config_fingerprint,
        "training_seeds": list(model.training_seeds),
        "episodes": model.episodes,
        "model_fingerprint": model.fingerprint,
        "policy": _policy_data(model.policy),
    }


def save_model(model: QModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_to_data(model), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- loading


def _field(data: dict[str, PlainData], key: str) -> PlainData:
    if key not in data:
        raise ModelError(f"Model is missing field '{key}'")
    return data[key]


def _number(data: dict[str, PlainData], key: str) -> float:
    value = _field(data, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ModelError(f"Model field '{key}' must be a number")
    return float(value)


def _integer(data: dict[str, PlainData], key: str) -> int:
    value = _field(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(f"Model field '{key}' must be an integer")
    return value


def _text(data: dict[str, PlainData], key: str) -> str:
    value = _field(data, key)
    if not isinstance(value, str):
        raise ModelError(f"Model field '{key}' must be a string")
    return value


def _mapping(data: dict[str, PlainData], key: str) -> dict[str, PlainData]:
    value = _field(data, key)
    if not isinstance(value, dict):
        raise ModelError(f"Model field '{key}' must be an object")
    return value


def _list(data: dict[str, PlainData], key: str) -> list[PlainData]:
    value = _field(data, key)
    if not isinstance(value, list):
        raise ModelError(f"Model field '{key}' must be a list")
    return value


def _bounds(data: dict[str, PlainData], key: str) -> tuple[float, float]:
    items = _list(data, key)
    numbers = [float(i) for i in items if isinstance(i, int | float) and not isinstance(i, bool)]
    if len(numbers) != 2 or len(items) != 2:
        raise ModelError(f"State bin '{key}' must be two numbers")
    return numbers[0], numbers[1]


def _parse_state(raw: PlainData) -> State:
    if not isinstance(raw, list) or len(raw) != STATE_DIMENSIONS:
        raise ModelError(f"Each Q-table state must be a list of {STATE_DIMENSIONS} integers")
    parts: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ModelError("Q-table state entries must be integers")
        parts.append(item)
    return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])


def _parse_row(raw: PlainData, action_count: int) -> list[float]:
    if not isinstance(raw, list) or len(raw) != action_count:
        raise ModelError(f"Each Q-table row must hold {action_count} numbers")
    row: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ModelError("Q-values must be numbers")
        row.append(float(item))
    return row


def _parse_table(entries: list[PlainData], action_count: int) -> QTable:
    values: dict[State, list[float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ModelError("Q-table entries must be objects with 'state' and 'q'")
        values[_parse_state(_field(entry, "state"))] = _parse_row(_field(entry, "q"), action_count)
    return QTable(action_count, values)


def _parse_candidate_weights(w: dict[str, PlainData]) -> CandidateWeights:
    return CandidateWeights(
        energy=_number(w, "energy"),
        link_quality=_number(w, "link_quality"),
        progress=_number(w, "progress"),
        utilization=_number(w, "utilization"),
        density=_number(w, "density"),
        tx_cost=_number(w, "tx_cost"),
    )


def _parse_reward_weights(r: dict[str, PlainData]) -> RewardWeights:
    return RewardWeights(
        hop_penalty=_number(r, "hop_penalty"),
        energy_cost=_number(r, "energy_cost"),
        utilization=_number(r, "utilization"),
        depletion=_number(r, "depletion"),
        progress=_number(r, "progress"),
        delivered_bonus=_number(r, "delivered_bonus"),
        failure_penalty=_number(r, "failure_penalty"),
        exhaustion_penalty=_number(r, "exhaustion_penalty"),
    )


def _parse_policy(data: dict[str, PlainData]) -> QPolicy:
    limit = _integer(data, "candidate_limit")
    if limit < 1:
        raise ModelError("candidate_limit must be at least 1")
    w = _mapping(data, "candidate_weights")
    b = _mapping(data, "state_bins")
    r = _mapping(data, "reward_weights")
    return QPolicy(
        table=_parse_table(_list(data, "q_table"), limit),
        candidate_weights=_parse_candidate_weights(w),
        state_bins=StateBins(
            energy=_bounds(b, "energy"),
            sink_distance=_bounds(b, "sink_distance"),
            load=_bounds(b, "load"),
            degree=_bounds(b, "degree"),
            link_quality=_bounds(b, "link_quality"),
            neighbor_energy=_bounds(b, "neighbor_energy"),
        ),
        reward_weights=_parse_reward_weights(r),
        alpha=_number(data, "alpha"),
        gamma=_number(data, "gamma"),
    )


def model_from_data(raw: PlainData) -> QModel:
    if not isinstance(raw, dict):
        raise ModelError("Model file must contain a JSON object")
    if raw.get("format") != MODEL_FORMAT:
        raise ModelError(f"Not a {MODEL_FORMAT} model (format={raw.get('format')!r})")
    version = raw.get("format_version")
    if version != MODEL_FORMAT_VERSION:
        raise ModelError(
            f"Unsupported model format_version {version!r}; this build reads "
            f"{MODEL_FORMAT_VERSION}. Retrain with `python -m wsn.cli train`."
        )
    seeds: list[int] = []
    for seed in _list(raw, "training_seeds"):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ModelError("training_seeds must be integers")
        seeds.append(seed)
    return QModel(
        policy=_parse_policy(_mapping(raw, "policy")),
        config_fingerprint=_text(raw, "config_fingerprint"),
        training_seeds=tuple(seeds),
        episodes=_integer(raw, "episodes"),
        trained_at_utc=_text(raw, "trained_at_utc"),
    )


def _read_model_file(path: Path) -> PlainData:
    if not path.is_file():
        raise ModelError(
            f"Model file not found: {path}. Train one first with `python -m wsn.cli train`."
        )
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelError(f"Model file {path} is not valid JSON: {error}") from error
    return to_plain_data(raw)


def load_model(path: Path) -> QModel:
    return model_from_data(_read_model_file(path))


def check_compatible(
    model: QModel, config: SimulationConfig, evaluation_seeds: tuple[int, ...]
) -> None:
    """Reject models whose action/state/ranking definitions differ, or that saw these seeds."""
    policy = model.policy
    problems: list[str] = []
    if policy.candidate_limit != config.routing.rl_candidate_limit:
        problems.append(
            f"candidate limit K: model {policy.candidate_limit}, "
            f"config {config.routing.rl_candidate_limit}"
        )
    if policy.state_bins != config.q_learning.state_bins:
        problems.append("state bin boundaries differ")
    if policy.candidate_weights != config.routing.candidate:
        problems.append("candidate-ranking weights differ")
    if problems:
        raise ModelError("Model is incompatible with this config: " + "; ".join(problems))
    _check_seed_leak(model.training_seeds, evaluation_seeds)


def _check_seed_leak(training_seeds: tuple[int, ...], evaluation_seeds: tuple[int, ...]) -> None:
    leaked = sorted(set(training_seeds) & set(evaluation_seeds))
    if leaked:
        raise ModelError(
            f"Evaluation seeds {leaked} were used for training this model; choose held-out seeds."
        )


# --------------------------------------------------------------------------- DQN

DQN_FORMAT = "wsn-ranked-slot-dqn"
DQN_FORMAT_VERSION = 1


@dataclass(slots=True)
class DQNModel:
    policy: DQNPolicy
    config_fingerprint: str
    training_seeds: tuple[int, ...]
    episodes: int
    trained_at_utc: str

    @property
    def fingerprint(self) -> str:
        text = json.dumps(_dqn_policy_data(self.policy), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def new_dqn_model(
    policy: DQNPolicy, config: SimulationConfig, training_seeds: tuple[int, ...]
) -> DQNModel:
    return DQNModel(
        policy=policy,
        config_fingerprint=config_fingerprint(config),
        training_seeds=training_seeds,
        episodes=len(training_seeds),
        trained_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _vector_data(values: FloatArray) -> list[PlainData]:
    return [float(value) for value in values]


def _dqn_policy_data(policy: DQNPolicy) -> dict[str, PlainData]:
    layers: list[PlainData] = [
        {
            "weight": [_vector_data(row) for row in layer.weight],
            "bias": _vector_data(layer.bias),
        }
        for layer in policy.network.layers
    ]
    return {
        "candidate_limit": policy.candidate_limit,
        "gamma": policy.gamma,
        "candidate_weights": as_mapping(to_plain_data(asdict(policy.candidate_weights)), "w"),
        "reward_weights": as_mapping(to_plain_data(asdict(policy.reward_weights)), "reward"),
        "layers": layers,
    }


def dqn_model_to_data(model: DQNModel) -> dict[str, PlainData]:
    return {
        "format": DQN_FORMAT,
        "format_version": DQN_FORMAT_VERSION,
        "policy_status": "trained/frozen",
        "trained_at_utc": model.trained_at_utc,
        "config_fingerprint": model.config_fingerprint,
        "training_seeds": list(model.training_seeds),
        "episodes": model.episodes,
        "parameters": model.policy.network.parameter_count(),
        "model_fingerprint": model.fingerprint,
        "policy": _dqn_policy_data(model.policy),
    }


def save_dqn_model(model: DQNModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dqn_model_to_data(model)), encoding="utf-8")


def _float_vector(raw: PlainData, where: str) -> list[float]:
    if not isinstance(raw, list):
        raise ModelError(f"{where} must be a list of numbers")
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ModelError(f"{where} must contain only numbers")
        values.append(float(item))
    return values


def _parse_layer(raw: PlainData) -> DenseLayer:
    if not isinstance(raw, dict):
        raise ModelError("Each DQN layer must be an object with 'weight' and 'bias'")
    matrix = [_float_vector(row, "weight row") for row in _list(raw, "weight")]
    if not matrix or len({len(row) for row in matrix}) != 1:
        raise ModelError("DQN layer weights must form a non-empty rectangular matrix")
    bias = _float_vector(_field(raw, "bias"), "bias")
    if len(bias) != len(matrix[0]):
        raise ModelError("DQN layer bias length must match its output size")
    return DenseLayer(np.array(matrix, dtype=np.float64), np.array(bias, dtype=np.float64))


def _integer_list(raw: PlainData, where: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise ModelError(f"{where} must be a list of integers")
    values: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ModelError(f"{where} must contain only integers")
        values.append(item)
    return tuple(values)


def dqn_model_from_data(raw: PlainData) -> DQNModel:
    if not isinstance(raw, dict):
        raise ModelError("Model file must contain a JSON object")
    if raw.get("format") != DQN_FORMAT:
        raise ModelError(f"Not a {DQN_FORMAT} model (format={raw.get('format')!r})")
    if raw.get("format_version") != DQN_FORMAT_VERSION:
        raise ModelError(
            f"Unsupported DQN format_version {raw.get('format_version')!r}; this build reads "
            f"{DQN_FORMAT_VERSION}. Retrain with `python -m wsn.cli train --algorithm dqn`."
        )
    policy = _mapping(raw, "policy")
    try:
        network = MLP([_parse_layer(layer) for layer in _list(policy, "layers")])
    except ValueError as error:
        raise ModelError(f"Invalid DQN network: {error}") from error
    if network.output_size != _integer(policy, "candidate_limit"):
        raise ModelError("DQN output size does not match candidate_limit")
    return DQNModel(
        policy=DQNPolicy(
            network=network,
            candidate_weights=_parse_candidate_weights(_mapping(policy, "candidate_weights")),
            reward_weights=_parse_reward_weights(_mapping(policy, "reward_weights")),
            gamma=_number(policy, "gamma"),
        ),
        config_fingerprint=_text(raw, "config_fingerprint"),
        training_seeds=_integer_list(_field(raw, "training_seeds"), "training_seeds"),
        episodes=_integer(raw, "episodes"),
        trained_at_utc=_text(raw, "trained_at_utc"),
    )


def load_dqn_model(path: Path) -> DQNModel:
    return dqn_model_from_data(_read_model_file(path))


def check_dqn_compatible(
    model: DQNModel, config: SimulationConfig, evaluation_seeds: tuple[int, ...]
) -> None:
    problems: list[str] = []
    if model.policy.candidate_limit != config.routing.rl_candidate_limit:
        problems.append(
            f"candidate limit K: model {model.policy.candidate_limit}, "
            f"config {config.routing.rl_candidate_limit}"
        )
    if model.policy.candidate_weights != config.routing.candidate:
        problems.append("candidate-ranking weights differ")
    if problems:
        raise ModelError("DQN model is incompatible with this config: " + "; ".join(problems))
    _check_seed_leak(model.training_seeds, evaluation_seeds)
