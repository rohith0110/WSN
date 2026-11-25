"""Typed, validated simulation configuration loaded from YAML.

Every tunable value lives here; modules receive these dataclasses instead of scattered constants.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from wsn.models import PlainData

ALGORITHMS: tuple[str, ...] = (
    "shortest_path",
    "energy_aware",
    "leach",
    "cluster_adaptive",
    "q_learning",
    "dqn",
)
LEARNED_ALGORITHMS: tuple[str, ...] = ("q_learning", "dqn")
TRAFFIC_PROFILES: tuple[str, ...] = ("uniform", "bursty")


class ConfigError(ValueError):
    """Raised for malformed or out-of-range configuration."""


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    node_count: int = 50
    area_width_m: float = 100.0
    area_height_m: float = 100.0
    sink_x_m: float = 50.0
    sink_y_m: float = 50.0
    communication_range_m: float = 30.0
    initial_energy_j: float = 0.2
    packet_bytes: int = 512
    max_rounds: int = 2000
    max_hops_multiplier: int = 2
    topology_require_connected: bool = True
    topology_max_tries: int = 100
    seed: int = 2000

    @property
    def packet_bits(self) -> int:
        return self.packet_bytes * 8

    @property
    def max_hops(self) -> int:
        return self.max_hops_multiplier * self.node_count

    @property
    def field_diagonal_m(self) -> float:
        return math.hypot(self.area_width_m, self.area_height_m)


@dataclass(frozen=True, slots=True)
class RadioConfig:
    e_elec_j_per_bit: float = 5.0e-8
    epsilon_fs_j_per_bit_m2: float = 1.0e-11
    epsilon_mp_j_per_bit_m4: float = 1.3e-15
    base_link_quality_intercept: float = 0.98
    base_link_quality_distance_coefficient: float = 0.20
    base_link_quality_min: float = 0.60
    base_link_quality_max: float = 0.99
    link_offset_amplitude: float = 0.08
    link_temporal_amplitude: float = 0.05
    link_refresh_rounds: int = 25
    min_success_probability: float = 0.50
    max_success_probability: float = 0.99


@dataclass(frozen=True, slots=True)
class TrafficConfig:
    profile: str = "uniform"
    offered_packets_per_node_per_round: int = 1
    tx_capacity_per_round: int = 12
    link_retries: int = 0
    burst_probability: float = 0.10
    burst_multiplier: int = 3


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """Edge-cost coefficients of the energy-aware heuristic (all multiply nonnegative terms)."""

    base: float = 1.0
    energy_cost: float = 1.0
    link_unreliability: float = 1.2
    depletion: float = 1.5
    utilization: float = 0.5
    sparsity: float = 0.2


@dataclass(frozen=True, slots=True)
class CandidateWeights:
    """Magnitudes of the RL candidate-ranking terms; signs are fixed by the formula."""

    energy: float = 0.30
    link_quality: float = 0.25
    progress: float = 0.35
    utilization: float = 0.20
    density: float = 0.05
    tx_cost: float = 0.10


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    rl_candidate_limit: int = 5
    density_reference: int = 10
    heuristic: HeuristicWeights = field(default_factory=HeuristicWeights)
    candidate: CandidateWeights = field(default_factory=CandidateWeights)


@dataclass(frozen=True, slots=True)
class StateBins:
    """Two ascending boundaries per dimension split it into low / medium / high bins."""

    energy: tuple[float, float] = (0.33, 0.66)
    sink_distance: tuple[float, float] = (0.15, 0.35)
    load: tuple[float, float] = (0.34, 0.67)
    degree: tuple[float, float] = (0.4, 0.8)
    link_quality: tuple[float, float] = (0.75, 0.88)
    neighbor_energy: tuple[float, float] = (0.33, 0.66)


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Magnitudes of the per-attempt reward terms; signs are fixed by the formula."""

    hop_penalty: float = 0.05
    energy_cost: float = 0.15
    utilization: float = 0.30
    depletion: float = 0.40
    progress: float = 0.50
    delivered_bonus: float = 10.0
    failure_penalty: float = 10.0
    exhaustion_penalty: float = 2.0


@dataclass(frozen=True, slots=True)
class QLearningConfig:
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    train_episodes: int = 40
    train_max_rounds: int = 400
    training_seed_start: int = 1000
    agent_seed: int = 7
    state_bins: StateBins = field(default_factory=StateBins)
    reward: RewardWeights = field(default_factory=RewardWeights)

    def training_seeds(self, episodes: int | None = None) -> tuple[int, ...]:
        count = self.train_episodes if episodes is None else episodes
        return tuple(range(self.training_seed_start, self.training_seed_start + count))


@dataclass(frozen=True, slots=True)
class ClusteringConfig:
    """Cluster-head routing (no data aggregation: every reading stays one packet)."""

    rotation_rounds: int = 20
    leach_ch_fraction: float = 0.10
    gateway_penalty: float = 0.5
    min_head_energy_fraction: float = 0.10
    adaptive_energy: float = 0.50
    adaptive_coverage: float = 0.20
    adaptive_sink_proximity: float = 0.20
    adaptive_load: float = 0.10


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Deep Q-network over padded ranked-candidate features (shares q_learning seeds/rewards)."""

    hidden_units: tuple[int, ...] = (64, 64)
    learning_rate: float = 1.0e-3
    batch_size: int = 64
    replay_capacity: int = 20000
    warmup_transitions: int = 1000
    train_every: int = 4
    target_sync_steps: int = 500
    huber_delta: float = 1.0
    seed: int = 11


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = tuple(range(2000, 2010))
    algorithms: tuple[str, ...] = ALGORITHMS


@dataclass(frozen=True, slots=True)
class OutputConfig:
    trace_sample_every: int = 20
    energy_snapshot_every: int = 10


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# --------------------------------------------------------------------------- parsing


def to_plain_data(raw: object) -> PlainData:
    """Validate an arbitrary loaded YAML/JSON value into ``PlainData``."""
    if raw is None or isinstance(raw, bool | int | float | str):
        return raw
    if isinstance(raw, list | tuple):
        return [to_plain_data(item) for item in raw]
    if isinstance(raw, dict):
        result: dict[str, PlainData] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ConfigError(f"Mapping keys must be strings, got {key!r}")
            result[key] = to_plain_data(value)
        return result
    raise ConfigError(f"Unsupported value of type {type(raw).__name__}")


def as_mapping(data: PlainData, where: str) -> dict[str, PlainData]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' must be a mapping")
    return data


class _Section:
    """Reads typed fields from one mapping and rejects keys nobody asked for."""

    def __init__(self, name: str, data: PlainData) -> None:
        self._name = name
        self._data = as_mapping(data, name)
        self._read: set[str] = set()

    def _get(self, key: str) -> PlainData:
        self._read.add(key)
        return self._data.get(key)

    def _where(self, key: str) -> str:
        return f"{self._name}.{key}"

    def number(self, key: str, default: float) -> float:
        value = self._get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{self._where(key)} must be a number, got {value!r}")
        return float(value)

    def integer(self, key: str, default: int) -> int:
        value = self._get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self._where(key)} must be an integer, got {value!r}")
        return value

    def text(self, key: str, default: str) -> str:
        value = self._get(key)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ConfigError(f"{self._where(key)} must be a string, got {value!r}")
        return value

    def flag(self, key: str, default: bool) -> bool:
        value = self._get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ConfigError(f"{self._where(key)} must be true or false, got {value!r}")
        return value

    def _sequence(self, key: str) -> list[PlainData] | None:
        value = self._get(key)
        if value is None:
            return None
        if not isinstance(value, list):
            raise ConfigError(f"{self._where(key)} must be a list, got {value!r}")
        return value

    def integers(self, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
        items = self._sequence(key)
        if items is None:
            return default
        result: list[int] = []
        for item in items:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigError(f"{self._where(key)} must contain integers, got {item!r}")
            result.append(item)
        return tuple(result)

    def texts(self, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        items = self._sequence(key)
        if items is None:
            return default
        result: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise ConfigError(f"{self._where(key)} must contain strings, got {item!r}")
            result.append(item)
        return tuple(result)

    def bounds(self, key: str, default: tuple[float, float]) -> tuple[float, float]:
        items = self._sequence(key)
        if items is None:
            return default
        numbers: list[float] = []
        for item in items:
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise ConfigError(f"{self._where(key)} must contain numbers, got {item!r}")
            numbers.append(float(item))
        if len(numbers) != 2 or not numbers[0] < numbers[1]:
            raise ConfigError(f"{self._where(key)} must be two ascending numbers")
        return numbers[0], numbers[1]

    def section(self, key: str) -> _Section:
        return _Section(self._where(key), self._get(key))

    def finish(self) -> None:
        unknown = sorted(set(self._data) - self._read)
        if unknown:
            raise ConfigError(f"Unknown key(s) in '{self._name}': {', '.join(unknown)}")


def _parse_scenario(s: _Section) -> ScenarioConfig:
    d = ScenarioConfig()
    cfg = ScenarioConfig(
        node_count=s.integer("node_count", d.node_count),
        area_width_m=s.number("area_width_m", d.area_width_m),
        area_height_m=s.number("area_height_m", d.area_height_m),
        sink_x_m=s.number("sink_x_m", d.sink_x_m),
        sink_y_m=s.number("sink_y_m", d.sink_y_m),
        communication_range_m=s.number("communication_range_m", d.communication_range_m),
        initial_energy_j=s.number("initial_energy_j", d.initial_energy_j),
        packet_bytes=s.integer("packet_bytes", d.packet_bytes),
        max_rounds=s.integer("max_rounds", d.max_rounds),
        max_hops_multiplier=s.integer("max_hops_multiplier", d.max_hops_multiplier),
        topology_require_connected=s.flag(
            "topology_require_connected", d.topology_require_connected
        ),
        topology_max_tries=s.integer("topology_max_tries", d.topology_max_tries),
        seed=s.integer("seed", d.seed),
    )
    s.finish()
    return cfg


def _parse_radio(s: _Section) -> RadioConfig:
    d = RadioConfig()
    cfg = RadioConfig(
        e_elec_j_per_bit=s.number("e_elec_j_per_bit", d.e_elec_j_per_bit),
        epsilon_fs_j_per_bit_m2=s.number("epsilon_fs_j_per_bit_m2", d.epsilon_fs_j_per_bit_m2),
        epsilon_mp_j_per_bit_m4=s.number("epsilon_mp_j_per_bit_m4", d.epsilon_mp_j_per_bit_m4),
        base_link_quality_intercept=s.number(
            "base_link_quality_intercept", d.base_link_quality_intercept
        ),
        base_link_quality_distance_coefficient=s.number(
            "base_link_quality_distance_coefficient", d.base_link_quality_distance_coefficient
        ),
        base_link_quality_min=s.number("base_link_quality_min", d.base_link_quality_min),
        base_link_quality_max=s.number("base_link_quality_max", d.base_link_quality_max),
        link_offset_amplitude=s.number("link_offset_amplitude", d.link_offset_amplitude),
        link_temporal_amplitude=s.number("link_temporal_amplitude", d.link_temporal_amplitude),
        link_refresh_rounds=s.integer("link_refresh_rounds", d.link_refresh_rounds),
        min_success_probability=s.number("min_success_probability", d.min_success_probability),
        max_success_probability=s.number("max_success_probability", d.max_success_probability),
    )
    s.finish()
    return cfg


def _parse_traffic(s: _Section) -> TrafficConfig:
    d = TrafficConfig()
    cfg = TrafficConfig(
        profile=s.text("profile", d.profile),
        offered_packets_per_node_per_round=s.integer(
            "offered_packets_per_node_per_round", d.offered_packets_per_node_per_round
        ),
        tx_capacity_per_round=s.integer("tx_capacity_per_round", d.tx_capacity_per_round),
        link_retries=s.integer("link_retries", d.link_retries),
        burst_probability=s.number("burst_probability", d.burst_probability),
        burst_multiplier=s.integer("burst_multiplier", d.burst_multiplier),
    )
    s.finish()
    return cfg


def _parse_heuristic(s: _Section) -> HeuristicWeights:
    d = HeuristicWeights()
    cfg = HeuristicWeights(
        base=s.number("base", d.base),
        energy_cost=s.number("energy_cost", d.energy_cost),
        link_unreliability=s.number("link_unreliability", d.link_unreliability),
        depletion=s.number("depletion", d.depletion),
        utilization=s.number("utilization", d.utilization),
        sparsity=s.number("sparsity", d.sparsity),
    )
    s.finish()
    return cfg


def _parse_candidate(s: _Section) -> CandidateWeights:
    d = CandidateWeights()
    cfg = CandidateWeights(
        energy=s.number("energy", d.energy),
        link_quality=s.number("link_quality", d.link_quality),
        progress=s.number("progress", d.progress),
        utilization=s.number("utilization", d.utilization),
        density=s.number("density", d.density),
        tx_cost=s.number("tx_cost", d.tx_cost),
    )
    s.finish()
    return cfg


def _parse_routing(s: _Section) -> RoutingConfig:
    d = RoutingConfig()
    cfg = RoutingConfig(
        rl_candidate_limit=s.integer("rl_candidate_limit", d.rl_candidate_limit),
        density_reference=s.integer("density_reference", d.density_reference),
        heuristic=_parse_heuristic(s.section("heuristic")),
        candidate=_parse_candidate(s.section("candidate")),
    )
    s.finish()
    return cfg


def _parse_bins(s: _Section) -> StateBins:
    d = StateBins()
    cfg = StateBins(
        energy=s.bounds("energy", d.energy),
        sink_distance=s.bounds("sink_distance", d.sink_distance),
        load=s.bounds("load", d.load),
        degree=s.bounds("degree", d.degree),
        link_quality=s.bounds("link_quality", d.link_quality),
        neighbor_energy=s.bounds("neighbor_energy", d.neighbor_energy),
    )
    s.finish()
    return cfg


def _parse_reward(s: _Section) -> RewardWeights:
    d = RewardWeights()
    cfg = RewardWeights(
        hop_penalty=s.number("hop_penalty", d.hop_penalty),
        energy_cost=s.number("energy_cost", d.energy_cost),
        utilization=s.number("utilization", d.utilization),
        depletion=s.number("depletion", d.depletion),
        progress=s.number("progress", d.progress),
        delivered_bonus=s.number("delivered_bonus", d.delivered_bonus),
        failure_penalty=s.number("failure_penalty", d.failure_penalty),
        exhaustion_penalty=s.number("exhaustion_penalty", d.exhaustion_penalty),
    )
    s.finish()
    return cfg


def _parse_q_learning(s: _Section) -> QLearningConfig:
    d = QLearningConfig()
    cfg = QLearningConfig(
        alpha=s.number("alpha", d.alpha),
        gamma=s.number("gamma", d.gamma),
        epsilon_start=s.number("epsilon_start", d.epsilon_start),
        epsilon_end=s.number("epsilon_end", d.epsilon_end),
        train_episodes=s.integer("train_episodes", d.train_episodes),
        train_max_rounds=s.integer("train_max_rounds", d.train_max_rounds),
        training_seed_start=s.integer("training_seed_start", d.training_seed_start),
        agent_seed=s.integer("agent_seed", d.agent_seed),
        state_bins=_parse_bins(s.section("state_bins")),
        reward=_parse_reward(s.section("reward")),
    )
    s.finish()
    return cfg


def _parse_clustering(s: _Section) -> ClusteringConfig:
    d = ClusteringConfig()
    cfg = ClusteringConfig(
        rotation_rounds=s.integer("rotation_rounds", d.rotation_rounds),
        leach_ch_fraction=s.number("leach_ch_fraction", d.leach_ch_fraction),
        gateway_penalty=s.number("gateway_penalty", d.gateway_penalty),
        min_head_energy_fraction=s.number("min_head_energy_fraction", d.min_head_energy_fraction),
        adaptive_energy=s.number("adaptive_energy", d.adaptive_energy),
        adaptive_coverage=s.number("adaptive_coverage", d.adaptive_coverage),
        adaptive_sink_proximity=s.number("adaptive_sink_proximity", d.adaptive_sink_proximity),
        adaptive_load=s.number("adaptive_load", d.adaptive_load),
    )
    s.finish()
    return cfg


def _parse_dqn(s: _Section) -> DQNConfig:
    d = DQNConfig()
    cfg = DQNConfig(
        hidden_units=s.integers("hidden_units", d.hidden_units),
        learning_rate=s.number("learning_rate", d.learning_rate),
        batch_size=s.integer("batch_size", d.batch_size),
        replay_capacity=s.integer("replay_capacity", d.replay_capacity),
        warmup_transitions=s.integer("warmup_transitions", d.warmup_transitions),
        train_every=s.integer("train_every", d.train_every),
        target_sync_steps=s.integer("target_sync_steps", d.target_sync_steps),
        huber_delta=s.number("huber_delta", d.huber_delta),
        seed=s.integer("seed", d.seed),
    )
    s.finish()
    return cfg


def _parse_evaluation(s: _Section) -> EvaluationConfig:
    d = EvaluationConfig()
    cfg = EvaluationConfig(
        seeds=s.integers("seeds", d.seeds), algorithms=s.texts("algorithms", d.algorithms)
    )
    s.finish()
    return cfg


def _parse_output(s: _Section) -> OutputConfig:
    d = OutputConfig()
    cfg = OutputConfig(
        trace_sample_every=s.integer("trace_sample_every", d.trace_sample_every),
        energy_snapshot_every=s.integer("energy_snapshot_every", d.energy_snapshot_every),
    )
    s.finish()
    return cfg


def parse_config(data: PlainData) -> SimulationConfig:
    """Build and validate a configuration from plain data (missing keys take defaults)."""
    root = _Section("config", data)
    config = SimulationConfig(
        scenario=_parse_scenario(root.section("scenario")),
        radio=_parse_radio(root.section("radio")),
        traffic=_parse_traffic(root.section("traffic")),
        routing=_parse_routing(root.section("routing")),
        q_learning=_parse_q_learning(root.section("q_learning")),
        clustering=_parse_clustering(root.section("clustering")),
        dqn=_parse_dqn(root.section("dqn")),
        evaluation=_parse_evaluation(root.section("evaluation")),
        output=_parse_output(root.section("output")),
    )
    root.finish()
    validate_config(config)
    return config


# --------------------------------------------------------------------------- validation


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate_scenario(c: ScenarioConfig) -> None:
    _require(c.node_count >= 1, "scenario.node_count must be at least 1")
    _require(c.area_width_m > 0 and c.area_height_m > 0, "scenario area must be positive")
    _require(
        0 <= c.sink_x_m <= c.area_width_m and 0 <= c.sink_y_m <= c.area_height_m,
        "scenario sink position must lie inside the area",
    )
    _require(c.communication_range_m > 0, "scenario.communication_range_m must be positive")
    _require(c.initial_energy_j > 0, "scenario.initial_energy_j must be positive")
    _require(c.packet_bytes > 0, "scenario.packet_bytes must be positive")
    _require(c.max_rounds >= 1, "scenario.max_rounds must be at least 1")
    _require(c.max_hops_multiplier >= 1, "scenario.max_hops_multiplier must be at least 1")
    _require(c.topology_max_tries >= 1, "scenario.topology_max_tries must be at least 1")


def _validate_radio(c: RadioConfig) -> None:
    _require(
        c.e_elec_j_per_bit > 0 and c.epsilon_fs_j_per_bit_m2 > 0 and c.epsilon_mp_j_per_bit_m4 > 0,
        "radio energy coefficients must be positive",
    )
    _require(
        0 < c.min_success_probability <= c.max_success_probability <= 1,
        "radio success probabilities need 0 < min <= max <= 1",
    )
    _require(
        0 < c.base_link_quality_min <= c.base_link_quality_max <= 1,
        "radio base link quality bounds need 0 < min <= max <= 1",
    )
    _require(c.base_link_quality_distance_coefficient >= 0, "link distance coefficient >= 0")
    _require(c.link_offset_amplitude >= 0, "radio.link_offset_amplitude must be >= 0")
    _require(c.link_temporal_amplitude >= 0, "radio.link_temporal_amplitude must be >= 0")
    _require(c.link_refresh_rounds >= 1, "radio.link_refresh_rounds must be at least 1")


def _validate_traffic(c: TrafficConfig) -> None:
    _require(
        c.profile in TRAFFIC_PROFILES,
        f"traffic.profile must be one of {', '.join(TRAFFIC_PROFILES)}, got '{c.profile}'",
    )
    _require(c.offered_packets_per_node_per_round >= 1, "traffic offered packets must be >= 1")
    _require(c.tx_capacity_per_round >= 1, "traffic.tx_capacity_per_round must be at least 1")
    _require(c.link_retries == 0, "traffic.link_retries must be 0 (retries are not modelled in v1)")
    _require(0 <= c.burst_probability <= 1, "traffic.burst_probability must be in [0, 1]")
    _require(c.burst_multiplier >= 1, "traffic.burst_multiplier must be at least 1")


def _all_nonnegative(values: tuple[float, ...]) -> bool:
    return all(value >= 0 for value in values)


def _validate_routing(c: RoutingConfig) -> None:
    _require(c.rl_candidate_limit >= 1, "routing.rl_candidate_limit must be at least 1")
    _require(c.density_reference >= 1, "routing.density_reference must be at least 1")
    h = c.heuristic
    _require(h.base > 0, "routing.heuristic.base must be positive (keeps edge costs > 0)")
    _require(
        _all_nonnegative(
            (h.energy_cost, h.link_unreliability, h.depletion, h.utilization, h.sparsity)
        ),
        "routing.heuristic weights must be nonnegative",
    )
    w = c.candidate
    _require(
        _all_nonnegative(
            (w.energy, w.link_quality, w.progress, w.utilization, w.density, w.tx_cost)
        ),
        "routing.candidate weights are magnitudes and must be nonnegative",
    )


def _validate_q_learning(c: QLearningConfig) -> None:
    _require(0 < c.alpha <= 1, "q_learning.alpha must satisfy 0 < alpha <= 1")
    _require(0 < c.gamma < 1, "q_learning.gamma must satisfy 0 < gamma < 1")
    _require(
        0 <= c.epsilon_end <= c.epsilon_start <= 1,
        "q_learning epsilons need 0 <= epsilon_end <= epsilon_start <= 1",
    )
    _require(c.train_episodes >= 1, "q_learning.train_episodes must be at least 1")
    _require(c.train_max_rounds >= 1, "q_learning.train_max_rounds must be at least 1")
    r = c.reward
    _require(
        _all_nonnegative(
            (
                r.hop_penalty,
                r.energy_cost,
                r.utilization,
                r.depletion,
                r.progress,
                r.delivered_bonus,
                r.failure_penalty,
                r.exhaustion_penalty,
            )
        ),
        "q_learning.reward weights are magnitudes and must be nonnegative",
    )


def _validate_clustering(c: ClusteringConfig) -> None:
    _require(c.rotation_rounds >= 1, "clustering.rotation_rounds must be at least 1")
    _require(0 < c.leach_ch_fraction <= 1, "clustering.leach_ch_fraction must be in (0, 1]")
    _require(c.gateway_penalty >= 0, "clustering.gateway_penalty must be >= 0")
    _require(
        0 <= c.min_head_energy_fraction < 1, "clustering.min_head_energy_fraction must be in [0, 1)"
    )
    _require(
        _all_nonnegative(
            (c.adaptive_energy, c.adaptive_coverage, c.adaptive_sink_proximity, c.adaptive_load)
        ),
        "clustering adaptive weights are magnitudes and must be nonnegative",
    )


def _validate_dqn(c: DQNConfig) -> None:
    _require(
        len(c.hidden_units) >= 1 and all(units >= 1 for units in c.hidden_units),
        "dqn.hidden_units must list at least one positive layer size",
    )
    _require(c.learning_rate > 0, "dqn.learning_rate must be positive")
    _require(c.batch_size >= 1, "dqn.batch_size must be at least 1")
    _require(c.replay_capacity >= c.batch_size, "dqn.replay_capacity must be >= batch_size")
    _require(c.warmup_transitions >= c.batch_size, "dqn.warmup_transitions must be >= batch_size")
    _require(c.train_every >= 1, "dqn.train_every must be at least 1")
    _require(c.target_sync_steps >= 1, "dqn.target_sync_steps must be at least 1")
    _require(c.huber_delta > 0, "dqn.huber_delta must be positive")


def _validate_evaluation(config: SimulationConfig) -> None:
    c = config.evaluation
    _require(len(c.seeds) >= 1, "evaluation.seeds must not be empty")
    _require(len(set(c.seeds)) == len(c.seeds), "evaluation.seeds must be unique")
    _require(len(c.algorithms) >= 1, "evaluation.algorithms must not be empty")
    for name in c.algorithms:
        validate_algorithm(name)
    overlap = set(c.seeds) & set(config.q_learning.training_seeds())
    _require(
        not overlap,
        f"evaluation seeds overlap training seeds: {sorted(overlap)}; keep them disjoint",
    )


def validate_algorithm(name: str) -> str:
    if name not in ALGORITHMS:
        raise ConfigError(f"Unknown algorithm '{name}'. Choose one of: {', '.join(ALGORITHMS)}")
    return name


def validate_config(config: SimulationConfig) -> None:
    _validate_scenario(config.scenario)
    _validate_radio(config.radio)
    _validate_traffic(config.traffic)
    _validate_routing(config.routing)
    _validate_q_learning(config.q_learning)
    _validate_clustering(config.clustering)
    _validate_dqn(config.dqn)
    _validate_evaluation(config)
    _require(config.output.trace_sample_every >= 1, "output.trace_sample_every must be >= 1")
    _require(config.output.energy_snapshot_every >= 1, "output.energy_snapshot_every must be >= 1")


# --------------------------------------------------------------------------- I/O helpers


def config_to_data(config: SimulationConfig) -> dict[str, PlainData]:
    """Plain-data form of a config, suitable for YAML/JSON and for ``parse_config``."""
    return as_mapping(to_plain_data(asdict(config)), "config")


def config_fingerprint(config: SimulationConfig) -> str:
    """Stable short hash of every resolved setting."""
    text = json.dumps(config_to_data(config), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deep_merge(base: PlainData, overrides: PlainData) -> PlainData:
    """Recursively overlay ``overrides`` onto ``base`` (mappings merge, other values replace)."""
    if isinstance(base, dict) and isinstance(overrides, dict):
        merged = dict(base)
        for key, value in overrides.items():
            merged[key] = deep_merge(base.get(key), value)
        return merged
    return overrides


def load_yaml(path: Path) -> PlainData:
    if not path.is_file():
        raise ConfigError(f"File not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return to_plain_data(yaml.safe_load(handle))
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error


def load_config(path: Path, overrides: PlainData = None) -> SimulationConfig:
    data = load_yaml(path)
    return parse_config(deep_merge(data, overrides) if overrides is not None else data)


def dump_yaml(data: PlainData) -> str:
    return yaml.safe_dump(data, sort_keys=False)
