from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from wsn.config import CandidateWeights, SimulationConfig, StateBins
from wsn.models import NodeId
from wsn.network import NetworkState, NetworkView, Scenario, build_scenario
from wsn.rl.serialization import (
    ModelError,
    check_compatible,
    load_model,
    model_to_data,
    new_model,
    save_model,
)
from wsn.rl.state import describe_candidate, observe_state, rank_candidates, to_bin
from wsn.rl.trainer import epsilon_for_episode, train
from wsn.routing.q_learning import (
    QLearningRouter,
    QPolicy,
    QTable,
    RouterMode,
    greedy_slot,
    td_target,
    td_update,
)
from wsn.simulator import run_simulation

from .support import tiny_config, tiny_scenario

# A hub (node 0 at 20,0) with five relays around it and a sink at (0,0).
HUB = ((20.0, 0.0), (12.0, 3.0), (12.0, -3.0), (25.0, 5.0), (25.0, -5.0), (20.0, 8.0))
WEIGHTS = CandidateWeights()
BINS = StateBins()


class FeatureView(NetworkView):
    """View whose per-node energy/utilization/density and link quality can be set directly."""

    def __init__(self, scenario: Scenario) -> None:
        state = NetworkState(scenario)
        state.round_number = 1
        super().__init__(scenario, state)
        self.energy: dict[NodeId, float] = {}
        self.load: dict[NodeId, float] = {}
        self.dense: dict[NodeId, float] = {}
        self.link: dict[NodeId, float] = {}

    def energy_fraction(self, node: NodeId) -> float:
        return self.energy.get(node, super().energy_fraction(node))

    def utilization(self, node: NodeId) -> float:
        return self.load.get(node, super().utilization(node))

    def density(self, node: NodeId) -> float:
        return self.dense.get(node, super().density(node))

    def link_success_probability(self, first: NodeId, second: NodeId) -> float:
        return self.link.get(second, 0.9)


def _hub_view() -> FeatureView:
    return FeatureView(tiny_scenario(HUB, tiny_config(), communication_range_m=12.0))


# ---------------------------------------------------------------- candidate ranking


def test_candidates_sorted_by_score_with_stable_ties() -> None:
    view = _hub_view()
    candidates = rank_candidates(view, 0, {0}, WEIGHTS, 5)
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    equal = CandidateWeights(0, 0, 0, 0, 0, 0)
    tied = rank_candidates(view, 0, {0}, equal, 5)
    assert [c.node_id for c in tied] == sorted(c.node_id for c in tied)


def test_candidate_limit_keeps_sink_when_directly_reachable() -> None:
    scenario = tiny_scenario(((5.0, 0.0), (8.0, 2.0), (8.0, -2.0)), tiny_config())
    view = FeatureView(scenario)
    sink = scenario.topology.sink_id
    view.link[sink] = 0.5  # make the sink rank last
    ranked = rank_candidates(view, 0, {0}, CandidateWeights(1, 1, 0, 0, 0, 0), 2)
    assert len(ranked) == 2
    assert ranked[-1].node_id == sink


def test_ranking_responds_to_every_factor() -> None:
    view = _hub_view()

    def score(node: NodeId) -> float:
        return describe_candidate(view, 0, node, WEIGHTS).score

    base = score(1)
    view.energy[1] = 0.2
    assert score(1) < base  # residual energy
    view.energy.pop(1)
    view.link[1] = 0.5
    assert score(1) < base  # link quality
    view.link.pop(1)
    view.load[1] = 1.0
    assert score(1) < base  # traffic / utilization
    view.load.pop(1)
    view.dense[1] = 0.0
    assert score(1) < base  # node density
    view.dense.pop(1)
    view.energy[3] = view.energy_fraction(1)
    view.dense[3] = view.density(1)
    assert score(3) < base  # distance: node 3 moves away from the sink
    assert describe_candidate(view, 0, 3, WEIGHTS).progress_m < 0


def test_state_bins_and_sentinel() -> None:
    assert [to_bin(v, (0.3, 0.6)) for v in (0.0, 0.3, 0.59, 0.6, 1.0)] == [0, 1, 1, 2, 2]
    view = _hub_view()
    assert observe_state(view, 0, (), BINS)[4:] == (3, 3)
    state = observe_state(view, 0, rank_candidates(view, 0, {0}, WEIGHTS, 5), BINS)
    assert all(0 <= part <= 2 for part in state)


# ---------------------------------------------------------------- Q mechanics


def test_q_update_matches_hand_calculation() -> None:
    target = td_target(reward=1.0, gamma=0.95, next_max=2.0)
    assert target == pytest.approx(2.9)
    assert td_update(0.5, target, alpha=0.15) == pytest.approx(0.5 + 0.15 * (2.9 - 0.5))
    assert td_update(0.5, target, alpha=0.15) == pytest.approx(0.86)


def test_terminal_update_has_no_bootstrap() -> None:
    assert td_target(reward=-10.0, gamma=0.95, next_max=None) == -10.0
    assert td_update(0.0, -10.0, alpha=0.15) == pytest.approx(-1.5)


def test_greedy_ignores_invalid_slots_and_breaks_ties_low() -> None:
    assert greedy_slot((0.0, 1.0, 9.0, 9.0, 9.0), valid_count=2) == 1
    assert greedy_slot((2.0, 2.0, 2.0, 0.0, 0.0), valid_count=3) == 0
    table = QTable(5, {(0, 0, 0, 0, 0, 0): [1.0, 1.0, 50.0, 0.0, 0.0]})
    assert table.max_valid((0, 0, 0, 0, 0, 0), 2) == 1.0
    assert table.values_for((1, 1, 1, 1, 1, 1)) == (0.0,) * 5
    assert len(table) == 1


def test_evaluation_mode_never_updates(smoke_config: SimulationConfig) -> None:
    policy = QPolicy.empty(smoke_config)
    policy.table.set_value((0, 0, 0, 0, 0, 0), 1, 3.0)
    before = dict(policy.table.items())
    router = QLearningRouter.frozen(policy)
    run_simulation(build_scenario(smoke_config, 2000), router)
    assert dict(policy.table.items()) == before
    assert router.stats.updates == 0
    with pytest.raises(ValueError, match="epsilon = 0"):
        QLearningRouter(policy, RouterMode.EVALUATE, epsilon=0.1)


def test_training_mode_updates_q_values(smoke_config: SimulationConfig) -> None:
    policy = QPolicy.empty(smoke_config)
    router = QLearningRouter(policy, RouterMode.TRAIN, epsilon=0.5, seed=3)
    run_simulation(build_scenario(smoke_config, 1000), router, max_rounds=5)
    assert len(policy.table) > 0
    assert router.stats.updates > 0
    assert router.stats.updates <= router.stats.interactions + 25 * 5


def test_epsilon_decays_linearly() -> None:
    assert epsilon_for_episode(0, 5, 1.0, 0.0) == 1.0
    assert epsilon_for_episode(4, 5, 1.0, 0.0) == 0.0
    assert epsilon_for_episode(2, 5, 1.0, 0.0) == pytest.approx(0.5)
    assert epsilon_for_episode(0, 1, 1.0, 0.05) == 0.05


# ---------------------------------------------------------------- persistence


@pytest.fixture
def trained(smoke_config: SimulationConfig, tmp_path: Path) -> Path:
    result = train(smoke_config, episodes=2)
    path = tmp_path / "q.json"
    save_model(result.model, path)
    assert result.history[-1].model_size == len(result.model.policy.table)
    return path


def test_model_roundtrip_reproduces_greedy_actions(
    trained: Path, smoke_config: SimulationConfig
) -> None:
    original = load_model(trained)
    resaved = trained.with_name("copy.json")
    save_model(original, resaved)
    reloaded = load_model(resaved)
    assert reloaded.fingerprint == original.fingerprint
    for state, values in original.policy.table.items():
        assert reloaded.policy.table.values_for(state) == values
        for valid in range(1, 6):
            assert greedy_slot(reloaded.policy.table.values_for(state), valid) == greedy_slot(
                values, valid
            )
    a = run_simulation(build_scenario(smoke_config, 2000), QLearningRouter.frozen(original.policy))
    b = run_simulation(build_scenario(smoke_config, 2000), QLearningRouter.frozen(reloaded.policy))
    assert a.summary.scientific_data() == b.summary.scientific_data()


def test_empty_model_loads(smoke_config: SimulationConfig, tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    save_model(new_model(QPolicy.empty(smoke_config), smoke_config, (1000,)), path)
    assert len(load_model(path).policy.table) == 0


def test_incompatible_model_is_rejected(
    trained: Path, smoke_config: SimulationConfig, tmp_path: Path
) -> None:
    model = load_model(trained)
    other_k = replace(smoke_config, routing=replace(smoke_config.routing, rl_candidate_limit=3))
    with pytest.raises(ModelError, match="candidate limit"):
        check_compatible(model, other_k, (2000,))
    with pytest.raises(ModelError, match="were used for training"):
        check_compatible(model, smoke_config, (1000, 2000))
    data = model_to_data(model)
    data["format_version"] = 99
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelError, match="format_version"):
        load_model(bad)
    with pytest.raises(ModelError, match="not found"):
        load_model(tmp_path / "missing.json")
    (tmp_path / "junk.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelError, match="not valid JSON"):
        load_model(tmp_path / "junk.json")
