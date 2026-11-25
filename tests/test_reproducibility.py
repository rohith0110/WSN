from __future__ import annotations

from dataclasses import replace

import pytest

from wsn.config import ConfigError, EvaluationConfig, SimulationConfig, parse_config
from wsn.experiments import LearnedModels, run_comparison, single_variant_plan
from wsn.network import build_scenario
from wsn.random_utils import keyed_uniform
from wsn.rl.trainer import train
from wsn.routing.energy_aware import EnergyAwareRouter
from wsn.routing.q_learning import QPolicy
from wsn.routing.shortest_path import ShortestPathRouter
from wsn.simulator import run_simulation


def test_same_seed_gives_identical_scenario(smoke_config: SimulationConfig) -> None:
    first = build_scenario(smoke_config, 2000)
    second = build_scenario(smoke_config, 2000)
    assert first.topology == second.topology
    for round_number in (1, 2, 50):
        assert first.traffic.packets_for_round(round_number) == second.traffic.packets_for_round(
            round_number
        )
    for a, b in first.topology.edge_lengths_m():
        assert first.links.success_probability(a, b, 30) == second.links.success_probability(
            a, b, 30
        )


def test_link_draws_are_keyed_not_sequential() -> None:
    assert keyed_uniform(1, "hop", 3, 7, 0, 1, 0) == keyed_uniform(1, "hop", 3, 7, 0, 1, 0)
    assert keyed_uniform(1, "hop", 3, 7, 0, 1, 0) != keyed_uniform(1, "hop", 3, 8, 0, 1, 0)


def test_rerun_gives_identical_scientific_outputs(smoke_config: SimulationConfig) -> None:
    for make in (ShortestPathRouter, lambda: EnergyAwareRouter(smoke_config.routing.heuristic)):
        a = run_simulation(build_scenario(smoke_config, 2001), make())
        b = run_simulation(build_scenario(smoke_config, 2001), make())
        assert a.summary.scientific_data() == b.summary.scientific_data()
        assert a.round_metrics == b.round_metrics
        assert a.traces == b.traces


def test_training_is_deterministic(smoke_config: SimulationConfig) -> None:
    first = train(smoke_config, episodes=2)
    second = train(smoke_config, episodes=2)
    assert first.model.fingerprint == second.model.fingerprint
    assert first.history == second.history


def test_training_and_evaluation_seeds_must_be_disjoint(smoke_config: SimulationConfig) -> None:
    with pytest.raises(ConfigError, match="overlap"):
        parse_config(
            {"evaluation": {"seeds": [1000, 2000]}, "q_learning": {"training_seed_start": 1000}}
        )
    leaky = replace(smoke_config, evaluation=EvaluationConfig(seeds=(1001,)))
    with pytest.raises(ValueError, match="overlap"):
        train(leaky, episodes=3)


def test_comparison_never_updates_the_q_table(smoke_config: SimulationConfig) -> None:
    model = train(smoke_config, episodes=2).model
    before = dict(model.policy.table.items())
    plan = single_variant_plan(smoke_config, (2000,), ("shortest_path", "q_learning"))
    runs = run_comparison(plan, LearnedModels(q_table=model))
    assert dict(model.policy.table.items()) == before
    assert {run.result.algorithm for run in runs} == {"shortest_path", "q_learning"}
    offered = {run.result.summary.offered_packets for run in runs}
    assert len(offered) == 1  # paired: same offered load for every algorithm


def test_q_learning_without_model_is_rejected(smoke_config: SimulationConfig) -> None:
    plan = single_variant_plan(smoke_config, (2000,), ("q_learning",))
    with pytest.raises(ConfigError, match="trained Q-table"):
        run_comparison(plan, LearnedModels())
    assert len(QPolicy.empty(smoke_config).table) == 0
