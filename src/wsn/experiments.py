"""Paired multi-seed comparison of routing algorithms and its statistical summary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from wsn.config import (
    ConfigError,
    SimulationConfig,
    as_mapping,
    deep_merge,
    load_yaml,
    parse_config,
    validate_algorithm,
)
from wsn.metrics import RunSummary, describe
from wsn.models import PlainData
from wsn.network import build_scenario
from wsn.rl.serialization import DQNModel, QModel, check_compatible, check_dqn_compatible
from wsn.routing.base import Router
from wsn.routing.clustering import AdaptiveClusterRouter, LeachRouter
from wsn.routing.dqn import DQNRouter
from wsn.routing.energy_aware import EnergyAwareRouter
from wsn.routing.q_learning import QLearningRouter
from wsn.routing.shortest_path import ShortestPathRouter
from wsn.simulator import SimulationResult, run_simulation

RunCallback = Callable[[int, int, str], None]
"""Called with (completed_runs, total_runs, label)."""


@dataclass(frozen=True, slots=True)
class LearnedModels:
    """Trained policies available to an evaluation (either may be absent)."""

    q_table: QModel | None = None
    dqn: DQNModel | None = None

    def fingerprint_for(self, algorithm: str) -> str | None:
        if algorithm == "q_learning" and self.q_table is not None:
            return self.q_table.fingerprint
        if algorithm == "dqn" and self.dqn is not None:
            return self.dqn.fingerprint
        return None

    def require(self, algorithms: tuple[str, ...]) -> None:
        if "q_learning" in algorithms and self.q_table is None:
            raise ConfigError("q_learning needs a trained Q-table: pass --model <path>")
        if "dqn" in algorithms and self.dqn is None:
            raise ConfigError("dqn needs a trained network: pass --dqn-model <path>")

    def check(
        self, config: SimulationConfig, seeds: tuple[int, ...], algorithms: tuple[str, ...]
    ) -> None:
        self.require(algorithms)
        if "q_learning" in algorithms and self.q_table is not None:
            check_compatible(self.q_table, config, seeds)
        if "dqn" in algorithms and self.dqn is not None:
            check_dqn_compatible(self.dqn, config, seeds)


def make_router(
    name: str, config: SimulationConfig, models: LearnedModels, scenario_seed: int
) -> Router:
    validate_algorithm(name)
    models.require((name,))
    if name == "shortest_path":
        return ShortestPathRouter()
    if name == "energy_aware":
        return EnergyAwareRouter(config.routing.heuristic)
    if name == "leach":
        return LeachRouter(config.clustering, scenario_seed)
    if name == "cluster_adaptive":
        return AdaptiveClusterRouter(config.clustering, scenario_seed)
    if name == "q_learning" and models.q_table is not None:
        return QLearningRouter.frozen(models.q_table.policy)
    if name == "dqn" and models.dqn is not None:
        return DQNRouter.frozen(models.dqn.policy)
    raise ConfigError(f"No router available for '{name}'")


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    name: str
    config: SimulationConfig


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    scenario_name: str
    description: str
    variants: tuple[ExperimentVariant, ...]
    seeds: tuple[int, ...]
    algorithms: tuple[str, ...]


def single_variant_plan(
    config: SimulationConfig, seeds: tuple[int, ...], algorithms: tuple[str, ...]
) -> ExperimentPlan:
    return ExperimentPlan(
        "custom", "Base configuration", (ExperimentVariant("base", config),), seeds, algorithms
    )


def load_experiment_plan(
    experiments_path: Path,
    scenario_name: str,
    base_config_data: PlainData,
    seeds: tuple[int, ...] | None,
    algorithms: tuple[str, ...] | None,
) -> ExperimentPlan:
    """Build variants by overlaying each variant's overrides on the base config."""
    scenarios = as_mapping(
        as_mapping(load_yaml(experiments_path), "experiments").get("scenarios"), "scenarios"
    )
    if scenario_name not in scenarios:
        raise ConfigError(
            f"Unknown experiment scenario '{scenario_name}'. "
            f"Available: {', '.join(sorted(scenarios))}"
        )
    spec = as_mapping(scenarios[scenario_name], scenario_name)
    raw_variants = spec.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ConfigError(f"Scenario '{scenario_name}' needs a non-empty 'variants' list")
    variants: list[ExperimentVariant] = []
    for raw in raw_variants:
        entry = as_mapping(raw, f"{scenario_name}.variants")
        name = entry.get("name")
        if not isinstance(name, str):
            raise ConfigError(f"Every variant in '{scenario_name}' needs a string 'name'")
        overrides = entry.get("overrides") or {}
        variants.append(
            ExperimentVariant(name, parse_config(deep_merge(base_config_data, overrides)))
        )
    description = spec.get("description")
    base = variants[0].config.evaluation
    return ExperimentPlan(
        scenario_name,
        description if isinstance(description, str) else scenario_name,
        tuple(variants),
        seeds if seeds is not None else _plan_seeds(spec, base.seeds),
        algorithms if algorithms is not None else base.algorithms,
    )


def _plan_seeds(spec: dict[str, PlainData], fallback: tuple[int, ...]) -> tuple[int, ...]:
    raw = spec.get("seeds")
    if raw is None:
        return fallback
    if not isinstance(raw, list) or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in raw
    ):
        raise ConfigError("Scenario 'seeds' must be a list of integers")
    return tuple(seed for seed in raw if isinstance(seed, int))


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    variant: str
    result: SimulationResult


def run_comparison(
    plan: ExperimentPlan, models: LearnedModels, progress: RunCallback | None = None
) -> list[ComparisonRun]:
    """Every algorithm runs on the same scenario object per (variant, seed): paired by design."""
    for algorithm in plan.algorithms:
        validate_algorithm(algorithm)
    for variant in plan.variants:
        models.check(variant.config, plan.seeds, plan.algorithms)
    total = len(plan.variants) * len(plan.seeds) * len(plan.algorithms)
    runs: list[ComparisonRun] = []
    for variant in plan.variants:
        for seed in plan.seeds:
            scenario = build_scenario(variant.config, seed)
            for algorithm in plan.algorithms:
                router = make_router(algorithm, variant.config, models, seed)
                result = run_simulation(
                    scenario, router, model_fingerprint=models.fingerprint_for(algorithm)
                )
                runs.append(ComparisonRun(variant.name, result))
                if progress is not None:
                    progress(len(runs), total, f"{variant.name} seed={seed} {algorithm}")
    return runs


# --------------------------------------------------------------------------- statistics

MetricGetter = Callable[[RunSummary], float | None]

PLAIN_METRICS: tuple[tuple[str, MetricGetter], ...] = (
    ("offered_load_pdr_pct", lambda s: s.offered_load_pdr_pct),
    ("active_source_pdr_pct", lambda s: s.active_source_pdr_pct),
    ("operational_pdr_pct", lambda s: s.operational_pdr_pct),
    ("delivered_packets", lambda s: float(s.delivered_packets)),
    ("energy_used_j", lambda s: s.consumed_energy_total_j),
    ("mean_delivered_hops", lambda s: s.mean_delivered_hops),
    ("throughput_packets_per_round", lambda s: s.throughput_packets_per_round),
    ("rounds_executed", lambda s: float(s.rounds_executed)),
    ("policy_compute_time_s", lambda s: s.policy_compute_time_s),
)

LIFETIME_METRICS: tuple[tuple[str, MetricGetter], ...] = (
    ("fnd_round", lambda s: _as_float(s.fnd.round_number)),
    ("hnd_round", lambda s: _as_float(s.hnd.round_number)),
    ("lnd_round", lambda s: _as_float(s.lnd.round_number)),
    ("first_disconnection_round", lambda s: _as_float(s.first_disconnection.round_number)),
)

PAIRED_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("energy_aware", "shortest_path"),
    ("q_learning", "energy_aware"),
    ("q_learning", "shortest_path"),
    ("dqn", "energy_aware"),
    ("dqn", "q_learning"),
    ("cluster_adaptive", "leach"),
    ("cluster_adaptive", "energy_aware"),
    ("leach", "shortest_path"),
)


def _as_float(value: int | None) -> float | None:
    return None if value is None else float(value)


def _summaries(runs: list[ComparisonRun], variant: str, algorithm: str) -> dict[int, RunSummary]:
    return {
        run.result.summary.seed: run.result.summary
        for run in runs
        if run.variant == variant and run.result.algorithm == algorithm
    }


def aggregate_runs(runs: list[ComparisonRun]) -> list[dict[str, PlainData]]:
    """Mean, std and bootstrap 95% CI per (variant, algorithm, metric).

    Lifetime events with any censored run report the observation rate only: averaging the
    observed subset would be biased, so mean/CI stay null and ``note`` explains why.
    """
    rows: list[dict[str, PlainData]] = []
    variants = list(dict.fromkeys(run.variant for run in runs))
    algorithms = list(dict.fromkeys(run.result.algorithm for run in runs))
    for variant in variants:
        for algorithm in algorithms:
            summaries = list(_summaries(runs, variant, algorithm).values())
            if not summaries:
                continue
            for metric, getter in (*PLAIN_METRICS, *LIFETIME_METRICS):
                rows.append(_metric_row(variant, algorithm, metric, getter, summaries))
    return rows


def _metric_row(
    variant: str,
    algorithm: str,
    metric: str,
    getter: MetricGetter,
    summaries: list[RunSummary],
) -> dict[str, PlainData]:
    raw = [getter(summary) for summary in summaries]
    observed = [value for value in raw if value is not None]
    row: dict[str, PlainData] = {
        "variant": variant,
        "algorithm": algorithm,
        "metric": metric,
        "n_runs": len(raw),
        "n_observed": len(observed),
        "observed_fraction": len(observed) / len(raw),
        "mean": None,
        "std": None,
        "ci95_low": None,
        "ci95_high": None,
        "note": "",
    }
    if observed and len(observed) == len(raw):
        stats = describe(observed)
        row.update(mean=stats.mean, std=stats.std, ci95_low=stats.ci_low, ci95_high=stats.ci_high)
        if len(raw) == 1:
            row["note"] = "single seed: no uncertainty estimate"
    elif observed:
        row["note"] = f"censored in {len(raw) - len(observed)}/{len(raw)} runs; mean withheld"
    else:
        row["note"] = "not reached in any run" if metric.endswith("_round") else "undefined"
    return row


def paired_differences(runs: list[ComparisonRun]) -> list[dict[str, PlainData]]:
    """Per-seed differences (A - B) on the same scenarios, with bootstrap 95% CI."""
    rows: list[dict[str, PlainData]] = []
    variants = list(dict.fromkeys(run.variant for run in runs))
    for variant in variants:
        for first, second in PAIRED_CONTRASTS:
            a = _summaries(runs, variant, first)
            b = _summaries(runs, variant, second)
            seeds = sorted(set(a) & set(b))
            if not seeds:
                continue
            for metric, getter in (*PLAIN_METRICS, *LIFETIME_METRICS):
                pairs = [(getter(a[seed]), getter(b[seed])) for seed in seeds]
                diffs = [x - y for x, y in pairs if x is not None and y is not None]
                row: dict[str, PlainData] = {
                    "variant": variant,
                    "contrast": f"{first} - {second}",
                    "metric": metric,
                    "n_pairs": len(seeds),
                    "n_complete_pairs": len(diffs),
                    "mean_difference": None,
                    "ci95_low": None,
                    "ci95_high": None,
                }
                if diffs and len(diffs) == len(seeds):
                    stats = describe(diffs)
                    row.update(
                        mean_difference=stats.mean, ci95_low=stats.ci_low, ci95_high=stats.ci_high
                    )
                rows.append(row)
    return rows
