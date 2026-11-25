"""Write run, comparison and training artifacts (CSV, JSON, YAML, Markdown, PNG)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from wsn.config import config_fingerprint, config_to_data, dump_yaml
from wsn.experiments import (
    ComparisonRun,
    ExperimentPlan,
    LearnedModels,
    aggregate_runs,
    paired_differences,
)
from wsn.models import SINK_LABEL, NodeId, PlainData
from wsn.network import Scenario
from wsn.rl.serialization import DQNModel, QModel, save_dqn_model, save_model
from wsn.rl.trainer import TrainingResult
from wsn.simulator import SimulationResult
from wsn.visualization import (
    comparison_line_figure,
    drop_reasons_figure,
    topology_figure,
    training_curve_figure,
)

FIGURE_DPI = 130


def _write_json(path: Path, data: PlainData) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI)


def node_label(node: NodeId, sink_id: NodeId) -> str:
    return SINK_LABEL if node == sink_id else str(node)


def scenario_data(scenario: Scenario) -> dict[str, PlainData]:
    topology = scenario.topology
    links: list[PlainData] = [
        {
            "a": node_label(first, topology.sink_id),
            "b": node_label(second, topology.sink_id),
            "distance_m": length,
        }
        for (first, second), length in sorted(topology.edge_lengths_m().items())
    ]
    return {
        "seed": scenario.seed,
        "node_count": topology.node_count,
        "sink_id": SINK_LABEL,
        "sink_internal_index": topology.sink_id,
        "sink_position_m": list(topology.positions[topology.sink_id]),
        "communication_range_m": topology.communication_range_m,
        "topology_generation_attempts": topology.generation_attempts,
        "packet_bits": scenario.packet_bits,
        "reference_energy_j": scenario.reference_energy_j,
        "potential_links": links,
    }


def nodes_frame(scenario: Scenario) -> pd.DataFrame:
    topology = scenario.topology
    initial = scenario.config.scenario.initial_energy_j
    return pd.DataFrame(
        [
            {
                "node_id": node_label(node, topology.sink_id),
                "x_m": position[0],
                "y_m": position[1],
                "initial_energy_j": None if node == topology.sink_id else initial,
                "potential_degree": len(topology.neighbors[node]),
                "is_sink": node == topology.sink_id,
            }
            for node, position in enumerate(topology.positions)
        ]
    )


def round_metrics_frame(results: Sequence[SimulationResult], variant: str = "") -> pd.DataFrame:
    rows = [
        {"variant": variant, **metrics.to_data()}
        for result in results
        for metrics in result.round_metrics
    ]
    return pd.DataFrame(rows)


def snapshots_frame(result: SimulationResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"round": snapshot.round_number, "node_id": node, "energy_j": energy}
            for snapshot in result.energy_snapshots
            for node, energy in enumerate(snapshot.energy_j)
        ]
    )


def traces_jsonl(result: SimulationResult) -> str:
    sink = result.scenario.topology.sink_id
    lines = [
        json.dumps(
            {
                "packet_id": trace.packet_id,
                "round": trace.round_number,
                "source": trace.source,
                "path": [node_label(node, sink) for node in trace.path],
                "outcome": trace.outcome.value,
                "hops_attempted": trace.hops_attempted,
            }
        )
        for trace in result.traces
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_run_data(result: SimulationResult, output_dir: Path) -> Path:
    """Every data file needed to inspect, replay or re-plot one run, without rerunning it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = result.scenario.config
    (output_dir / "resolved_config.yaml").write_text(
        dump_yaml(config_to_data(config)), encoding="utf-8"
    )
    _write_json(output_dir / "scenario.json", scenario_data(result.scenario))
    nodes_frame(result.scenario).to_csv(output_dir / "nodes_initial.csv", index=False)
    round_metrics_frame([result]).drop(columns="variant").to_csv(
        output_dir / "round_metrics.csv", index=False
    )
    snapshots_frame(result).to_csv(output_dir / "node_energy_snapshots.csv", index=False)
    _write_json(output_dir / "summary.json", result.summary.to_data())
    (output_dir / "sampled_packet_traces.jsonl").write_text(traces_jsonl(result), encoding="utf-8")
    return output_dir


def write_run_artifacts(result: SimulationResult, output_dir: Path) -> Path:
    """Run data plus a static topology figure of the final state."""
    write_run_data(result, output_dir)
    config = result.scenario.config
    final = result.energy_snapshots[-1]
    figure = topology_figure(
        result.scenario.topology,
        final.energy_j,
        config.scenario.initial_energy_j,
        title=(
            f"{result.algorithm}, seed {result.scenario.seed}: state at round {final.round_number}"
        ),
    )
    _save_figure(figure, output_dir / "topology.png")
    return output_dir


# --------------------------------------------------------------------------- training


TRAINING_FILE_PREFIX: dict[str, str] = {"q_learning": "", "dqn": "dqn_"}
"""History/curve file prefix per algorithm, so both models can live in one folder."""


def training_history_path(model_path: Path, algorithm: str) -> Path:
    return model_path.parent / f"{TRAINING_FILE_PREFIX[algorithm]}training_history.csv"


def write_training_artifacts(
    result: TrainingResult[QModel] | TrainingResult[DQNModel], model_path: Path
) -> Path:
    model = result.model
    if isinstance(model, QModel):
        save_model(model, model_path)
    else:
        save_dqn_model(model, model_path)
    history_path = training_history_path(model_path, result.algorithm)
    history = training_history_frame(result)
    history.to_csv(history_path, index=False)
    curve = model_path.parent / f"{TRAINING_FILE_PREFIX[result.algorithm]}training_curve.png"
    _save_figure(training_curve_figure(history), curve)
    return history_path


def training_history_frame(
    result: TrainingResult[QModel] | TrainingResult[DQNModel],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": record.episode,
                "seed": record.seed,
                "epsilon": record.epsilon,
                "rounds": record.rounds,
                "packets_offered": record.packets_offered,
                "packets_delivered": record.packets_delivered,
                "offered_load_pdr_pct": record.offered_load_pdr_pct,
                "interactions": record.interactions,
                "mean_reward": record.mean_reward,
                "exploratory_actions": record.exploratory_actions,
                "alive_nodes_end": record.alive_nodes_end,
                "model_size": record.model_size,
            }
            for record in result.history
        ]
    )


# --------------------------------------------------------------------------- comparison


def per_run_frame(runs: Sequence[ComparisonRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"variant": run.variant, **run.result.summary.flat_record()} for run in runs]
    )


def write_comparison_artifacts(
    plan: ExperimentPlan, runs: list[ComparisonRun], models: LearnedModels, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment: dict[str, PlainData] = {
        "scenario": plan.scenario_name,
        "description": plan.description,
        "seeds": list(plan.seeds),
        "algorithms": list(plan.algorithms),
        "q_table_fingerprint": models.q_table.fingerprint if models.q_table else None,
        "dqn_fingerprint": models.dqn.fingerprint if models.dqn else None,
        "q_table_training_seeds": list(models.q_table.training_seeds) if models.q_table else None,
        "dqn_training_seeds": list(models.dqn.training_seeds) if models.dqn else None,
        "variants": {
            variant.name: {
                "config_fingerprint": config_fingerprint(variant.config),
                "config": config_to_data(variant.config),
            }
            for variant in plan.variants
        },
    }
    (output_dir / "experiment_config.yaml").write_text(dump_yaml(experiment), encoding="utf-8")
    per_run = per_run_frame(runs)
    per_run.to_csv(output_dir / "per_run.csv", index=False)
    aggregate = pd.DataFrame(aggregate_runs(runs))
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    paired = pd.DataFrame(paired_differences(runs))
    paired.to_csv(output_dir / "paired_differences.csv", index=False)
    rounds = pd.concat(
        [round_metrics_frame([run.result], run.variant) for run in runs], ignore_index=True
    )
    rounds.to_csv(output_dir / "round_metrics.csv", index=False)
    for run in runs:
        summary = run.result.summary
        write_run_data(
            run.result,
            output_dir / "runs" / f"{run.variant}_seed{summary.seed}_{summary.algorithm}",
        )
    for variant in plan.variants:
        _write_variant_plots(
            variant.name, rounds, per_run, output_dir / "plots", len(plan.variants)
        )
    (output_dir / "summary.md").write_text(
        comparison_markdown(plan, aggregate, paired), encoding="utf-8"
    )
    return output_dir


def _write_variant_plots(
    variant: str, rounds: pd.DataFrame, per_run: pd.DataFrame, plot_dir: Path, variants: int
) -> None:
    prefix = f"{variant}_" if variants > 1 else ""
    subset = rounds[rounds["variant"] == variant]
    runs_subset = per_run[per_run["variant"] == variant]
    plots = (
        ("alive_nodes", "Alive sensor nodes", "Alive nodes vs round"),
        ("mean_remaining_energy_j", "Mean residual energy (J)", "Residual energy vs round"),
        ("cumulative_offered_pdr", "Cumulative offered-load PDR (%)", "Packet delivery vs round"),
        ("delivered_packets_round", "Packets delivered this round", "Delivered packets vs round"),
    )
    names = ("alive_nodes", "residual_energy", "packet_delivery", "delivered_packets")
    for (column, ylabel, title), name in zip(plots, names, strict=True):
        figure = comparison_line_figure(subset, column, ylabel, f"{title} [{variant}]")
        _save_figure(figure, plot_dir / f"{prefix}{name}.png")
    _save_figure(
        drop_reasons_figure(runs_subset, f"Drop reasons [{variant}]"),
        plot_dir / f"{prefix}drop_reasons.png",
    )


def format_cell(value: object) -> str:
    """Render one table cell; ``None``/NaN become ``N/A`` (never 0 or a placeholder)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in frame[list(columns)].itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def comparison_markdown(plan: ExperimentPlan, aggregate: pd.DataFrame, paired: pd.DataFrame) -> str:
    seeds = len(plan.seeds)
    lines = [
        f"# Comparison: {plan.scenario_name}",
        "",
        plan.description,
        "",
        f"- Evaluation seeds ({seeds}): {', '.join(str(s) for s in plan.seeds)}",
        f"- Algorithms: {', '.join(plan.algorithms)}",
        f"- Variants: {', '.join(v.name for v in plan.variants)}",
        "- Statistics: mean, sample std, percentile bootstrap 95% CI of the mean "
        "(5000 resamples, paired by seed for differences).",
        "- Lifetime events (FND/HND/LND, first disconnection) are censored when not reached; "
        "their mean is withheld unless every run observed the event.",
        "- Time is in simulation rounds; delay is hop count; energy in joules.",
    ]
    if seeds < 5:
        lines += [
            "",
            f"> **Smoke-level result ({seeds} seed{'s' if seeds != 1 else ''}).** "
            "Not enough seeds for scientific conclusions.",
        ]
    columns = [
        "algorithm",
        "metric",
        "n_observed",
        "n_runs",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
        "note",
    ]
    for variant in plan.variants:
        subset = aggregate[aggregate["variant"] == variant.name]
        lines += ["", f"## Variant `{variant.name}`", "", _markdown_table(subset, columns)]
        if not paired.empty:
            diffs = paired[paired["variant"] == variant.name]
            lines += [
                "",
                "### Paired differences (same seed)",
                "",
                _markdown_table(
                    diffs,
                    [
                        "contrast",
                        "metric",
                        "n_complete_pairs",
                        "mean_difference",
                        "ci95_low",
                        "ci95_high",
                    ],
                ),
            ]
    lines += ["", "Plots: see `plots/`. Per-run rows: `per_run.csv`.", ""]
    return "\n".join(lines)


def replot_comparison(output_dir: Path) -> list[str]:
    """Redraw a comparison's plots from its saved CSV files (no simulation rerun)."""
    per_run = pd.read_csv(output_dir / "per_run.csv")
    rounds = pd.read_csv(output_dir / "round_metrics.csv")
    variants = list(dict.fromkeys(str(v) for v in per_run["variant"]))
    for variant in variants:
        _write_variant_plots(variant, rounds, per_run, output_dir / "plots", len(variants))
    return variants
