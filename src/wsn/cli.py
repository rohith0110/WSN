"""Command-line interface: ``python -m wsn.cli <command> --help``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from wsn import ns3_validation
from wsn.config import (
    LEARNED_ALGORITHMS,
    ConfigError,
    SimulationConfig,
    config_fingerprint,
    config_to_data,
    dump_yaml,
    load_config,
    load_yaml,
    validate_algorithm,
)
from wsn.experiments import (
    ExperimentPlan,
    LearnedModels,
    load_experiment_plan,
    make_router,
    run_comparison,
    single_variant_plan,
)
from wsn.network import build_scenario
from wsn.reporting import (
    replot_comparison,
    write_comparison_artifacts,
    write_run_artifacts,
    write_training_artifacts,
)
from wsn.rl.serialization import ModelError, load_dqn_model, load_model
from wsn.rl.trainer import train, train_dqn
from wsn.simulator import run_simulation
from wsn.topology import TopologyError

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2

DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_EXPERIMENTS = Path("configs/experiments.yaml")


def _seed_list(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"seeds must be comma-separated integers: {text}"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _algorithm_list(text: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    try:
        return tuple(validate_algorithm(name) for name in names)
    except ConfigError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _print(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- commands


def _cmd_validate(args: argparse.Namespace) -> int:
    config: SimulationConfig = load_config(args.config)
    _print(f"Config OK: {args.config} (fingerprint {config_fingerprint(config)})")
    _print(dump_yaml(config_to_data(config)))
    return EXIT_OK


def _load_models(args: argparse.Namespace, algorithms: tuple[str, ...]) -> LearnedModels:
    """Load only the models the requested algorithms need."""
    q_path: Path | None = args.model
    dqn_path: Path | None = args.dqn_model
    return LearnedModels(
        q_table=load_model(q_path) if q_path is not None and "q_learning" in algorithms else None,
        dqn=load_dqn_model(dqn_path) if dqn_path is not None and "dqn" in algorithms else None,
    )


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    seed: int = args.seed if args.seed is not None else config.scenario.seed
    algorithm: str = args.algorithm
    models = _load_models(args, (algorithm,))
    models.check(config, (seed,), (algorithm,))
    scenario = build_scenario(config, seed)
    router = make_router(algorithm, config, models, seed)
    _print(
        f"Simulating {algorithm} on seed {seed}: {scenario.topology.node_count} nodes, "
        f"up to {args.rounds or config.scenario.max_rounds} rounds"
    )
    result = run_simulation(
        scenario,
        router,
        max_rounds=args.rounds,
        model_fingerprint=models.fingerprint_for(algorithm),
    )
    output: Path = args.output or Path("results/runs") / f"{_timestamp()}_{algorithm}_seed{seed}"
    write_run_artifacts(result, output)
    summary = result.summary
    _print(
        f"Done after {summary.rounds_executed} simulated rounds ({summary.stop_reason}). "
        f"Delivered {summary.delivered_packets}/{summary.offered_packets} offered packets "
        f"(offered-load PDR {summary.offered_load_pdr_pct:.2f}%). "
        f"FND={summary.fnd.round_number or 'not reached'}, "
        f"HND={summary.hnd.round_number or 'not reached'}, "
        f"LND={summary.lnd.round_number or 'not reached'}."
    )
    _print(f"Artifacts: {output}")
    return EXIT_OK


def _cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    episodes: int = args.episodes or config.q_learning.train_episodes
    if episodes < 1:
        raise ConfigError("--episodes must be at least 1")
    seeds = config.q_learning.training_seeds(episodes)
    algorithm: str = args.algorithm
    label = "the DQN" if algorithm == "dqn" else "Q-Learning"
    _print(
        f"Training {label} for {episodes} episodes on training seeds "
        f"{seeds[0]}..{seeds[-1]} (up to {config.q_learning.train_max_rounds} rounds each)"
    )

    def report(done: int, total: int) -> None:
        _print(f"  episode {done}/{total} complete")

    result = (train_dqn if algorithm == "dqn" else train)(config, episodes, report)
    output: Path = args.output or Path("models") / (
        "dqn.json" if algorithm == "dqn" else "q_table.json"
    )
    history_path = write_training_artifacts(result, output)
    last = result.history[-1]
    _print(
        f"Saved model to {output}. "
        f"Final training episode PDR {last.offered_load_pdr_pct:.2f}%. History: {history_path}"
    )
    return EXIT_OK


def _comparison_plan(args: argparse.Namespace, config: SimulationConfig) -> ExperimentPlan:
    seeds: tuple[int, ...] | None = args.seeds
    algorithms: tuple[str, ...] | None = args.algorithms
    if args.scenario is None:
        return single_variant_plan(
            config,
            seeds or config.evaluation.seeds,
            algorithms or config.evaluation.algorithms,
        )
    return load_experiment_plan(
        args.experiments, args.scenario, load_yaml(args.config), seeds, algorithms
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    plan = _comparison_plan(args, config)
    models = _load_models(args, plan.algorithms)
    total = len(plan.variants) * len(plan.seeds) * len(plan.algorithms)
    _print(
        f"Comparing {', '.join(plan.algorithms)} on {len(plan.seeds)} held-out seed(s) "
        f"x {len(plan.variants)} variant(s) = {total} runs"
    )

    def report(done: int, count: int, label: str) -> None:
        _print(f"  [{done}/{count}] {label}")

    runs = run_comparison(plan, models, report)
    output: Path = (
        args.output or Path("results/comparisons") / f"{_timestamp()}_{plan.scenario_name}"
    )
    write_comparison_artifacts(plan, runs, models, output)
    _print(f"Comparison written to {output} (see summary.md and plots/)")
    return EXIT_OK


def _cmd_ns3_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    csv_path: Path = args.csv or output / "ns3_psr.csv"
    if args.binary is not None:
        _print(f"Running ns-3 link measurements with {args.binary} (this takes a few minutes)")
        ns3_validation.run_ns3(args.binary, csv_path, list(args.ns3_args or []))
    samples = ns3_validation.load_samples(csv_path)
    rows = ns3_validation.compare(samples, config)
    summaries = ns3_validation.summarise(rows)
    ns3_validation.figure(samples, config).savefig(output / "link_validation.png", dpi=130)
    report = ns3_validation.report_markdown(samples, summaries, config)
    replays: list[Path] = list(args.replay or [])
    if replays:
        rounds: int = args.rounds
        results = [
            ns3_validation.python_reference(config, args.seed, rounds),
            *(ns3_validation.load_replay(path) for path in replays),
        ]
        report += ns3_validation.replay_markdown(results, args.seed, rounds)
        for result in results:
            _print(f"{result.source}: delivered {result.pdr:.1%} of {result.generated} readings")
    (output / "summary.md").write_text(report, encoding="utf-8")
    for summary in summaries:
        _print(
            f"exponent {summary.exponent:g}: mean |ns-3 - model| = {summary.mean_abs_gap:.3f}, "
            f"at range ns-3 {summary.psr_at_range:.3f} vs model {summary.model_at_range:.3f}"
        )
    _print(f"Validation written to {output}")
    return EXIT_OK


def _cmd_replot(args: argparse.Namespace) -> int:
    folder: Path = args.comparison
    if not (folder / "per_run.csv").is_file():
        raise ConfigError(f"{folder} is not a comparison folder (per_run.csv missing)")
    variants = replot_comparison(folder)
    _print(f"Redrew plots for {len(variants)} variant(s) in {folder / 'plots'}")
    return EXIT_OK


def _cmd_ns3_export(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    routed = ns3_validation.export_field(build_scenario(config, args.seed), args.output)
    _print(
        f"Exported field {args.seed} ({config.scenario.node_count} sensors, {routed} routed) "
        f"to {args.output}"
    )
    return EXIT_OK


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wsn.cli",
        description="WSN energy-aware routing simulator: shortest path, energy-aware heuristic "
        "and tabular Q-Learning.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log topology regeneration")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config", help="parse and print the resolved config")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.set_defaults(handler=_cmd_validate)

    simulate = commands.add_parser("simulate", help="run one algorithm on one seed")
    simulate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    simulate.add_argument("--algorithm", required=True, type=validate_algorithm_arg)
    simulate.add_argument("--seed", type=int, help="scenario seed (default: scenario.seed)")
    simulate.add_argument("--model", type=Path, help="Q-table JSON (required for q_learning)")
    simulate.add_argument("--dqn-model", type=Path, help="DQN JSON (required for dqn)")
    simulate.add_argument("--rounds", type=_positive_int, help="override scenario.max_rounds")
    simulate.add_argument("--output", type=Path, help="output directory")
    simulate.set_defaults(handler=_cmd_simulate)

    train_cmd = commands.add_parser("train", help="train a learned policy on training seeds")
    train_cmd.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_cmd.add_argument(
        "--algorithm", choices=LEARNED_ALGORITHMS, default="q_learning", help="policy to train"
    )
    train_cmd.add_argument("--episodes", type=_positive_int, help="override train_episodes")
    train_cmd.add_argument(
        "--output", type=Path, help="model path (default models/q_table.json or models/dqn.json)"
    )
    train_cmd.set_defaults(handler=_cmd_train)

    compare = commands.add_parser("compare", help="paired evaluation on held-out seeds")
    compare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    compare.add_argument("--model", type=Path, help="Q-table JSON (required for q_learning)")
    compare.add_argument("--dqn-model", type=Path, help="DQN JSON (required for dqn)")
    compare.add_argument("--seeds", type=_seed_list, help="comma-separated evaluation seeds")
    compare.add_argument("--algorithms", type=_algorithm_list, help="comma-separated subset")
    compare.add_argument(
        "--scenario",
        help="named experiment from --experiments (smoke, standard, scalability, stress)",
    )
    compare.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    compare.add_argument("--output", type=Path, help="output directory")
    compare.set_defaults(handler=_cmd_compare)

    ns3 = commands.add_parser(
        "ns3-validate", help="compare the link model with ns-3 802.15.4 measurements"
    )
    ns3.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ns3.add_argument("--binary", type=Path, help="built wsn-link-validation executable to run")
    ns3.add_argument("--csv", type=Path, help="existing ns-3 CSV to analyse instead")
    ns3.add_argument("--output", type=Path, default=Path("results/ns3"))
    ns3.add_argument("--replay", type=Path, nargs="*", help="ns-3 field replay CSV files")
    ns3.add_argument("--seed", type=int, default=2000, help="field seed used for the replay")
    ns3.add_argument("--rounds", type=_positive_int, default=50, help="replay rounds")
    ns3.add_argument("ns3_args", nargs="*", help="extra arguments passed to the ns-3 program")
    ns3.set_defaults(handler=_cmd_ns3_validate)

    replot = commands.add_parser("replot", help="redraw a comparison's plots from its CSVs")
    replot.add_argument("--comparison", type=Path, required=True, help="comparison folder")
    replot.set_defaults(handler=_cmd_replot)

    export = commands.add_parser("ns3-export", help="export a field for ns-3 replay")
    export.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    export.add_argument("--seed", type=int, default=2000)
    export.add_argument("--output", type=Path, default=Path("results/ns3/field_2000.txt"))
    export.set_defaults(handler=_cmd_ns3_export)
    return parser


def validate_algorithm_arg(text: str) -> str:
    try:
        return validate_algorithm(text)
    except ConfigError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text}") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    try:
        code: int = args.handler(args)
    except (ConfigError, ModelError, ns3_validation.ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except TopologyError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    return code


if __name__ == "__main__":
    sys.exit(main())
