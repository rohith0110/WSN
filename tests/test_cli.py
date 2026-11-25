from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wsn.cli import EXIT_OK, EXIT_USAGE_ERROR, main
from wsn.config import ALGORITHMS, ConfigError, load_config

from .support import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SMOKE_CONFIG_PATH

SMOKE = str(SMOKE_CONFIG_PATH)
EXPERIMENTS = str(PROJECT_ROOT / "configs" / "experiments.yaml")


def test_shipped_configs_validate() -> None:
    assert main(["validate-config", "--config", str(DEFAULT_CONFIG_PATH)]) == EXIT_OK
    assert main(["validate-config", "--config", SMOKE]) == EXIT_OK


def test_full_smoke_workflow(tmp_path: Path) -> None:
    model = tmp_path / "models" / "q_table.json"
    assert main(["train", "--config", SMOKE, "--episodes", "2", "--output", str(model)]) == EXIT_OK
    assert (model.parent / "training_history.csv").is_file()
    dqn = tmp_path / "models" / "dqn.json"
    assert (
        main(
            [
                "train",
                "--config",
                SMOKE,
                "--algorithm",
                "dqn",
                "--episodes",
                "2",
                "--output",
                str(dqn),
            ]
        )
        == EXIT_OK
    )
    assert (dqn.parent / "dqn_training_history.csv").is_file()
    run_dir = tmp_path / "run"
    code = main(
        [
            "simulate",
            "--config",
            SMOKE,
            "--algorithm",
            "q_learning",
            "--model",
            str(model),
            "--seed",
            "2000",
            "--output",
            str(run_dir),
        ]
    )
    assert code == EXIT_OK
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_fingerprint"] is not None
    comparison = tmp_path / "comparison"
    code = main(
        [
            "compare",
            "--config",
            SMOKE,
            "--model",
            str(model),
            "--dqn-model",
            str(dqn),
            "--scenario",
            "smoke",
            "--experiments",
            EXPERIMENTS,
            "--output",
            str(comparison),
        ]
    )
    assert code == EXIT_OK
    per_run = pd.read_csv(comparison / "per_run.csv")
    assert set(per_run["algorithm"]) == set(ALGORITHMS)
    replayable = comparison / "runs" / "n25_seed2000_dqn"
    assert (replayable / "nodes_initial.csv").is_file()
    assert (replayable / "round_metrics.csv").is_file()
    for plot in ("alive_nodes", "residual_energy", "packet_delivery", "drop_reasons"):
        assert (comparison / "plots" / f"{plot}.png").is_file()
    assert "Smoke-level result" in (comparison / "summary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "argv",
    [
        ["simulate", "--config", SMOKE, "--algorithm", "q_learning"],
        ["simulate", "--config", SMOKE, "--algorithm", "q_learning", "--model", "missing.json"],
        ["compare", "--config", SMOKE, "--seeds", "2000", "--algorithms", "q_learning"],
        ["validate-config", "--config", "does-not-exist.yaml"],
        ["compare", "--config", SMOKE, "--scenario", "nope", "--experiments", EXPERIMENTS],
    ],
)
def test_usage_errors_exit_with_code_2(argv: list[str]) -> None:
    assert main(argv) == EXIT_USAGE_ERROR


def test_unknown_algorithm_is_rejected_by_parser() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["simulate", "--config", SMOKE, "--algorithm", "flooding"])
    assert raised.value.code == EXIT_USAGE_ERROR


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("scenario:\n  node_cnt: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="node_cnt"):
        load_config(bad)
    bad.write_text("q_learning:\n  gamma: 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="gamma"):
        load_config(bad)
