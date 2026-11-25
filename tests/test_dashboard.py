from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from wsn.artifacts import load_comparison, load_run
from wsn.cli import EXIT_OK, main
from wsn.dashboard.catalog import METRICS
from wsn.dashboard.charts import (
    RankRow,
    Series,
    ValueFormat,
    json_for_script,
    line_chart,
    ranking,
    round_ticks,
)
from wsn.dashboard.components import mount_svgs
from wsn.dashboard.data import Workspace, display_horizon, settled_round
from wsn.dashboard.insights import best, lifetime_headline, paired, scores

from .support import PROJECT_ROOT, SMOKE_CONFIG_PATH

APP = PROJECT_ROOT / "dashboard" / "app.py"
SMOKE = str(SMOKE_CONFIG_PATH)
PAGES = ("results", "replay", "compare", "run", "learning", "validation")


def _project(root: Path) -> Path:
    """A project folder with the dashboard launcher and configs, but no results."""
    (root / "dashboard").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)
    shutil.copy(APP, root / "dashboard" / "app.py")
    shutil.copy(PROJECT_ROOT / "configs" / "default.yaml", root / "configs" / "default.yaml")
    return root


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Results and models produced by the real CLI for every algorithm."""
    root = _project(tmp_path_factory.mktemp("project"))
    models = root / "models"
    for algorithm, file in (("q_learning", "q_table.json"), ("dqn", "dqn.json")):
        code = main(
            [
                "train",
                "--config",
                SMOKE,
                "--algorithm",
                algorithm,
                "--episodes",
                "2",
                "--output",
                str(models / file),
            ]
        )
        assert code == EXIT_OK
    code = main(
        [
            "compare",
            "--config",
            SMOKE,
            "--model",
            str(models / "q_table.json"),
            "--dqn-model",
            str(models / "dqn.json"),
            "--output",
            str(root / "results" / "comparisons" / "smoke"),
        ]
    )
    assert code == EXIT_OK
    return root


def _open(root: str, page: str) -> None:
    from pathlib import Path

    from wsn.dashboard.app import main

    main(Path(root), page)


def _app(root: Path, page: str = "results") -> AppTest:
    app = AppTest.from_function(_open, args=(str(root), page), default_timeout=90)
    app.run()
    return app


# ---------------------------------------------------------------- rendering


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_saved_results(project: Path, page: str) -> None:
    app = _app(project, page)
    assert not app.exception


def test_results_page_states_the_measured_leader(project: Path) -> None:
    app = _app(project)
    stored = load_comparison(project / "results" / "comparisons" / "smoke")
    headline = lifetime_headline(stored.per_run, "base")
    assert headline is not None
    markup = " ".join(str(element.proto) for element in app.get("html"))
    assert headline.sentence.split(" keeps")[0] in markup


@pytest.mark.parametrize("page", PAGES)
def test_empty_workspace_explains_next_step(tmp_path: Path, page: str) -> None:
    app = _app(_project(tmp_path), page)
    assert not app.exception
    markup = " ".join(str(element.proto) for element in app.get("html"))
    assert "rl-empty" in markup or "Not trained" in markup or page == "run"


def test_validation_page_reports_measured_gap(project: Path) -> None:
    ns3 = project / "results" / "ns3"
    ns3.mkdir(parents=True, exist_ok=True)
    header = "exponent,nakagami_m,distance_m,msdu_bytes,tx_power_dbm,rx_sensitivity_dbm,sent,"
    rows = [f"3,1,{d},100,0,-106.58,100,{100 - d},{(100 - d) / 100}" for d in range(2, 40, 2)]
    lines = [header + "received,psr", *rows]
    (ns3 / "ns3_psr.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    app = _app(project, "validation")
    assert not app.exception
    markup = " ".join(str(element.proto) for element in app.get("html"))
    assert "percentage points" in markup


def test_run_page_simulates_and_saves(tmp_path: Path) -> None:
    root = _project(tmp_path)
    app = _app(root, "run")
    app.slider[0].set_value(20)
    app.slider[1].set_value(40)
    app.slider[2].set_value(50)
    app.button[0].click()
    app.run()
    assert not app.exception
    saved = list((root / "results" / "runs").glob("dashboard_*"))
    assert len(saved) == 1
    run = load_run(saved[0])
    assert run.summary.node_count == 20


# ---------------------------------------------------------------- insights


def _per_run(values: dict[str, list[float]], metric: str = "fnd_round") -> pd.DataFrame:
    rows = [
        {"variant": "v", "algorithm": name, "seed": seed, metric: value}
        for name, series in values.items()
        for seed, value in enumerate(series)
    ]
    return pd.DataFrame(rows)


def test_best_respects_direction_and_censoring() -> None:
    fnd = METRICS["fnd_round"]
    frame = _per_run({"a": [10, 20], "b": [30, 40], "c": [90, float("nan")]})
    items = scores(frame, "v", fnd)
    leader = best(items, fnd)
    assert leader is not None
    assert leader.algorithm == "b"  # c is censored in one field, so it cannot lead
    energy = METRICS["energy_used_j"]
    used = _per_run({"a": [5, 6], "b": [3, 4]}, "energy_used_j")
    lowest = best(scores(used, "v", energy), energy)
    assert lowest is not None
    assert lowest.algorithm == "b"


def test_paired_differences_use_same_fields() -> None:
    fnd = METRICS["fnd_round"]
    frame = _per_run({"ref": [10, 20, 30], "x": [15, 25, 35]})
    (diff,) = paired(frame, "v", fnd, "ref")
    assert diff.mean == pytest.approx(5.0)
    assert diff.low == pytest.approx(5.0)
    assert diff.certain


def test_headline_changes_with_the_data() -> None:
    frame = _per_run({"shortest_path": [10, 12], "energy_aware": [30, 30]})
    delivered = _per_run({"shortest_path": [5, 5], "energy_aware": [1, 1]}, "delivered_packets")
    frame["delivered_packets"] = delivered["delivered_packets"]
    headline = lifetime_headline(frame, "v")
    assert headline is not None
    assert headline.leader == "energy_aware"
    assert "Shortest path delivers the most packets" in headline.sentence


# ---------------------------------------------------------------- charts


def test_round_ticks_are_round_numbers() -> None:
    assert round_ticks(0, 50, 5) == [0, 10, 20, 30, 40, 50]
    assert round_ticks(1, 420) == [100, 200, 300, 400]


def test_embedded_json_cannot_contain_markup() -> None:
    encoded = json_for_script({"name": "</script><svg>"})
    assert "<" not in encoded
    assert json.loads(encoded) == {"name": "</script><svg>"}


def test_svgs_are_mounted_by_a_script_that_follows_them() -> None:
    markup = mount_svgs('<div><svg viewBox="0 0 1 1"></svg></div><script>run()</script>')
    assert "<svg" not in markup.split("<script>")[0]
    assert markup.index("rl-svg-slot") < markup.index("<script>(() =>")
    assert markup.index("<script>(() =>") < markup.index("<script>run()")


def test_line_chart_escapes_names_and_draws_each_series() -> None:
    series = [
        Series("a", "<b>A</b>", "#fff", (1.0, 2.0), (3.0, 4.0)),
        Series("b", "B", "#000", (1.0, 2.0), (1.0, 0.0)),
    ]
    markup = line_chart(series, y_label="Alive", value_format=ValueFormat.INTEGER)
    assert "<b>A</b>" not in markup
    assert markup.count('class="rl-series') == 2


def test_ranking_orders_by_direction() -> None:
    rows = [RankRow("low", "#000", 1.0, "1"), RankRow("high", "#000", 9.0, "9")]
    assert ranking(rows, higher_is_better=True).index("high") < ranking(
        rows, higher_is_better=True
    ).index("low")
    assert ranking(rows, higher_is_better=False).index("low") < ranking(
        rows, higher_is_better=False
    ).index("high")


def test_display_horizon_stops_after_curves_settle() -> None:
    table = pd.DataFrame({"a": [50, 40, 30, 30, 30, 30, 30, 30, 30, 30]}, index=range(1, 11))
    assert settled_round(table) == 3
    assert display_horizon(table) == 4


def test_workspace_lists_only_complete_conditions(project: Path, tmp_path: Path) -> None:
    assert [c.folder for c in Workspace(project).conditions()] == ["smoke"]
    assert Workspace(tmp_path).conditions() == []
