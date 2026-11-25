from __future__ import annotations

from pathlib import Path

import pytest

from wsn.config import SimulationConfig
from wsn.network import build_scenario
from wsn.ns3_validation import (
    CSV_COLUMNS,
    FIELD_FORMAT,
    ValidationError,
    compare,
    export_field,
    fragments_for,
    load_replay,
    load_samples,
    python_reference,
    replay_markdown,
    report_markdown,
    summarise,
)
from wsn.radio import LinkQualityModel


def _write_samples(path: Path, rows: list[tuple[float, float, int]]) -> Path:
    lines = [",".join(CSV_COLUMNS)]
    lines += [
        f"{exponent},1,{distance},100,0,-106.58,1000,{received},{received / 1000}"
        for exponent, distance, received in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_fragments_round_up() -> None:
    assert fragments_for(512, 100) == 6
    assert fragments_for(100, 100) == 1


def test_compare_uses_packet_success_and_range(
    smoke_config: SimulationConfig, tmp_path: Path
) -> None:
    path = _write_samples(
        tmp_path / "psr.csv", [(3.0, 10.0, 1000), (3.0, 20.0, 900), (3.0, 90.0, 0)]
    )
    samples = load_samples(path)
    rows = compare(samples, smoke_config)
    assert [r.distance_m for r in rows] == [10.0, 20.0]  # 90 m is beyond the 35 m range
    fragments = fragments_for(smoke_config.scenario.packet_bytes, 100)
    assert rows[1].ns3_packet_psr == pytest.approx(0.9**fragments)
    expected = LinkQualityModel.base_probability(
        20.0, smoke_config.radio, smoke_config.scenario.communication_range_m
    )
    assert rows[1].model_probability == pytest.approx(expected)
    (summary,) = summarise(rows)
    gaps = [r.ns3_packet_psr - r.model_probability for r in rows]
    assert summary.mean_gap == pytest.approx(sum(gaps) / 2)
    assert "| 3 |" in report_markdown(samples, [summary], smoke_config)


def test_malformed_ns3_output_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("distance_m,psr\n10,1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="lacks columns"):
        load_samples(bad)
    with pytest.raises(ValidationError, match="not found"):
        load_samples(tmp_path / "missing.csv")


def test_field_export_lists_every_node_and_route(
    smoke_config: SimulationConfig, tmp_path: Path
) -> None:
    scenario = build_scenario(smoke_config, 2000)
    path = tmp_path / "field.txt"
    routed = export_field(scenario, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == FIELD_FORMAT
    count = scenario.topology.node_count
    assert lines[1] == f"sensors {count}"
    assert sum(line.startswith("position ") for line in lines) == count + 1
    hops = [line.split() for line in lines if line.startswith("next ")]
    assert len(hops) == count
    assert routed == count
    for _, node, hop in hops:
        assert int(hop) in scenario.topology.neighbors[int(node)]


def test_replay_results_and_reference(smoke_config: SimulationConfig, tmp_path: Path) -> None:
    replay = tmp_path / "replay.csv"
    replay.write_text(
        "mode,exponent,nakagami_m,rounds,sensors,generated,delivered,pdr,mean_hops\n"
        "isolated,3,1,4,25,100,90,0.9,1.8\n",
        encoding="utf-8",
    )
    result = load_replay(replay)
    assert result.pdr == pytest.approx(0.9)
    reference = python_reference(smoke_config, 2000, 4)
    assert reference.generated == 4 * smoke_config.scenario.node_count
    assert 0 < reference.delivered <= reference.generated
    table = replay_markdown([reference, result], 2000, 4)
    assert "90.0%" in table
