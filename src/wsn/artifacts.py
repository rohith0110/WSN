"""Read saved run, comparison and model artifacts back into typed values (used by the dashboard)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wsn.config import to_plain_data
from wsn.models import SINK_LABEL, NodeId, PlainData
from wsn.topology import Topology, build_topology

RUN_FILES = ("summary.json", "round_metrics.csv", "nodes_initial.csv", "scenario.json")
COMPARISON_FILES = ("per_run.csv", "aggregate.csv", "round_metrics.csv")


class ArtifactError(ValueError):
    """A results folder is missing files or holds data in an unexpected shape."""


def _read_json(path: Path) -> dict[str, PlainData]:
    if not path.is_file():
        raise ArtifactError(f"Missing file: {path}")
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    data = to_plain_data(raw)
    if not isinstance(data, dict):
        raise ArtifactError(f"{path} must contain a JSON object")
    return data


def _section(data: dict[str, PlainData], key: str) -> dict[str, PlainData]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected an object at '{key}'")
    return value


def _int(data: dict[str, PlainData], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"Expected an integer at '{key}'")
    return value


def _float(data: dict[str, PlainData], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"Expected a number at '{key}'")
    return float(value)


def _optional_float(data: dict[str, PlainData], key: str) -> float | None:
    return None if data.get(key) is None else _float(data, key)


def _str(data: dict[str, PlainData], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ArtifactError(f"Expected text at '{key}'")
    return value


def _event_round(events: dict[str, PlainData], key: str) -> int | None:
    event = _section(events, key)
    value = event.get("round")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"Lifetime event '{key}' has a non-integer round")
    return value


@dataclass(frozen=True, slots=True)
class StoredSummary:
    algorithm: str
    seed: int
    node_count: int
    rounds_executed: int
    max_rounds: int
    stop_reason: str
    offered_packets: int
    delivered_packets: int
    offered_load_pdr_pct: float
    operational_pdr_pct: float | None
    mean_delivered_hops: float | None
    final_alive_nodes: int
    remaining_energy_j: float
    initial_energy_j: float
    fnd: int | None
    hnd: int | None
    lnd: int | None
    first_disconnection: int | None
    model_fingerprint: str | None


def parse_summary(data: dict[str, PlainData]) -> StoredSummary:
    packets = _section(data, "packets")
    metrics = _section(data, "metrics")
    energy = _section(data, "energy")
    events = _section(data, "lifetime_events")
    model = data.get("model_fingerprint")
    return StoredSummary(
        algorithm=_str(data, "algorithm"),
        seed=_int(data, "seed"),
        node_count=_int(data, "node_count"),
        rounds_executed=_int(data, "rounds_executed"),
        max_rounds=_int(data, "max_rounds"),
        stop_reason=_str(data, "stop_reason"),
        offered_packets=_int(packets, "offered"),
        delivered_packets=_int(packets, "delivered"),
        offered_load_pdr_pct=_float(metrics, "offered_load_pdr_pct"),
        operational_pdr_pct=_optional_float(metrics, "operational_pdr_pct"),
        mean_delivered_hops=_optional_float(metrics, "mean_delivered_hops"),
        final_alive_nodes=_int(metrics, "final_alive_nodes"),
        remaining_energy_j=_float(energy, "remaining_total_j"),
        initial_energy_j=_float(energy, "initial_total_j"),
        fnd=_event_round(events, "fnd"),
        hnd=_event_round(events, "hnd"),
        lnd=_event_round(events, "lnd"),
        first_disconnection=_event_round(events, "first_disconnection"),
        model_fingerprint=model if isinstance(model, str) else None,
    )


@dataclass(frozen=True, slots=True)
class StoredTrace:
    packet_id: int
    round_number: int
    source: int
    path: tuple[NodeId, ...]
    outcome: str


@dataclass(frozen=True, slots=True)
class StoredRun:
    directory: Path
    summary: StoredSummary
    summary_data: dict[str, PlainData]
    rounds: pd.DataFrame
    snapshots: pd.DataFrame
    topology: Topology
    initial_energy_j: float
    traces: tuple[StoredTrace, ...]

    def energy_at(self, round_number: int) -> list[float]:
        """Recorded per-node energy at a snapshot round (sensor order)."""
        rows = self.snapshots[self.snapshots["round"] == round_number].sort_values("node_id")
        if rows.empty:
            raise ArtifactError(f"No energy snapshot at round {round_number}")
        return [float(value) for value in rows["energy_j"]]

    def snapshot_rounds(self) -> list[int]:
        return sorted(int(value) for value in self.snapshots["round"].unique())


def _node_id(label: str, sink_id: NodeId) -> NodeId:
    return sink_id if label == SINK_LABEL else int(label)


def _load_topology(directory: Path) -> tuple[Topology, float]:
    nodes = pd.read_csv(directory / "nodes_initial.csv", dtype={"node_id": str})
    scenario = _read_json(directory / "scenario.json")
    sensors = nodes[~nodes["is_sink"]]
    sink = nodes[nodes["is_sink"]]
    if len(sink) != 1:
        raise ArtifactError("nodes_initial.csv must contain exactly one sink row")
    ordered = sensors.assign(order=sensors["node_id"].astype(int)).sort_values("order")
    positions = [(float(x), float(y)) for x, y in zip(ordered["x_m"], ordered["y_m"], strict=True)]
    sink_position = (float(sink["x_m"].iloc[0]), float(sink["y_m"].iloc[0]))
    topology = build_topology(positions, sink_position, _float(scenario, "communication_range_m"))
    initial = float(ordered["initial_energy_j"].iloc[0]) if len(ordered) else 0.0
    return topology, initial


def _load_traces(path: Path, sink_id: NodeId) -> tuple[StoredTrace, ...]:
    if not path.is_file():
        return ()
    traces: list[StoredTrace] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw: object = json.loads(line)
        data = to_plain_data(raw)
        if not isinstance(data, dict):
            raise ArtifactError("Each packet trace line must be a JSON object")
        path_raw = data.get("path")
        if not isinstance(path_raw, list):
            raise ArtifactError("Packet trace 'path' must be a list")
        hops = tuple(_node_id(str(node), sink_id) for node in path_raw)
        traces.append(
            StoredTrace(
                _int(data, "packet_id"),
                _int(data, "round"),
                _int(data, "source"),
                hops,
                _str(data, "outcome"),
            )
        )
    return tuple(traces)


def load_run(directory: Path) -> StoredRun:
    missing = [name for name in RUN_FILES if not (directory / name).is_file()]
    if missing:
        raise ArtifactError(f"{directory} is not a run folder (missing {', '.join(missing)})")
    summary_data = _read_json(directory / "summary.json")
    topology, initial = _load_topology(directory)
    snapshots_path = directory / "node_energy_snapshots.csv"
    snapshots = (
        pd.read_csv(snapshots_path)
        if snapshots_path.is_file()
        else pd.DataFrame({"round": [], "node_id": [], "energy_j": []})
    )
    return StoredRun(
        directory=directory,
        summary=parse_summary(summary_data),
        summary_data=summary_data,
        rounds=pd.read_csv(directory / "round_metrics.csv"),
        snapshots=snapshots,
        topology=topology,
        initial_energy_j=initial,
        traces=_load_traces(directory / "sampled_packet_traces.jsonl", topology.sink_id),
    )


def find_folders(root: Path, required: tuple[str, ...]) -> list[Path]:
    """Folders under ``root`` (recursively) that contain every required file, newest first."""
    if not root.is_dir():
        return []
    found = {
        marker.parent
        for marker in root.rglob(required[0])
        if all((marker.parent / name).is_file() for name in required)
    }
    return sorted(found, key=lambda folder: folder.stat().st_mtime, reverse=True)


@dataclass(frozen=True, slots=True)
class StoredComparison:
    directory: Path
    per_run: pd.DataFrame
    aggregate: pd.DataFrame
    paired: pd.DataFrame
    rounds: pd.DataFrame
    summary_markdown: str


def load_comparison(directory: Path) -> StoredComparison:
    missing = [name for name in COMPARISON_FILES if not (directory / name).is_file()]
    if missing:
        raise ArtifactError(f"{directory} is not a comparison (missing {', '.join(missing)})")
    paired_path = directory / "paired_differences.csv"
    summary_path = directory / "summary.md"
    return StoredComparison(
        directory=directory,
        per_run=pd.read_csv(directory / "per_run.csv"),
        aggregate=pd.read_csv(directory / "aggregate.csv"),
        paired=pd.read_csv(paired_path) if paired_path.is_file() else pd.DataFrame(),
        rounds=pd.read_csv(directory / "round_metrics.csv"),
        summary_markdown=summary_path.read_text(encoding="utf-8") if summary_path.is_file() else "",
    )
