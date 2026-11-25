"""Optional extension C: compare the abstract link model with ns-3's IEEE 802.15.4 PHY.

ns-3 is a separate C++ simulator. ``ns3/wsn-link-validation.cc`` measures the packet
success rate (PSR) of 802.15.4 frames against distance; this module reads its CSV, puts the
Python simulator's assumed success probability next to it and summarises the gap.

802.15.4 frames carry at most 127 bytes, so a 512-byte reading needs several frames. The
implied packet success assumes every fragment must arrive and fragments fail independently:
``PSR_packet = PSR_frame ** fragments``.
"""

from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from matplotlib.figure import Figure

from wsn.config import ConfigError, SimulationConfig
from wsn.models import PacketContext
from wsn.network import NetworkState, NetworkView, Scenario, build_scenario
from wsn.radio import LinkQualityModel
from wsn.routing.shortest_path import ShortestPathRouter
from wsn.simulator import run_simulation

FRAME_PAYLOAD_BYTES = 100
CSV_COLUMNS = (
    "exponent",
    "nakagami_m",
    "distance_m",
    "msdu_bytes",
    "tx_power_dbm",
    "rx_sensitivity_dbm",
    "sent",
    "received",
    "psr",
)


class ValidationError(ValueError):
    """The ns-3 output is missing or malformed."""


@dataclass(frozen=True, slots=True)
class PsrSample:
    exponent: float
    nakagami_m: float
    distance_m: float
    msdu_bytes: int
    sent: int
    received: int

    @property
    def frame_psr(self) -> float:
        return self.received / self.sent if self.sent else 0.0


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError) as error:
        raise ValidationError(f"Column '{key}' is missing or not a number") from error


def load_samples(path: Path) -> list[PsrSample]:
    if not path.is_file():
        raise ValidationError(f"ns-3 output not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValidationError(f"ns-3 output lacks columns: {', '.join(missing)}")
        samples = [
            PsrSample(
                exponent=_number(row, "exponent"),
                nakagami_m=_number(row, "nakagami_m"),
                distance_m=_number(row, "distance_m"),
                msdu_bytes=int(_number(row, "msdu_bytes")),
                sent=int(_number(row, "sent")),
                received=int(_number(row, "received")),
            )
            for row in reader
        ]
    if not samples:
        raise ValidationError(f"ns-3 output is empty: {path}")
    return samples


def fragments_for(packet_bytes: int, frame_payload: int) -> int:
    return math.ceil(packet_bytes / frame_payload)


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    exponent: float
    distance_m: float
    ns3_frame_psr: float
    ns3_packet_psr: float
    model_probability: float


def compare(samples: list[PsrSample], config: SimulationConfig) -> list[ComparisonRow]:
    """Pair each ns-3 distance inside radio range with the simulator's base probability."""
    radio_range = config.scenario.communication_range_m
    fragments = fragments_for(config.scenario.packet_bytes, samples[0].msdu_bytes)
    return [
        ComparisonRow(
            exponent=s.exponent,
            distance_m=s.distance_m,
            ns3_frame_psr=s.frame_psr,
            ns3_packet_psr=s.frame_psr**fragments,
            model_probability=LinkQualityModel.base_probability(
                s.distance_m, config.radio, radio_range
            ),
        )
        for s in samples
        if s.distance_m <= radio_range
    ]


@dataclass(frozen=True, slots=True)
class ExponentSummary:
    exponent: float
    mean_abs_gap: float
    mean_gap: float
    psr_at_range: float
    model_at_range: float


def summarise(rows: list[ComparisonRow]) -> list[ExponentSummary]:
    result: list[ExponentSummary] = []
    for exponent in sorted({r.exponent for r in rows}):
        subset = [r for r in rows if r.exponent == exponent]
        gaps = [r.ns3_packet_psr - r.model_probability for r in subset]
        edge = max(subset, key=lambda r: r.distance_m)
        result.append(
            ExponentSummary(
                exponent=exponent,
                mean_abs_gap=sum(abs(g) for g in gaps) / len(gaps),
                mean_gap=sum(gaps) / len(gaps),
                psr_at_range=edge.ns3_packet_psr,
                model_at_range=edge.model_probability,
            )
        )
    return result


def figure(samples: list[PsrSample], config: SimulationConfig) -> Figure:
    fragments = fragments_for(config.scenario.packet_bytes, samples[0].msdu_bytes)
    radio_range = config.scenario.communication_range_m
    fig = Figure(figsize=(8.0, 4.5), layout="constrained")
    axes = fig.add_subplot()
    palette = ("#3a7bd5", "#2a9d8f", "#e76f51", "#8e5ea2", "#6c757d")
    for index, exponent in enumerate(sorted({s.exponent for s in samples})):
        color = palette[index % len(palette)]
        subset = sorted((s for s in samples if s.exponent == exponent), key=lambda s: s.distance_m)
        xs = [s.distance_m for s in subset]
        axes.plot(
            xs,
            [s.frame_psr for s in subset],
            color=color,
            linewidth=1,
            linestyle=":",
            label=f"Exponent {exponent:g}: one frame",
        )
        axes.plot(
            xs,
            [s.frame_psr**fragments for s in subset],
            color=color,
            linewidth=2,
            label=f"Exponent {exponent:g}: {config.scenario.packet_bytes}-byte reading",
        )
    model_x = [d / 2 for d in range(0, int(radio_range * 2) + 1)]
    axes.plot(
        model_x,
        [LinkQualityModel.base_probability(d, config.radio, radio_range) for d in model_x],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Simulator base probability (within range)",
    )
    axes.axvline(radio_range, color="grey", linewidth=1)
    axes.set_xlabel("Distance (m)")
    axes.set_ylabel("Success probability")
    axes.set_ylim(0, 1.02)
    axes.set_title("Link success: abstract model vs ns-3 IEEE 802.15.4 PHY")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7)
    return fig


def report_markdown(
    samples: list[PsrSample], summaries: list[ExponentSummary], config: SimulationConfig
) -> str:
    first = samples[0]
    fragments = fragments_for(config.scenario.packet_bytes, first.msdu_bytes)
    lines = [
        "# ns-3 link validation",
        "",
        f"- ns-3 LR-WPAN, 2.4 GHz O-QPSK error model, {first.sent} frames per distance, "
        f"{first.msdu_bytes}-byte MAC payload, Nakagami m = {first.nakagami_m:g} fading on a "
        "log-distance path loss, no acknowledgements or retries.",
        f"- A {config.scenario.packet_bytes}-byte reading needs {fragments} frames; packet "
        "success assumes every frame must arrive and frames fail independently.",
        f"- Compared over the simulator's radio range (up to "
        f"{config.scenario.communication_range_m:g} m) with its distance-based base "
        "probability (random per-link offsets excluded).",
        "",
        "| Path loss exponent | Mean absolute gap | Mean gap (ns-3 minus model) | "
        "ns-3 at range | Model at range |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {s.exponent:g} | {s.mean_abs_gap:.3f} | {s.mean_gap:+.3f} | {s.psr_at_range:.3f} | "
        f"{s.model_at_range:.3f} |"
        for s in summaries
    ]
    lines.append("")
    return "\n".join(lines)


def run_ns3(binary: Path, output_csv: Path, extra_args: list[str]) -> None:
    """Run a built ``wsn-link-validation`` executable."""
    if not binary.is_file():
        raise ConfigError(f"ns-3 program not found: {binary}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(binary), f"--output={output_csv}", *extra_args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValidationError(
            f"ns-3 exited with code {completed.returncode}: {completed.stderr.strip()[-500:]}"
        )


# --------------------------------------------------------------------------- field replay

FIELD_FORMAT = "# wsn-field v1"


def export_field(scenario: Scenario, path: Path) -> int:
    """Write positions and shortest-path next hops of the initial field; returns routed sensors."""
    state = NetworkState(scenario)
    state.round_number = 1
    view = NetworkView(scenario, state)
    router = ShortestPathRouter()
    topology = scenario.topology
    lines = [FIELD_FORMAT, f"sensors {topology.node_count}"]
    lines += [f"position {node} {x:.4f} {y:.4f}" for node, (x, y) in enumerate(topology.positions)]
    routed = 0
    for node in range(topology.node_count):
        context = PacketContext(packet_id=0, source=node, round_number=1, hops_attempted=0)
        hop = router.select_next_hop(node, topology.sink_id, {node}, view, context)
        routed += hop is not None
        lines.append(f"next {node} {-1 if hop is None else hop}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return routed


@dataclass(frozen=True, slots=True)
class ReplayResult:
    source: str
    generated: int
    delivered: int
    mean_hops: float

    @property
    def pdr(self) -> float:
        return self.delivered / self.generated if self.generated else 0.0


def load_replay(path: Path) -> ReplayResult:
    if not path.is_file():
        raise ValidationError(f"ns-3 replay output not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValidationError(f"Expected one result row in {path}")
    row = rows[0]
    mode = row.get("mode", "")
    return ReplayResult(
        source=f"ns-3, {mode} traffic, exponent {_number(row, 'exponent'):g}",
        generated=int(_number(row, "generated")),
        delivered=int(_number(row, "delivered")),
        mean_hops=_number(row, "mean_hops"),
    )


def python_reference(config: SimulationConfig, seed: int, rounds: int) -> ReplayResult:
    """The same field and routing in this simulator, with batteries and capacity unlimited
    so only the link model can drop readings."""
    unlimited = replace(
        config,
        scenario=replace(config.scenario, initial_energy_j=1.0e6, max_rounds=rounds),
        traffic=replace(config.traffic, tx_capacity_per_round=10**6),
    )
    summary = run_simulation(build_scenario(unlimited, seed), ShortestPathRouter()).summary
    return ReplayResult(
        source="This simulator (abstract link model)",
        generated=summary.offered_packets,
        delivered=summary.delivered_packets,
        mean_hops=summary.mean_delivered_hops or 0.0,
    )


def replay_markdown(results: list[ReplayResult], seed: int, rounds: int) -> str:
    lines = [
        "",
        f"## Field replay: seed {seed}, {rounds} rounds, shortest-path routes",
        "",
        "Batteries and per-round capacity are unlimited in this simulator here, so only link "
        "failures drop readings. In ns-3, *isolated* traffic sends one reading at a time; "
        "*contention* lets all readings of a round compete for the channel (CSMA/CA). "
        "Each ns-3 reading is one 100-byte frame, while this simulator models 512-byte "
        "readings; per the link study above, a 512-byte reading sent as several frames "
        "would succeed less often per hop.",
        "",
        "| Source | Readings | Delivered | Delivery rate | Mean hops |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.source} | {r.generated:,} | {r.delivered:,} | {r.pdr:.1%} | {r.mean_hops:.2f} |"
        for r in results
    ]
    lines.append("")
    return "\n".join(lines)
