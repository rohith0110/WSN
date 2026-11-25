"""Reality check page: the abstract link model next to ns-3's IEEE 802.15.4 measurements."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from wsn.config import ConfigError, RadioConfig, load_config
from wsn.dashboard.charts import Series, ValueFormat, line_chart
from wsn.dashboard.components import Stat, empty_state, page_head, panel, show, stats
from wsn.dashboard.data import Workspace
from wsn.dashboard.theme import ACCENT
from wsn.ns3_validation import (
    PsrSample,
    ValidationError,
    compare,
    fragments_for,
    load_replay,
    load_samples,
    python_reference,
    summarise,
)
from wsn.radio import LinkQualityModel

EXPONENT_COLORS = ("#5FC4AE", "#A9B3BA", "#E59A5B", "#E7718A", "#B08CE8")
REPLAYS = (
    ("replay_isolated.csv", "One packet at a time"),
    ("replay_contention.csv", "All packets competing"),
)


def render(workspace: Workspace) -> None:
    folder = workspace.results / "ns3"
    try:
        samples = load_samples(folder / "ns3_psr.csv")
        config = load_config(workspace.config)
    except (ValidationError, ConfigError):
        page_head(
            "Does the simple radio model hold up?",
            "Once ns-3 measurements exist, this page compares the simulator's radio links "
            "with ns-3, a detailed network simulator.",
        )
        empty_state(
            "No ns-3 measurements yet",
            "Build the two programs in the ns3 folder with ns-3, run them, then analyse "
            "their output.",
            "python -m wsn.cli ns3-validate --csv results/ns3/ns3_psr.csv --output results/ns3",
        )
        return
    summaries = summarise(compare(samples, config))
    closest = min(summaries, key=lambda s: s.mean_abs_gap)
    page_head(
        "Does the simple radio model hold up?",
        f"Within the {config.scenario.communication_range_m:g} m radio range, the simulator's "
        f"link reliability is on average {closest.mean_abs_gap * 100:.0f} percentage points "
        "from ns-3's "
        f"802.15.4 radio when signals fade like in a cluttered outdoor field (path loss "
        f"exponent {closest.exponent:g}).",
        answer_color=ACCENT,
    )
    _link_chart(
        samples,
        config.scenario.packet_bytes,
        config.scenario.communication_range_m,
        config.radio,
    )
    _replay(folder, workspace)
    st.caption(
        "ns-3 measurements: real 802.15.4 frames (2.4 GHz), Rayleigh fading, no retries. "
        "A 512-byte packet needs "
        f"{fragments_for(config.scenario.packet_bytes, samples[0].msdu_bytes)} radio frames, "
        "and all of them must arrive."
    )


def _link_chart(
    samples: list[PsrSample], packet_bytes: int, radio_range: float, radio: RadioConfig
) -> None:
    fragments = fragments_for(packet_bytes, samples[0].msdu_bytes)
    exponents = sorted({s.exponent for s in samples})
    distances = sorted({s.distance_m for s in samples})
    model_xs = tuple(d for d in distances if d <= radio_range)
    series = [
        Series(
            key="model",
            name="This simulator",
            color=ACCENT,
            xs=model_xs,
            ys=tuple(LinkQualityModel.base_probability(d, radio, radio_range) for d in model_xs),
        )
    ]
    for index, exponent in enumerate(exponents):
        subset = sorted((s for s in samples if s.exponent == exponent), key=lambda s: s.distance_m)
        series.append(
            Series(
                key=f"ns3-{exponent:g}",
                name=f"ns-3, path loss {exponent:g}",
                color=EXPONENT_COLORS[index % len(EXPONENT_COLORS)],
                xs=tuple(s.distance_m for s in subset),
                ys=tuple(s.frame_psr**fragments for s in subset),
                emphasis=False,
            )
        )
    show(
        panel(
            "Chance a packet crosses one hop, by distance",
            line_chart(
                series,
                y_label="Success",
                value_format=ValueFormat.PERCENT_SHARE,
                x_label="Distance (m)",
                y_max=1.0,
            ),
            note=(
                "Higher path loss means more clutter between sensors. The simulator only "
                f"links sensors up to {radio_range:g} m apart, so its line stops there."
            ),
        )
    )


def _replay(folder: Path, workspace: Workspace) -> None:
    results = [
        (label, load_replay(folder / file)) for file, label in REPLAYS if (folder / file).is_file()
    ]
    if not results:
        return
    config = load_config(workspace.config)
    rounds = results[0][1].generated // config.scenario.node_count
    reference = python_reference(config, 2000, rounds)
    tiles = [
        Stat(
            "This simulator",
            f"{reference.pdr:.0%}",
            change=f"of {reference.generated:,} packets delivered",
        ),
        *(
            Stat(
                f"ns-3: {label.lower()}",
                f"{result.pdr:.0%}",
                change=f"of {result.generated:,} packets delivered",
            )
            for label, result in results
        ),
    ]
    st.space("small")
    show(
        panel(
            "The same field replayed in ns-3",
            stats(tiles),
            note=(
                f"Field 2000, {rounds} rounds of shortest-path routing, batteries never run out. "
                "When packets compete, they collide and wait for a free channel, which the "
                "simulator does not model, so ns-3 delivers fewer of them than one at a time."
            ),
        )
    )
