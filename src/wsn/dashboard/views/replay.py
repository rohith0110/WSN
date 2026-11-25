"""Replay page: two strategies, one field, one shared timeline."""

from __future__ import annotations

import streamlit as st

from wsn.artifacts import ArtifactError
from wsn.config import ConfigError, load_config
from wsn.dashboard import data
from wsn.dashboard.catalog import algorithm, variant_name
from wsn.dashboard.components import empty_state, page_head, show
from wsn.dashboard.data import Workspace
from wsn.dashboard.replay import ReplayRun, replay

DEFAULT_PAIR = ("energy_aware", "shortest_path")


def render(workspace: Workspace) -> None:
    page_head(
        "Watch two strategies drain the same field",
        "Both sides start with identical sensors, batteries and traffic. Press play and see "
        "where the batteries run out first.",
    )
    conditions = workspace.conditions()
    if not conditions:
        empty_state(
            "Nothing to replay yet",
            "Replays use the runs saved by a comparison. Run the standard comparison first.",
            "python -m wsn.cli compare --model models/q_table.json --dqn-model models/dqn.json "
            "--scenario standard --output results/comparisons/standard",
        )
        return
    names = {c.folder: c.name for c in conditions}
    folder = (
        st.segmented_control(
            "Test",
            options=list(names),
            format_func=lambda key: names[key],
            default=conditions[0].folder,
            key="replay_condition",
        )
        or conditions[0].folder
    )
    stored = data.comparison(workspace.comparison_path(folder))
    variants = data.variants(stored)
    algorithms = data.algorithms(stored)
    left_default, right_default = (
        a if a in algorithms else algorithms[min(i, len(algorithms) - 1)]
        for i, a in enumerate(DEFAULT_PAIR)
    )
    widths = [1, 1, 1, 1] if len(variants) > 1 else [1, 1, 1]
    picks = st.columns(widths, gap="small", vertical_alignment="bottom")
    left = picks[0].selectbox(
        "Left strategy",
        algorithms,
        index=algorithms.index(left_default),
        format_func=lambda a: algorithm(a).name,
        key="replay_left",
    )
    right = picks[1].selectbox(
        "Right strategy",
        algorithms,
        index=algorithms.index(right_default),
        format_func=lambda a: algorithm(a).name,
        key="replay_right",
    )
    variant = (
        picks[2].selectbox("Setting", variants, format_func=variant_name, key="replay_variant")
        if len(variants) > 1
        else variants[0]
    )
    fields = data.seeds(stored, variant)
    seed = picks[-1].selectbox(
        "Field",
        fields,
        format_func=lambda s: f"Field {s}",
        help="Each field number always places the sensors the same way.",
        key="replay_field",
    )
    comparison_path = workspace.comparison_path(folder)
    try:
        entries = [
            ReplayRun(
                algorithm(name).name,
                algorithm(name).color,
                data.run(data.comparison_run_path(comparison_path, variant, int(seed), name)),
            )
            for name in dict.fromkeys((left, right))
        ]
        area = load_config(entries[0].run.directory / "resolved_config.yaml").scenario
    except (ArtifactError, ConfigError, OSError) as error:
        empty_state(
            "This run has no replay data",
            f"Run the comparison again to save replay data. Missing: {error}",
        )
        return
    show(replay(entries, width=area.area_width_m, height=area.area_height_m))
    first, second = entries[0].run.summary, entries[-1].run.summary
    if len(entries) == 2:
        st.caption(
            f"{_outcome(entries[0].label, first.fnd, first.rounds_executed)} "
            f"{_outcome(entries[1].label, second.fnd, second.rounds_executed)} "
            "Moving dots are real recorded packets; a sample of every run's packets is traced."
        )


def _outcome(label: str, first_death: int | None, cut_off: int) -> str:
    death = (
        f"first sensor died in round {first_death}" if first_death is not None else "no sensor died"
    )
    return f"{label}: {death}, sink unreachable after round {cut_off}."
