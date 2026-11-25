"""Run page: configure a small field, simulate it, then replay the saved result."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from wsn.artifacts import ArtifactError, StoredRun
from wsn.config import ALGORITHMS, ConfigError, deep_merge, load_config, load_yaml, parse_config
from wsn.dashboard import data
from wsn.dashboard.catalog import algorithm
from wsn.dashboard.components import Stat, battery, empty_state, page_head, show, stats
from wsn.dashboard.data import Workspace
from wsn.dashboard.replay import ReplayRun, replay
from wsn.experiments import LearnedModels, make_router
from wsn.models import PlainData
from wsn.network import build_scenario
from wsn.reporting import write_run_artifacts
from wsn.rl.serialization import ModelError, load_dqn_model, load_model
from wsn.simulator import run_simulation
from wsn.topology import TopologyError

LAST_RUN = "run_last_path"
ROUND_LIMIT = 1000
BATTERIES = (0.05, 0.1, 0.2, 0.5)


def _learned(workspace: Workspace) -> LearnedModels:
    q_path, dqn_path = workspace.models / "q_table.json", workspace.models / "dqn.json"
    try:
        q_table = load_model(q_path) if q_path.is_file() else None
    except ModelError:
        q_table = None
    try:
        dqn = load_dqn_model(dqn_path) if dqn_path.is_file() else None
    except ModelError:
        dqn = None
    return LearnedModels(q_table=q_table, dqn=dqn)


def render(workspace: Workspace) -> None:
    page_head(
        "Run your own field",
        "Choose how many sensors, how much battery and which strategy. The simulation is "
        "saved, then replayed below.",
    )
    models = _learned(workspace)
    available = [
        a
        for a in ALGORITHMS
        if not (a == "q_learning" and models.q_table is None)
        and not (a == "dqn" and models.dqn is None)
    ]
    form_col, result_col = st.columns([1, 2.2], gap="large")
    with form_col, st.form("run_form", border=True):
        choice = st.selectbox(
            "Strategy",
            available,
            index=available.index("energy_aware") if "energy_aware" in available else 0,
            format_func=lambda a: algorithm(a).name,
        )
        st.caption(algorithm(choice).summary)
        sensors = st.slider("Sensors", min_value=10, max_value=150, value=50, step=5)
        radio = st.slider("Radio range (m)", min_value=15, max_value=60, value=30)
        energy = st.select_slider(
            "Battery per sensor",
            options=BATTERIES,
            value=0.2,
            format_func=lambda j: f"{j:g} J",
        )
        traffic = (
            st.segmented_control(
                "Traffic",
                options=("uniform", "bursty"),
                default="uniform",
                format_func=lambda t: "Steady" if t == "uniform" else "Bursty",
            )
            or "uniform"
        )
        rounds = st.slider(
            "Maximum rounds", min_value=50, max_value=ROUND_LIMIT, value=400, step=50
        )
        seed = st.number_input(
            "Field number",
            min_value=0,
            max_value=999_999,
            value=2000,
            help="The same number always places the sensors the same way.",
        )
        submitted = st.form_submit_button("Run simulation", type="primary", width="stretch")
        if len(available) < len(ALGORITHMS):
            st.caption("Q-Learning and the deep Q-network join this list once they are trained.")
    with result_col:
        if submitted:
            overrides: dict[str, PlainData] = {
                "scenario": {
                    "node_count": int(sensors),
                    "communication_range_m": float(radio),
                    "initial_energy_j": float(energy),
                    "max_rounds": int(rounds),
                    "seed": int(seed),
                },
                "traffic": {"profile": str(traffic)},
            }
            _simulate(workspace, str(choice), int(seed), overrides, models)
        _show_saved(workspace)


def _simulate(
    workspace: Workspace,
    name: str,
    seed: int,
    overrides: dict[str, PlainData],
    models: LearnedModels,
) -> None:
    try:
        config = parse_config(deep_merge(load_yaml(workspace.config), overrides))
        models.check(config, (seed,), (name,))
        scenario = build_scenario(config, seed)
    except (ConfigError, ModelError) as error:
        st.error(f"Unable to start the simulation. {error}")
        return
    except TopologyError:
        st.error(
            "No layout connected every sensor to the sink. Increase the radio range or the "
            "number of sensors, then run again."
        )
        return
    bar = st.progress(0.0, text="Starting the simulation")

    def report(done: int, total: int) -> None:
        if done % 5 == 0 or done == total:
            bar.progress(done / total, text=f"Simulating round {done} of up to {total}")

    result = run_simulation(
        scenario,
        make_router(name, config, models, seed),
        model_fingerprint=models.fingerprint_for(name),
        progress=report,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    folder = workspace.results / "runs" / f"dashboard_{stamp}_{name}_seed{seed}"
    write_run_artifacts(result, folder)
    bar.empty()
    st.session_state[LAST_RUN] = str(folder)
    st.toast("Simulation saved")


def _show_saved(workspace: Workspace) -> None:
    saved = workspace.dashboard_runs()
    if not saved:
        empty_state(
            "No simulations yet",
            "Set up a field on the left and choose Run simulation. Small fields finish in a "
            "few seconds.",
        )
        return
    remembered = st.session_state.get(LAST_RUN)
    labels = {str(p): _label(p) for p in saved}
    keys = list(labels)
    index = keys.index(remembered) if isinstance(remembered, str) and remembered in keys else 0
    picked = st.selectbox("Your simulations", keys, index=index, format_func=lambda k: labels[k])
    try:
        stored = data.run(Path(picked))
        area = load_config(stored.directory / "resolved_config.yaml").scenario
    except (ArtifactError, ConfigError, OSError) as error:
        st.error(f"Unable to open this run. {error}")
        return
    _summary(stored)
    info = algorithm(stored.summary.algorithm)
    show(
        replay(
            [ReplayRun(info.name, info.color, stored)],
            width=area.area_width_m,
            height=area.area_height_m,
        )
    )
    _downloads(stored)


def _label(path: Path) -> str:
    parts = path.name.split("_")
    stamp = parts[1] if len(parts) > 1 else ""
    when = (
        datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC).strftime("%d %b, %H:%M UTC")
        if len(stamp) == 15
        else path.name
    )
    name = "_".join(parts[2:-1]) if len(parts) > 3 else ""
    return f"{algorithm(name).name if name else path.name}, {when}"


def _summary(stored: StoredRun) -> None:
    s = stored.summary
    left = s.remaining_energy_j / s.initial_energy_j if s.initial_energy_j else 0.0
    show(
        stats(
            [
                Stat(
                    "First sensor dies",
                    f"{s.fnd:,}" if s.fnd else "Not reached",
                    "rounds" if s.fnd else "",
                ),
                Stat(
                    "Packets delivered",
                    f"{s.delivered_packets:,}",
                    change=f"of {s.offered_packets:,} sent",
                ),
                Stat("Battery left", f"{left:.0%}", glyph=battery(left)),
                Stat(
                    "Rounds until cut off",
                    f"{s.rounds_executed:,}",
                    "rounds",
                    change="then no sensor could reach the sink"
                    if s.stop_reason != "max_rounds"
                    else "still connected at the round limit",
                ),
            ]
        )
    )


def _downloads(stored: StoredRun) -> None:
    with st.expander("Download this simulation"):
        cols = st.columns(2)
        for column, name in zip(cols, ("round_metrics.csv", "summary.json"), strict=True):
            path = stored.directory / name
            if path.is_file():
                column.download_button(
                    f"Download {name}",
                    path.read_bytes(),
                    file_name=name,
                    key=f"dl_{stored.directory.name}_{name}",
                    width="stretch",
                )
