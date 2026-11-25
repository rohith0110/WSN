"""Learning page: how the two learned strategies were trained and what they became."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wsn.dashboard.catalog import algorithm
from wsn.dashboard.charts import Series, ValueFormat, line_chart
from wsn.dashboard.components import Stat, empty_state, page_head, panel, show, stats
from wsn.dashboard.data import Workspace
from wsn.reporting import training_history_path
from wsn.rl.serialization import ModelError, load_dqn_model, load_model

MODELS = (("q_learning", "q_table.json"), ("dqn", "dqn.json"))


def render(workspace: Workspace) -> None:
    page_head(
        "How the learned strategies were trained",
        "Both learned on their own set of training fields, then were frozen. Every result "
        "elsewhere comes from fields they never saw.",
    )
    histories: dict[str, pd.DataFrame] = {}
    tiles: list[Stat] = []
    for name, file in MODELS:
        path = workspace.models / file
        info = algorithm(name)
        if not path.is_file():
            tiles.append(
                Stat(
                    info.name,
                    "Not trained",
                    change=f"python -m wsn.cli train --algorithm {name}",
                )
            )
            continue
        try:
            if name == "q_learning":
                model = load_model(path)
                size = f"{len(model.policy.table):,} situations in its table"
                seeds, episodes = model.training_seeds, model.episodes
            else:
                dqn = load_dqn_model(path)
                size = f"{dqn.policy.network.parameter_count():,} weights in its network"
                seeds, episodes = dqn.training_seeds, dqn.episodes
        except ModelError as error:
            tiles.append(Stat(info.name, "Unreadable", change=str(error)))
            continue
        tiles.append(
            Stat(
                info.name,
                f"{episodes}",
                "training fields",
                change=f"{size}. Fields {seeds[0]} to {seeds[-1]}",
            )
        )
        history = training_history_path(path, name)
        if history.is_file():
            histories[name] = pd.read_csv(history)
    show(stats(tiles))
    if not histories:
        st.space("small")
        empty_state(
            "No training history yet",
            "Train a strategy and its learning curve appears here.",
            "python -m wsn.cli train --algorithm q_learning",
        )
        return
    st.space("small")
    show(
        panel(
            "Reward per decision, field by field",
            _chart(histories, "mean_reward", "Mean reward", ValueFormat.DECIMAL),
            note=(
                "Each point is one training field. Early fields are mostly random exploration, "
                "so rewards rise as the strategy relies more on what it learned."
            ),
        )
    )
    show(
        panel(
            "Delivery rate on each training field",
            _chart(histories, "offered_load_pdr_pct", "Delivered", ValueFormat.PERCENT),
            note=(
                "Share of all packets delivered on that field. Fields differ, so the line is noisy."
            ),
        )
    )
    st.caption(
        "Test fields are never used for training. Random exploration falls from 100% to 5% "
        "of choices across the training fields and is switched off for every test."
    )


def _chart(histories: dict[str, pd.DataFrame], column: str, label: str, style: ValueFormat) -> str:
    series = [
        Series(
            key=name,
            name=algorithm(name).name,
            color=algorithm(name).color,
            xs=tuple(float(x) for x in frame["episode"]),
            ys=tuple(float(y) for y in frame[column].fillna(0.0)),
        )
        for name, frame in histories.items()
    ]
    return line_chart(
        series, y_label=label, value_format=style, x_label="Training field", animate=False
    )
