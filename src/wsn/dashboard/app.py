"""App shell: theme, brand and navigation between the five pages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from wsn.dashboard.components import BRAND, show
from wsn.dashboard.data import Workspace
from wsn.dashboard.navigation import PAGES
from wsn.dashboard.theme import STYLESHEET
from wsn.dashboard.views import compare, learning, replay, results, run, validation

PageRenderer = Callable[[Workspace], None]

PAGE_SPECS: tuple[tuple[str, str, str, PageRenderer], ...] = (
    ("results", "Results", ":material/insights:", results.render),
    ("replay", "Replay", ":material/play_circle:", replay.render),
    ("compare", "Compare", ":material/compare_arrows:", compare.render),
    ("run", "Run your own", ":material/tune:", run.render),
    ("learning", "Learned strategies", ":material/model_training:", learning.render),
    ("validation", "Reality check", ":material/fact_check:", validation.render),
)


def _bind(renderer: PageRenderer, workspace: Workspace, key: str) -> Callable[[], None]:
    def page() -> None:
        renderer(workspace)

    page.__name__ = f"page_{key}"
    return page


def main(root: Path, start: str = "results") -> None:
    """Render the app for the project at ``root``, opening on page ``start``."""
    st.set_page_config(
        page_title="Routing Lab",
        page_icon=":material/sensors:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    show(STYLESHEET)
    workspace = Workspace(root)
    pages = [
        PAGES.register(
            key,
            st.Page(
                _bind(renderer, workspace, key),
                title=title,
                icon=icon,
                url_path=key,
                default=key == start,
            ),
        )
        for key, title, icon, renderer in PAGE_SPECS
    ]
    navigation = st.navigation(pages, position="hidden")
    with st.sidebar:
        show(BRAND)
        st.space("small")
        for page in pages:
            st.page_link(page, label=page.title, icon=page.icon)
        st.space("medium")
        st.caption(
            "A simulated wireless sensor network. Time is counted in rounds, energy in joules."
        )
    navigation.run()
