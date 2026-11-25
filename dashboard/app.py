"""Launch with ``streamlit run dashboard/app.py`` from the project root."""

from __future__ import annotations

from pathlib import Path

from wsn.dashboard.app import main

main(Path(__file__).resolve().parents[1])
