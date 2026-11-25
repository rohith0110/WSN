"""Locating and caching saved results for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from wsn.artifacts import (
    COMPARISON_FILES,
    RUN_FILES,
    StoredComparison,
    StoredRun,
    find_folders,
    load_comparison,
    load_run,
)
from wsn.config import load_yaml
from wsn.dashboard.catalog import CONDITIONS, ConditionInfo


@dataclass(frozen=True, slots=True)
class Workspace:
    """The project folder whose ``results/`` and ``models/`` the dashboard reads."""

    root: Path

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def config(self) -> Path:
        return self.root / "configs" / "default.yaml"

    def comparison_path(self, folder: str) -> Path:
        return self.results / "comparisons" / folder

    def conditions(self) -> list[ConditionInfo]:
        """Known test conditions that have a complete comparison on disk."""
        return [
            c
            for c in CONDITIONS
            if all((self.comparison_path(c.folder) / name).is_file() for name in COMPARISON_FILES)
        ]

    def dashboard_runs(self) -> list[Path]:
        return [
            folder
            for folder in find_folders(self.results / "runs", RUN_FILES)
            if folder.name.startswith("dashboard_")
        ]


def _stamp(path: Path) -> float:
    marker = path / "summary.json" if (path / "summary.json").is_file() else path / "per_run.csv"
    return marker.stat().st_mtime if marker.is_file() else 0.0


@st.cache_data(show_spinner=False, max_entries=16)
def _comparison(path: str, stamp: float) -> StoredComparison:
    return load_comparison(Path(path))


@st.cache_data(show_spinner=False, max_entries=48)
def _run(path: str, stamp: float) -> StoredRun:
    return load_run(Path(path))


def comparison(path: Path) -> StoredComparison:
    return _comparison(str(path), _stamp(path))


def run(path: Path) -> StoredRun:
    return _run(str(path), _stamp(path))


def comparison_run_path(comparison_path: Path, variant: str, seed: int, algorithm: str) -> Path:
    return comparison_path / "runs" / f"{variant}_seed{seed}_{algorithm}"


def variants(stored: StoredComparison) -> list[str]:
    return list(dict.fromkeys(str(v) for v in stored.per_run["variant"]))


def seeds(stored: StoredComparison, variant: str) -> list[int]:
    rows = stored.per_run[stored.per_run["variant"] == variant]
    return sorted({int(s) for s in rows["seed"]})


def algorithms(stored: StoredComparison) -> list[str]:
    return list(dict.fromkeys(str(a) for a in stored.per_run["algorithm"]))


def mean_by_round(stored: StoredComparison, variant: str, column: str) -> pd.DataFrame:
    """Rows: round; columns: algorithm; values: mean across seeds."""
    subset = stored.rounds[stored.rounds["variant"] == variant]
    return subset.pivot_table(index="round", columns="algorithm", values=column, aggfunc="mean")


def last_active_round(stored: StoredComparison, variant: str) -> int:
    """Last round any run in this variant was still simulated."""
    subset = stored.rounds[(stored.rounds["variant"] == variant) & stored.rounds["simulated"]]
    return int(subset["round"].max()) if not subset.empty else 1


def settled_round(table: pd.DataFrame, share: float = 0.95) -> int:
    """First round by which every curve has covered ``share`` of its total change."""
    settled = int(table.index.min())
    for column in table.columns:
        values = table[column]
        final = float(values.iloc[-1])
        total = abs(final - float(values.iloc[0]))
        if total == 0:
            continue
        near = values[(values - final).abs() <= (1 - share) * total]
        settled = max(settled, int(near.index.min()))
    return settled


def display_horizon(table: pd.DataFrame) -> int:
    """Rounds worth plotting: until the curves settle, plus a short flat tail."""
    return min(int(settled_round(table) * 1.15) + 1, int(table.index.max()))


def variant_fingerprints(stored: StoredComparison) -> dict[str, str]:
    """Config fingerprint of each variant, read from the saved experiment_config.yaml."""
    path = stored.directory / "experiment_config.yaml"
    if not path.is_file():
        return {}
    root = load_yaml(path)
    variants = root.get("variants") if isinstance(root, dict) else None
    if not isinstance(variants, dict):
        return {}
    result: dict[str, str] = {}
    for name, entry in variants.items():
        fingerprint = entry.get("config_fingerprint") if isinstance(entry, dict) else None
        if isinstance(fingerprint, str):
            result[name] = fingerprint
    return result
