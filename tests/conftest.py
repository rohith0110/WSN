"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from wsn.config import SimulationConfig, load_config

from .support import SMOKE_CONFIG_PATH


@pytest.fixture
def smoke_config() -> SimulationConfig:
    return load_config(SMOKE_CONFIG_PATH)
