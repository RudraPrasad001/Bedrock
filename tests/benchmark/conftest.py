"""Shared fixtures for benchmark, security and integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmark.workspace import Workspace, build_workspace


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME at an empty temp dir and drop real API keys for every benchmark test.

    This guarantees that '~' can never resolve to the real home directory and
    that scripted benchmarks cannot reach a live LLM.
    """
    home = tmp_path / "fake-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return home


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return build_workspace(tmp_path / "workspace")
