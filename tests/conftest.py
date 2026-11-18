"""Pytest configuration enforcing workspace-scoped temp directories (never touching C:\\Users)."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from react_mcp_runtime.mcp.sample_env import (
    IncidentWorkspace,
    build_incident_workspace,
    clone_incident_workspace,
    force_rmtree,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_WORKSPACE_ROOT = PROJECT_ROOT / ".tmp_workspace" / "pytest_runs"


@pytest.fixture(scope="session")
def session_tmp_root() -> Iterator[Path]:
    force_rmtree(TMP_WORKSPACE_ROOT)
    TMP_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    yield TMP_WORKSPACE_ROOT
    force_rmtree(TMP_WORKSPACE_ROOT)


@pytest.fixture(scope="session")
def template_incident_ws(session_tmp_root: Path) -> IncidentWorkspace:
    template_dir = session_tmp_root / "_template_incident"
    return build_incident_workspace(template_dir, overwrite=True)


@pytest.fixture()
def ws_tmp_dir(session_tmp_root: Path) -> Iterator[Path]:
    unique_dir = session_tmp_root / f"case_{uuid.uuid4().hex[:10]}"
    unique_dir.mkdir(parents=True, exist_ok=True)
    yield unique_dir
    force_rmtree(unique_dir)


@pytest.fixture()
def incident_ws(template_incident_ws: IncidentWorkspace, ws_tmp_dir: Path) -> IncidentWorkspace:
    return clone_incident_workspace(template_incident_ws, ws_tmp_dir / "incident")
