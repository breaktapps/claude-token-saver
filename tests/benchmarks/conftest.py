"""Shared fixtures and options for benchmark tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_BASELINES_DIR = _HERE / "baselines"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --update-baseline option to pytest CLI."""
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help=(
            "After running regression tests, update baselines/latest.json with "
            "current metrics and archive the previous baseline as "
            "baselines/v{version}.json."
        ),
    )


@pytest.fixture(scope="session")
def update_baseline(request: pytest.FixtureRequest) -> bool:
    """Return True if --update-baseline was passed on the CLI."""
    return request.config.getoption("--update-baseline", default=False)


@pytest.fixture(scope="session")
def baselines_dir() -> Path:
    """Return the baselines directory path."""
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    return _BASELINES_DIR


@pytest.fixture(scope="session")
def current_version() -> str:
    """Return the current plugin version from pyproject.toml."""
    pyproject = _HERE.parent.parent / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            line = line.strip()
            if line.startswith("version"):
                # version = "0.1.6"
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"').strip("'")
    return "unknown"


def archive_baseline(baselines_dir: Path, version: str) -> Path | None:
    """Copy latest.json to v{version}.json. Returns archive path or None."""
    latest = baselines_dir / "latest.json"
    if not latest.exists():
        return None
    archive = baselines_dir / f"v{version}.json"
    shutil.copy2(latest, archive)
    return archive


def save_baseline(baselines_dir: Path, metrics: dict) -> None:
    """Write metrics dict to latest.json."""
    latest = baselines_dir / "latest.json"
    with open(latest, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
