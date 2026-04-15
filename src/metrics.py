"""Token savings metrics for claude-token-saver.

Calculates how many tokens are saved by using semantic search
compared to reading full files via cat/grep.

Persists cumulative totals to metrics.json in the index directory.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("cts.metrics")

# Per-index-path locks for thread-safe read-modify-write
_metrics_locks: dict[str, threading.Lock] = {}
_metrics_locks_lock = threading.Lock()


def _get_metrics_lock(index_path: Path) -> threading.Lock:
    """Return a per-path threading.Lock for metrics updates."""
    key = str(index_path.resolve())
    with _metrics_locks_lock:
        if key not in _metrics_locks:
            _metrics_locks[key] = threading.Lock()
        return _metrics_locks[key]


def calculate_savings(
    results: list[dict],
    repo_path: Path | None = None,
    index_path: Path | None = None,
) -> dict:
    """Calculate token savings from search results.

    Args:
        results: List of search result dicts (must contain file_path, content).
        repo_path: Repository root path for reading file sizes.
        index_path: Index directory path for persisting cumulative metrics.

    Returns:
        Dict with without_plugin, with_plugin, saved, reduction_pct.
    """
    if not results:
        return {
            "without_plugin": 0,
            "with_plugin": 0,
            "saved": 0,
            "reduction_pct": 0.0,
        }

    # without_plugin: estimate tokens from reading unique full files
    unique_files = set()
    for r in results:
        fp = r.get("file_path", "")
        if fp:
            unique_files.add(fp)

    without_plugin = 0
    for fp in unique_files:
        file_size = _get_file_size(fp, repo_path)
        without_plugin += math.ceil(file_size / 4)

    # with_plugin: tokens in the actual JSON response
    response_json = json.dumps(results, ensure_ascii=False)
    with_plugin = math.ceil(len(response_json) / 4)

    saved = max(without_plugin - with_plugin, 0)
    reduction_pct = (saved / without_plugin * 100) if without_plugin > 0 else 0.0

    per_query = {
        "without_plugin": without_plugin,
        "with_plugin": with_plugin,
        "saved": saved,
        "reduction_pct": round(reduction_pct, 1),
    }

    # Persist cumulative metrics if index_path provided
    if index_path is not None:
        _update_cumulative(index_path, saved)

    return per_query


def _update_cumulative(index_path: Path, saved: int) -> None:
    """Atomically update cumulative metrics in metrics.json.

    Thread-safe: uses a per-path lock to protect read-modify-write.

    Args:
        index_path: Directory containing metrics.json.
        saved: Number of tokens saved in this query.
    """
    lock = _get_metrics_lock(index_path)
    with lock:
        metrics = _load_metrics(index_path)
        metrics["total_saved"] = metrics.get("total_saved", 0) + saved
        metrics["total_queries"] = metrics.get("total_queries", 0) + 1
        metrics["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        _save_metrics(index_path, metrics)


def _load_metrics(index_path: Path) -> dict:
    """Load metrics.json from index_path, returning defaults if not found.

    Args:
        index_path: Directory containing metrics.json.

    Returns:
        Dict with total_saved, total_queries, last_updated.
    """
    metrics_file = index_path / "metrics.json"
    if not metrics_file.exists():
        return {"total_saved": 0, "total_queries": 0, "last_updated": ""}

    try:
        with open(metrics_file) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"total_saved": 0, "total_queries": 0, "last_updated": ""}
        return data
    except (json.JSONDecodeError, OSError):
        return {"total_saved": 0, "total_queries": 0, "last_updated": ""}


def _save_metrics(index_path: Path, data: dict) -> None:
    """Atomically save metrics to metrics.json.

    Writes to a temp file in the same directory, then renames (atomic on POSIX).

    Args:
        index_path: Directory to write metrics.json into.
        data: Metrics dict to persist.
    """
    index_path.mkdir(parents=True, exist_ok=True)
    metrics_file = index_path / "metrics.json"

    fd, tmp_path = tempfile.mkstemp(dir=str(index_path), prefix=".metrics_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(metrics_file))
        logger.debug("Saved metrics to %s", metrics_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_file_size(file_path: str, repo_path: Path | None) -> int:
    """Get file size in bytes."""
    if repo_path:
        full_path = repo_path / file_path
        if full_path.exists():
            return full_path.stat().st_size

    # Try as absolute path
    p = Path(file_path)
    if p.exists():
        return p.stat().st_size

    # Fallback: estimate from content in results
    return 0
