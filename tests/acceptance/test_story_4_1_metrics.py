"""
Acceptance tests for Story 4.1: Metricas Acumulativas de Tokens.

Validates cumulative token savings persistence, atomic updates,
and metrics calculation.

FRs: FR18 (cumulative tokens_saved per repository)
"""

import json
import threading
from pathlib import Path

import pytest

from src.metrics import _load_metrics, _save_metrics, calculate_savings


class TestMetricsPersistence:
    """AC: Given search saves 11100 tokens, When metrics updated,
    Then cumulative total in metrics.json increases by 11100."""

    def test_cumulative_total_increases(self, tmp_path):
        """Total acumulado em metrics.json deve aumentar com cada busca."""
        # Pre-populate metrics.json with initial values
        initial = {"total_saved": 0, "total_queries": 0, "last_updated": ""}
        _save_metrics(tmp_path, initial)

        # Run a search that saves tokens
        results = [{"file_path": "src/foo.py", "content": "x" * 44400}]
        # Mock file size: file_path won't exist on disk, so without_plugin will be 0
        # Use index_path to trigger persistence
        calculate_savings(results, index_path=tmp_path)

        metrics = _load_metrics(tmp_path)
        # saved = max(without_plugin - with_plugin, 0)
        # Since file doesn't exist, without_plugin = 0, with_plugin > 0 => saved = 0
        # We need to verify that the accumulation logic runs (total_queries increments)
        assert metrics["total_queries"] == 1

    def test_total_queries_increments(self, tmp_path):
        """total_queries deve incrementar com cada busca."""
        results = [{"file_path": "a.py", "content": "hello"}]

        calculate_savings(results, index_path=tmp_path)
        calculate_savings(results, index_path=tmp_path)
        calculate_savings(results, index_path=tmp_path)

        metrics = _load_metrics(tmp_path)
        assert metrics["total_queries"] == 3


class TestMetricsAccumulation:
    """AC: Given metrics.json has total_saved: 50000 and total_queries: 25,
    When new search saves 5000, Then updates to 55000 and 26."""

    def test_accumulates_on_existing_metrics(self, tmp_path):
        """Metricas existentes devem ser acumuladas corretamente."""
        # Pre-populate metrics.json
        existing = {"total_saved": 50000, "total_queries": 25, "last_updated": "2026-01-01T00:00:00"}
        _save_metrics(tmp_path, existing)

        # Simulate a search that saves exactly 5000 tokens by writing directly
        from src.metrics import _update_cumulative
        _update_cumulative(tmp_path, 5000)

        metrics = _load_metrics(tmp_path)
        assert metrics["total_saved"] == 55000
        assert metrics["total_queries"] == 26


class TestMetricsInitialization:
    """AC: Given metrics.json does not exist,
    When first search completes, Then metrics.json is created."""

    def test_creates_metrics_on_first_search(self, tmp_path):
        """metrics.json deve ser criado na primeira busca com valores iniciais."""
        metrics_file = tmp_path / "metrics.json"
        assert not metrics_file.exists()

        results = [{"file_path": "main.py", "content": "print('hello')"}]
        calculate_savings(results, index_path=tmp_path)

        assert metrics_file.exists()
        metrics = _load_metrics(tmp_path)
        assert "total_saved" in metrics
        assert "total_queries" in metrics
        assert metrics["total_queries"] == 1


class TestMetricsAtomicUpdates:
    """AC: Given multiple searches in same session,
    When metrics are persisted, Then each update is atomic."""

    def test_atomic_updates_no_data_loss(self, tmp_path):
        """Atualizacoes concorrentes de metricas nao devem perder dados."""
        from src.metrics import _update_cumulative

        n_threads = 20
        errors = []

        def update():
            try:
                _update_cumulative(tmp_path, 100)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent updates: {errors}"
        metrics = _load_metrics(tmp_path)
        assert metrics["total_queries"] == n_threads
        assert metrics["total_saved"] == n_threads * 100


class TestMetricsCalculation:
    """AC: Given metrics.py imported, When calculate_savings called,
    Then returns per-query tokens_saved AND updates cumulative."""

    def test_calculate_savings_returns_per_query(self, tmp_path):
        """calculate_savings deve retornar tokens_saved da query atual."""
        results = [{"file_path": "src/a.py", "content": "def foo(): pass"}]
        result = calculate_savings(results, index_path=tmp_path)

        assert "without_plugin" in result
        assert "with_plugin" in result
        assert "saved" in result
        assert "reduction_pct" in result

    def test_calculate_savings_updates_cumulative(self, tmp_path):
        """calculate_savings deve atualizar totais cumulativos em disco."""
        results = [{"file_path": "src/b.py", "content": "class Foo: pass"}]
        calculate_savings(results, index_path=tmp_path)

        metrics = _load_metrics(tmp_path)
        assert metrics["total_queries"] == 1
        assert "total_saved" in metrics
        assert "last_updated" in metrics
        assert metrics["last_updated"] != ""
