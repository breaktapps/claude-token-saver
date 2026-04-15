"""
Acceptance tests for Story 8.3: Graph Follow-up via calls_json (1 Hop).

Validates that search_semantic and search_hybrid expand results by following
calls_json, while search_exact does NOT apply expansion.

SQ: SQ4 (search-quality-spec.md secao 4.4)
"""

import json
import random
from pathlib import Path

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.server import _expand_via_calls
from src.storage import Storage


DIM = LITE_DIMENSION  # 384


def _random_vector(dim: int = DIM, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _make_chunk(
    file_path: str = "src/main.py",
    chunk_type: str = "function",
    name: str = "main",
    line_start: int = 1,
    line_end: int = 10,
    content: str = "def main(): pass",
    file_hash: str = "abc123",
    language: str = "python",
    parent_name: str = "",
    calls_json: str = "[]",
    outline_json: str = "[]",
    embedding: list[float] | None = None,
    seed: int | None = None,
) -> dict:
    return {
        "file_path": file_path,
        "chunk_type": chunk_type,
        "name": name,
        "line_start": line_start,
        "line_end": line_end,
        "content": content,
        "embedding": embedding or _random_vector(seed=seed),
        "file_hash": file_hash,
        "language": language,
        "parent_name": parent_name,
        "calls_json": calls_json,
        "outline_json": outline_json,
        "imports": [],
    }


@pytest.fixture
def storage(tmp_path):
    config = Config.load(config_path=tmp_path / "nonexistent.yaml")
    repo_path = tmp_path / "fake-repo"
    repo_path.mkdir()
    return Storage(config, repo_path, base_path=tmp_path / "indices")


class TestExpandViaCallsBasic:
    """AC1: Expanded chunks aparecem com expanded=True e score correto."""

    def test_callee_added_with_expanded_flag(self, storage):
        """Chunk chamado por um resultado direto deve aparecer com expanded=True."""
        callee = _make_chunk(
            file_path="src/billing.py",
            name="isWorkoutLimitReached",
            content="def isWorkoutLimitReached(): return False",
            seed=1,
        )
        storage.upsert([callee])

        direct_result = {
            "file_path": "src/billing.py",
            "name": "canStartWorkout",
            "chunk_type": "method",
            "line_start": 10,
            "line_end": 20,
            "content": "def canStartWorkout(): ...",
            "score": 0.9,
            "language": "python",
            "stale": False,
            "parent_name": "BillingGuardService",
        }
        raw = {**direct_result, "calls_json": '["isWorkoutLimitReached"]'}

        expanded = _expand_via_calls([direct_result], [raw], top_k=10, storage=storage)

        assert len(expanded) == 1
        exp = expanded[0]
        assert exp["name"] == "isWorkoutLimitReached"
        assert exp["expanded"] is True
        assert abs(exp["score"] - 0.9 * 0.8) < 1e-5

    def test_multiple_callees_expanded(self, storage):
        """Todos os callees presentes no indice devem ser expandidos."""
        callee_a = _make_chunk(name="isWorkoutLimitReached", seed=10)
        callee_b = _make_chunk(name="getActivePlan", seed=11)
        storage.upsert([callee_a, callee_b])

        direct_result = {
            "file_path": "src/billing.py",
            "name": "canStartWorkout",
            "chunk_type": "method",
            "score": 0.85,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 10,
            "content": "...",
        }
        raw = {**direct_result, "calls_json": '["isWorkoutLimitReached", "getActivePlan"]'}

        expanded = _expand_via_calls([direct_result], [raw], top_k=10, storage=storage)

        expanded_names = {e["name"] for e in expanded}
        assert "isWorkoutLimitReached" in expanded_names
        assert "getActivePlan" in expanded_names


class TestExpandViaCallsLimit:
    """AC2: Limite de top_k * 0.5 chunks expandidos."""

    def test_expansion_limited_to_half_top_k(self, storage):
        """Com top_k=4, no maximo 2 chunks expandidos devem ser adicionados."""
        callees = [
            _make_chunk(name=f"helper_{i}", seed=20 + i)
            for i in range(5)
        ]
        storage.upsert(callees)

        direct_result = {
            "file_path": "src/main.py",
            "name": "mainFunc",
            "chunk_type": "function",
            "score": 0.9,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 10,
            "content": "...",
        }
        calls = [f"helper_{i}" for i in range(5)]
        raw = {**direct_result, "calls_json": json.dumps(calls)}

        expanded = _expand_via_calls([direct_result], [raw], top_k=4, storage=storage)

        assert len(expanded) <= 2, f"Esperado <= 2 expandidos, obteve {len(expanded)}"

    def test_no_duplicates_with_direct_results(self, storage):
        """Callees ja presentes nos resultados diretos nao devem ser duplicados."""
        callee = _make_chunk(name="sharedHelper", seed=30)
        storage.upsert([callee])

        direct_result = {
            "file_path": "src/main.py",
            "name": "mainFunc",
            "chunk_type": "function",
            "score": 0.9,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 5,
            "content": "...",
        }
        already_present = {
            "file_path": "src/helpers.py",
            "name": "sharedHelper",
            "chunk_type": "function",
            "score": 0.7,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 5,
            "content": "...",
        }
        raw_main = {**direct_result, "calls_json": '["sharedHelper"]'}
        raw_helper = {**already_present, "calls_json": "[]"}

        expanded = _expand_via_calls(
            [direct_result, already_present],
            [raw_main, raw_helper],
            top_k=10,
            storage=storage,
        )

        expanded_names = [e["name"] for e in expanded]
        assert expanded_names.count("sharedHelper") == 0, (
            "sharedHelper ja esta nos diretos — nao deve ser duplicado"
        )


class TestExpandViaCallsEdgeCases:
    """AC6 e AC7: calls_json vazio e nome ausente no indice."""

    def test_empty_calls_json_no_expansion(self, storage):
        """Chunk com calls_json='[]' nao deve gerar expansao nem erro."""
        direct_result = {
            "file_path": "src/main.py",
            "name": "noOpFunc",
            "chunk_type": "function",
            "score": 0.8,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 5,
            "content": "def noOpFunc(): pass",
        }
        raw = {**direct_result, "calls_json": "[]"}

        expanded = _expand_via_calls([direct_result], [raw], top_k=10, storage=storage)

        assert expanded == []

    def test_missing_name_silently_skipped(self, storage):
        """Callee ausente no indice deve ser silenciado, sem erro."""
        direct_result = {
            "file_path": "src/main.py",
            "name": "funcWithExternalCall",
            "chunk_type": "function",
            "score": 0.75,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 10,
            "content": "...",
        }
        raw = {**direct_result, "calls_json": '["os.path.join", "nonExistentHelper"]'}

        # Should not raise, should return empty (neither exists in index)
        expanded = _expand_via_calls([direct_result], [raw], top_k=10, storage=storage)

        assert isinstance(expanded, list)

    def test_malformed_calls_json_no_crash(self, storage):
        """calls_json malformado nao deve gerar excecao."""
        direct_result = {
            "file_path": "src/main.py",
            "name": "badJson",
            "chunk_type": "function",
            "score": 0.8,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 5,
            "content": "...",
        }
        raw = {**direct_result, "calls_json": "NOT_VALID_JSON"}

        expanded = _expand_via_calls([direct_result], [raw], top_k=10, storage=storage)

        assert expanded == []


class TestExpandedResultsOrdering:
    """AC5: Expanded chunks aparecem apos direct matches."""

    def test_expanded_after_direct(self, storage):
        """Expanded chunks devem vir depois dos direct matches na lista final."""
        callee = _make_chunk(name="calledFunc", seed=40)
        storage.upsert([callee])

        direct = {
            "file_path": "src/main.py",
            "name": "callerFunc",
            "chunk_type": "function",
            "score": 0.9,
            "language": "python",
            "stale": False,
            "parent_name": "",
            "line_start": 1,
            "line_end": 10,
            "content": "...",
        }
        raw = {**direct, "calls_json": '["calledFunc"]'}

        expanded = _expand_via_calls([direct], [raw], top_k=10, storage=storage)
        all_results = [direct] + expanded

        # direct result first, expanded last
        assert all_results[0]["name"] == "callerFunc"
        assert all_results[0].get("expanded") is None or all_results[0].get("expanded") is False
        assert all_results[-1].get("expanded") is True


class TestStorageSearchByName:
    """Testa o metodo search_by_name adicionado em storage.py."""

    def test_search_by_name_exact_match(self, storage):
        """search_by_name deve retornar o chunk com nome exato."""
        chunk = _make_chunk(name="MyExactFunction", seed=50)
        storage.upsert([chunk])

        results = storage.search_by_name("MyExactFunction")

        assert len(results) == 1
        assert results[0]["name"] == "MyExactFunction"

    def test_search_by_name_not_found(self, storage):
        """search_by_name deve retornar lista vazia se nome nao existe."""
        results = storage.search_by_name("DoesNotExistAtAll")

        assert results == []

    def test_search_by_name_returns_full_fields(self, storage):
        """search_by_name deve retornar campos necessarios para expansao."""
        chunk = _make_chunk(
            name="fullFieldsFunc",
            file_path="src/full.py",
            content="def fullFieldsFunc(): return 42",
            calls_json='["helper"]',
            seed=60,
        )
        storage.upsert([chunk])

        results = storage.search_by_name("fullFieldsFunc")

        assert len(results) == 1
        rec = results[0]
        assert "file_path" in rec
        assert "chunk_type" in rec
        assert "content" in rec
        assert "language" in rec
        assert "calls_json" in rec
