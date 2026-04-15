"""
Acceptance tests for Story 8.4: Coverage Transparency.

Valida que respostas de busca incluem metadados de cobertura
e nivel de confianca, informando o dev sobre recall parcial.

SQ: SQ5 (search-quality-spec.md secao 4.5)
"""

import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.server import _build_coverage
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
    }


def _make_result(language: str, score: float) -> dict:
    return {
        "file_path": f"src/main.{language}",
        "name": "some_fn",
        "chunk_type": "function",
        "line_start": 1,
        "line_end": 5,
        "content": "body",
        "score": score,
        "language": language,
        "stale": False,
        "parent_name": "",
    }


def _mock_storage(languages_in_index: list[str]) -> Storage:
    storage = MagicMock(spec=Storage)
    metadata = [{"language": lang, "file_path": f"file.{lang}", "chunk_type": "function", "file_hash": "x"} for lang in languages_in_index]
    storage.get_all_chunks_metadata.return_value = metadata
    return storage


class TestCoverageHighConfidence:
    """AC2: confidence 'high' quando resultados cobrem >= 80% das linguagens e score > 0.7."""

    def test_high_when_all_languages_covered_and_high_score(self):
        storage = _mock_storage(["typescript", "python"])
        results = [
            _make_result("typescript", 0.85),
            _make_result("python", 0.90),
        ]
        cov = _build_coverage(results, storage, "how does the app handle payments")
        assert cov["confidence"] == "high"
        assert cov["suggestion"] == ""

    def test_high_languages_indexed_equals_languages_with_results(self):
        storage = _mock_storage(["dart", "python", "typescript"])
        results = [
            _make_result("dart", 0.80),
            _make_result("python", 0.82),
            _make_result("typescript", 0.85),
        ]
        cov = _build_coverage(results, storage, "feature gate implementation")
        assert cov["confidence"] == "high"
        assert cov["languages_indexed"] == ["dart", "python", "typescript"]
        assert cov["languages_with_results"] == ["dart", "python", "typescript"]


class TestCoveragePartialConfidence:
    """AC3: confidence 'partial' quando resultados em >= 50% das linguagens OU score 0.5-0.7."""

    def test_partial_when_half_languages_covered(self):
        storage = _mock_storage(["typescript", "dart", "python", "kotlin"])
        results = [
            _make_result("typescript", 0.40),
            _make_result("dart", 0.38),
        ]
        cov = _build_coverage(results, storage, "validate subscription plan")
        assert cov["confidence"] == "partial"

    def test_partial_when_score_in_range_0_5_to_0_7(self):
        storage = _mock_storage(["typescript", "dart", "python", "kotlin"])
        results = [
            _make_result("typescript", 0.60),
        ]
        cov = _build_coverage(results, storage, "check billing limits")
        assert cov["confidence"] == "partial"

    def test_partial_languages_with_results_populated_correctly(self):
        storage = _mock_storage(["typescript", "dart"])
        results = [_make_result("typescript", 0.55)]
        cov = _build_coverage(results, storage, "handle payment error")
        assert cov["languages_with_results"] == ["typescript"]
        assert cov["languages_indexed"] == ["dart", "typescript"]


class TestCoverageLowConfidence:
    """AC4: confidence 'low' quando < 50% das linguagens E score < 0.5."""

    def test_low_when_few_languages_and_low_score(self):
        storage = _mock_storage(["typescript", "dart", "python", "kotlin"])
        results = [_make_result("typescript", 0.30)]
        cov = _build_coverage(results, storage, "find subscription guard")
        assert cov["confidence"] == "low"

    def test_low_when_no_results(self):
        storage = _mock_storage(["typescript", "dart"])
        results = []
        cov = _build_coverage(results, storage, "complex multi word query here")
        assert cov["confidence"] == "low"
        assert cov["languages_with_results"] == []


class TestCoverageSuggestion:
    """AC5/AC6: suggestion gerada para partial/low com query multi-word; ausente para high."""

    def test_suggestion_present_for_partial_multiword_query(self):
        storage = _mock_storage(["typescript", "dart", "python"])
        results = [_make_result("typescript", 0.40)]
        cov = _build_coverage(results, storage, "como o app valida planos")
        assert "Explorer" in cov["suggestion"]
        assert "typescript" in cov["suggestion"]

    def test_no_suggestion_for_high_confidence(self):
        storage = _mock_storage(["typescript", "python"])
        results = [
            _make_result("typescript", 0.85),
            _make_result("python", 0.90),
        ]
        cov = _build_coverage(results, storage, "feature gate service check plan")
        assert cov["suggestion"] == ""

    def test_no_suggestion_for_single_word_query(self):
        storage = _mock_storage(["typescript", "dart", "python"])
        results = [_make_result("typescript", 0.35)]
        cov = _build_coverage(results, storage, "billing")
        assert cov["suggestion"] == ""

    def test_suggestion_for_low_confidence_multiword(self):
        storage = _mock_storage(["typescript", "dart", "python", "kotlin"])
        results = [_make_result("typescript", 0.20)]
        cov = _build_coverage(results, storage, "validate feature gate limits")
        assert cov["confidence"] == "low"
        assert "Explorer" in cov["suggestion"]


class TestCoverageEdgeCases:
    """Casos extremos: indice vazio, linguagem ausente nos resultados."""

    def test_empty_index_returns_low(self):
        storage = _mock_storage([])
        results = []
        cov = _build_coverage(results, storage, "any query here")
        assert cov["confidence"] == "low"
        assert cov["languages_indexed"] == []
        assert cov["languages_with_results"] == []

    def test_results_language_field_missing_ignored(self):
        storage = _mock_storage(["python"])
        results = [{"file_path": "x.py", "name": "fn", "chunk_type": "function",
                    "line_start": 1, "line_end": 5, "content": "body", "score": 0.85,
                    "stale": False, "parent_name": ""}]
        cov = _build_coverage(results, storage, "query with missing language field")
        assert cov["languages_with_results"] == []

    def test_coverage_does_not_call_extra_db_methods(self):
        storage = _mock_storage(["python", "typescript"])
        results = [_make_result("python", 0.75), _make_result("typescript", 0.80)]
        _build_coverage(results, storage, "multi word query test")
        storage.get_all_chunks_metadata.assert_called_once()
        storage.search_vector.assert_not_called()
        storage.search_fts.assert_not_called()
        storage.get_chunks_by_file.assert_not_called()


class TestCoverageIntegration:
    """Testa _build_coverage com Storage real via LanceDB."""

    @pytest.fixture
    def storage(self, tmp_path):
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        repo_path = tmp_path / "fake-repo"
        repo_path.mkdir()
        return Storage(config, repo_path, base_path=tmp_path / "indices")

    def test_coverage_with_real_storage_single_language(self, storage):
        chunks = [
            _make_chunk(file_path="src/main.py", language="python", seed=1),
            _make_chunk(file_path="src/utils.py", name="helper", language="python", seed=2),
        ]
        storage.upsert(chunks)

        results = [_make_result("python", 0.85)]
        cov = _build_coverage(results, storage, "main utility function")
        assert cov["languages_indexed"] == ["python"]
        assert cov["languages_with_results"] == ["python"]
        assert cov["confidence"] == "high"

    def test_coverage_with_real_storage_multi_language_partial(self, storage):
        chunks = [
            _make_chunk(file_path="src/main.py", language="python", seed=1),
            _make_chunk(file_path="lib/widget.dart", name="Widget", language="dart", seed=2),
            _make_chunk(file_path="src/app.ts", name="AppService", language="typescript", seed=3),
        ]
        storage.upsert(chunks)

        results = [_make_result("python", 0.40)]
        cov = _build_coverage(results, storage, "handle payment validation flow")
        assert set(cov["languages_indexed"]) == {"python", "dart", "typescript"}
        assert cov["languages_with_results"] == ["python"]
        assert cov["confidence"] in ("partial", "low")
        assert "Explorer" in cov["suggestion"]
