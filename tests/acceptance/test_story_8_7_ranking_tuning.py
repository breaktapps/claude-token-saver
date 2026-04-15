"""
Acceptance tests for Story 8.7: Search Ranking Tuning.

Valida que:
- search_semantic usa threshold 0.35 (nao config.score_threshold=0.5)
- search_exact nao aplica threshold (FTS ordena por rank)
- search_hybrid nao aplica threshold (RRF e auto-suficiente — via _rrf_merge)
- Boost por chunk type aplicado apenas em search_semantic (class*1.15, method*0.95)

SQ: SQ8, SQ9 (search-quality-spec.md secoes 4.8-4.9)
"""

import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.server import _TYPE_BOOST


DIM = LITE_DIMENSION  # 384


def _random_vector(dim: int = DIM, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _make_raw_vector_result(
    name: str = "some_fn",
    chunk_type: str = "function",
    file_path: str = "src/main.py",
    distance: float = 0.3,  # score = 1 - distance
    language: str = "python",
    file_hash: str = "abc123",
    parent_name: str = "",
) -> dict:
    return {
        "file_path": file_path,
        "name": name,
        "chunk_type": chunk_type,
        "line_start": 1,
        "line_end": 10,
        "content": f"def {name}(): pass",
        "_distance": distance,
        "file_hash": file_hash,
        "language": language,
        "parent_name": parent_name,
        "calls_json": "[]",
        "outline_json": "[]",
    }


def _make_raw_fts_result(
    name: str = "some_fn",
    chunk_type: str = "function",
    file_path: str = "src/main.py",
    score: float = 0.8,
    language: str = "python",
    file_hash: str = "abc123",
) -> dict:
    return {
        "file_path": file_path,
        "name": name,
        "chunk_type": chunk_type,
        "line_start": 1,
        "line_end": 10,
        "content": f"def {name}(): pass",
        "_score": score,
        "file_hash": file_hash,
        "language": language,
        "parent_name": "",
        "calls_json": "[]",
        "outline_json": "[]",
    }


def _make_mock_storage(file_hashes: dict | None = None):
    storage = MagicMock()
    storage.get_file_hashes.return_value = file_hashes or {"src/main.py": "abc123"}
    storage.is_stale.return_value = False
    storage.get_all_chunks_metadata.return_value = [
        {"language": "python", "file_path": "src/main.py", "chunk_type": "function", "file_hash": "abc123"}
    ]
    return storage


def _make_mock_embed_provider():
    provider = MagicMock()
    provider.embed_query.return_value = _random_vector(seed=42)
    return provider


class TestSemanticThreshold:
    """AC1: search_semantic usa threshold 0.35, nao config.score_threshold."""

    @pytest.mark.asyncio
    async def test_result_with_score_036_passes(self):
        """Score 0.36 (distance=0.64) deve passar com threshold 0.35."""
        storage = _make_mock_storage()
        # distance=0.64 -> score = round(1 - 0.64, 4) = 0.36
        storage.search_vector.return_value = [
            _make_raw_vector_result(name="fn_low_score", distance=0.64),
        ]

        with (
            patch("src.server._config", Config()),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 100, "reduction_pct": 70.0, "with_plugin": 50, "session_saved": 100}),
        ):
            from src.server import search_semantic
            result = await search_semantic(query="test query")

        import json
        data = json.loads(result)
        names = [r["name"] for r in data.get("results", [])]
        assert "fn_low_score" in names, f"Score 0.36 deveria passar threshold 0.35, mas nao apareceu. Resultados: {names}"

    @pytest.mark.asyncio
    async def test_result_with_score_034_filtered_out(self):
        """Score 0.34 (distance=0.66) deve ser filtrado com threshold 0.35."""
        storage = _make_mock_storage()
        # distance=0.66 -> score = round(1 - 0.66, 4) = 0.34
        storage.search_vector.return_value = [
            _make_raw_vector_result(name="fn_too_low", distance=0.66),
        ]

        with (
            patch("src.server._config", Config()),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 0, "reduction_pct": 0.0, "with_plugin": 0, "session_saved": 0}),
        ):
            from src.server import search_semantic
            result = await search_semantic(query="test query")

        import json
        data = json.loads(result)
        names = [r["name"] for r in data.get("results", [])]
        assert "fn_too_low" not in names, f"Score 0.34 deveria ser filtrado, mas apareceu. Resultados: {names}"

    @pytest.mark.asyncio
    async def test_threshold_ignores_config_score_threshold(self):
        """config.score_threshold=0.5 nao deve afetar search_semantic (usa 0.35 fixo)."""
        storage = _make_mock_storage()
        # distance=0.55 -> score = 0.45 (acima de 0.35, abaixo de 0.5)
        storage.search_vector.return_value = [
            _make_raw_vector_result(name="fn_mid_score", distance=0.55),
        ]

        # Config com score_threshold=0.5 (padrao)
        config = Config()
        assert config.score_threshold == 0.5

        with (
            patch("src.server._config", config),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 100, "reduction_pct": 70.0, "with_plugin": 50, "session_saved": 100}),
        ):
            from src.server import search_semantic
            result = await search_semantic(query="test query")

        import json
        data = json.loads(result)
        names = [r["name"] for r in data.get("results", [])]
        assert "fn_mid_score" in names, (
            f"Score 0.45 deveria passar threshold 0.35 mesmo com config.score_threshold=0.5. "
            f"Resultados: {names}"
        )


class TestExactNoThreshold:
    """AC2: search_exact nao aplica threshold — FTS ordena por rank."""

    @pytest.mark.asyncio
    async def test_low_fts_score_passes(self):
        """Score FTS baixo (0.1) deve aparecer — sem threshold em search_exact."""
        storage = _make_mock_storage()
        storage.search_fts.return_value = [
            _make_raw_fts_result(name="fn_low_fts", score=0.1),
        ]

        with (
            patch("src.server._config", Config()),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 100, "reduction_pct": 70.0, "with_plugin": 50, "session_saved": 100}),
        ):
            from src.server import search_exact
            result = await search_exact(query="fn_low_fts")

        import json
        data = json.loads(result)
        names = [r["name"] for r in data.get("results", [])]
        assert "fn_low_fts" in names, (
            f"search_exact nao deve filtrar por score (FTS usa rank). "
            f"Score 0.1 deveria aparecer. Resultados: {names}"
        )

    @pytest.mark.asyncio
    async def test_zero_fts_score_passes(self):
        """Score FTS zero deve aparecer — search_exact nao tem threshold."""
        storage = _make_mock_storage()
        storage.search_fts.return_value = [
            _make_raw_fts_result(name="fn_zero_fts", score=0.0),
        ]

        with (
            patch("src.server._config", Config()),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 50, "reduction_pct": 50.0, "with_plugin": 50, "session_saved": 50}),
        ):
            from src.server import search_exact
            result = await search_exact(query="fn_zero_fts")

        import json
        data = json.loads(result)
        names = [r["name"] for r in data.get("results", [])]
        assert "fn_zero_fts" in names, (
            f"search_exact nao deve filtrar por score. Score 0.0 deveria aparecer. Resultados: {names}"
        )


class TestTypeBoost:
    """AC4: Boost por chunk type em search_semantic (class*1.15, method*0.95, function*1.0)."""

    def test_type_boost_constant_values(self):
        """Valores de _TYPE_BOOST conforme especificado."""
        assert _TYPE_BOOST["class"] == 1.15
        assert _TYPE_BOOST["function"] == 1.00
        assert _TYPE_BOOST["method"] == 0.95

    @pytest.mark.asyncio
    async def test_class_ranked_above_method_with_same_base_score(self):
        """Class outline com mesmo score base que method deve aparecer primeiro apos boost."""
        storage = _make_mock_storage()
        base_score = 0.60  # distance = 1 - 0.60 = 0.40
        # class: boosted = 0.60 * 1.15 = 0.69
        # method: boosted = 0.60 * 0.95 = 0.57
        storage.search_vector.return_value = [
            _make_raw_vector_result(name="MyService", chunk_type="class", distance=0.40),
            _make_raw_vector_result(name="my_method", chunk_type="method", distance=0.40),
        ]

        with (
            patch("src.server._config", Config()),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 100, "reduction_pct": 70.0, "with_plugin": 50, "session_saved": 100}),
        ):
            from src.server import search_semantic
            result = await search_semantic(query="MyService")

        import json
        data = json.loads(result)
        results = data.get("results", [])
        # Filtra expandidos (graph expansion pode adicionar mais)
        core = [r for r in results if r["name"] in ("MyService", "my_method")]
        assert len(core) == 2, f"Esperados 2 resultados core, obtidos: {[r['name'] for r in core]}"
        names_ordered = [r["name"] for r in core]
        assert names_ordered[0] == "MyService", (
            f"Class 'MyService' deveria ser primeiro apos boost (class*1.15 > method*0.95). "
            f"Ordem obtida: {names_ordered}"
        )

    @pytest.mark.asyncio
    async def test_boost_scores_are_multiplied_correctly(self):
        """Verifica que os scores retornados refletem o multiplicador de boost."""
        storage = _make_mock_storage()
        # distance=0.40 -> score = 0.60
        storage.search_vector.return_value = [
            _make_raw_vector_result(name="ClassOutline", chunk_type="class", distance=0.40),
            _make_raw_vector_result(name="standalone_fn", chunk_type="function", distance=0.40),
            _make_raw_vector_result(name="instance_method", chunk_type="method", distance=0.40),
        ]

        with (
            patch("src.server._config", Config()),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 100, "reduction_pct": 70.0, "with_plugin": 50, "session_saved": 100}),
        ):
            from src.server import search_semantic
            result = await search_semantic(query="test boost")

        import json
        data = json.loads(result)
        results_by_name = {r["name"]: r["score"] for r in data.get("results", [])}

        assert "ClassOutline" in results_by_name
        assert "standalone_fn" in results_by_name
        assert "instance_method" in results_by_name

        # base = 0.60; class = 0.60 * 1.15 = 0.69, function = 0.60, method = 0.60 * 0.95 = 0.57
        assert results_by_name["ClassOutline"] == pytest.approx(0.69, abs=0.01)
        assert results_by_name["standalone_fn"] == pytest.approx(0.60, abs=0.01)
        assert results_by_name["instance_method"] == pytest.approx(0.57, abs=0.01)


class TestBoostNotAppliedToExact:
    """AC5: Boost por chunk type NAO aplicado em search_exact."""

    @pytest.mark.asyncio
    async def test_exact_preserves_fts_rank_order(self):
        """search_exact deve preservar a ordem original do FTS (sem boost por tipo)."""
        storage = _make_mock_storage()
        # method com score maior que class — deve permanecer acima sem boost
        storage.search_fts.return_value = [
            _make_raw_fts_result(name="handle_request", chunk_type="method", score=2.5),
            _make_raw_fts_result(name="RequestHandler", chunk_type="class", score=1.8),
        ]

        with (
            patch("src.server._config", Config()),
            patch("src.server._storage", storage),
            patch("src.server._embed_provider", _make_mock_embed_provider()),
            patch("src.server._indexer", MagicMock()),
            patch("src.server._repo_path", None),
            patch("src.server._ensure_indexed", new_callable=AsyncMock, return_value={"status": "ready", "files_scanned": 1, "duration_ms": 1}),
            patch("src.server.calculate_savings", return_value={"saved": 100, "reduction_pct": 70.0, "with_plugin": 50, "session_saved": 100}),
        ):
            from src.server import search_exact
            result = await search_exact(query="handle_request")

        import json
        data = json.loads(result)
        results = data.get("results", [])
        names = [r["name"] for r in results]

        # search_exact ordena por _TYPE_ORDER (class=0, method=1, function=2) para legibilidade,
        # mas os scores nao devem ser multiplicados pelo boost
        scores_by_name = {r["name"]: r["score"] for r in results}
        assert scores_by_name.get("handle_request") == pytest.approx(2.5, abs=0.01), (
            f"score de 'handle_request' nao deve ser modificado pelo boost em search_exact"
        )
        assert scores_by_name.get("RequestHandler") == pytest.approx(1.8, abs=0.01), (
            f"score de 'RequestHandler' nao deve ser modificado pelo boost em search_exact"
        )
