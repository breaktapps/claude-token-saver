"""
Acceptance tests for Story 10.2: Re-ranking Heuristic.

Valida que _rerank_results aplica:
- File diversity: max 2 chunks por arquivo; excedentes (menor score) são descartados.
- Short chunk penalty: chunks com < 3 linhas de conteúdo recebem score * 0.5.
- Co-location boost: se 2+ resultados compartilham o mesmo diretório, score * 1.05.
- Re-sort por score decrescente após ajustes.
- search_exact NÃO aplica re-ranking.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.server import _rerank_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    file_path: str = "src/main.py",
    name: str = "fn",
    score: float = 0.8,
    content: str = "line1\nline2\nline3\nline4",
    chunk_type: str = "function",
) -> dict:
    return {
        "file_path": file_path,
        "name": name,
        "chunk_type": chunk_type,
        "line_start": 1,
        "line_end": 5,
        "content": content,
        "score": score,
        "language": "python",
        "stale": False,
        "parent_name": "",
    }


# ---------------------------------------------------------------------------
# File diversity
# ---------------------------------------------------------------------------

class TestFileDiversity:
    def test_three_chunks_same_file_keeps_two_highest(self):
        results = [
            _make_result(file_path="src/a.py", name="fn1", score=0.9),
            _make_result(file_path="src/a.py", name="fn2", score=0.7),
            _make_result(file_path="src/a.py", name="fn3", score=0.5),
        ]
        out = _rerank_results(results)
        names = [r["name"] for r in out]
        assert "fn3" not in names, "Terceiro chunk (menor score) deve ser descartado"
        assert "fn1" in names
        assert "fn2" in names
        assert len(out) == 2

    def test_two_chunks_same_file_both_kept(self):
        results = [
            _make_result(file_path="src/b.py", name="fn1", score=0.9),
            _make_result(file_path="src/b.py", name="fn2", score=0.6),
        ]
        out = _rerank_results(results)
        assert len(out) == 2

    def test_one_chunk_per_file_all_kept(self):
        results = [
            _make_result(file_path="src/a.py", name="fn_a", score=0.9),
            _make_result(file_path="src/b.py", name="fn_b", score=0.8),
            _make_result(file_path="src/c.py", name="fn_c", score=0.7),
        ]
        out = _rerank_results(results)
        assert len(out) == 3

    def test_diversity_keeps_highest_two_by_score(self):
        """Quando há 4 chunks do mesmo arquivo, os 2 com maior score são mantidos."""
        results = [
            _make_result(file_path="src/x.py", name="fn1", score=0.95),
            _make_result(file_path="src/x.py", name="fn2", score=0.85),
            _make_result(file_path="src/x.py", name="fn3", score=0.75),
            _make_result(file_path="src/x.py", name="fn4", score=0.65),
        ]
        out = _rerank_results(results)
        names = [r["name"] for r in out]
        assert "fn1" in names
        assert "fn2" in names
        assert "fn3" not in names
        assert "fn4" not in names


# ---------------------------------------------------------------------------
# Short chunk penalty
# ---------------------------------------------------------------------------

class TestShortChunkPenalty:
    def test_two_line_content_gets_half_score(self):
        original_score = 0.8
        result = _make_result(content="line1\nline2", score=original_score)
        out = _rerank_results([result])
        assert len(out) == 1
        assert out[0]["score"] == pytest.approx(original_score * 0.5, rel=1e-4)

    def test_three_line_content_unchanged(self):
        original_score = 0.8
        result = _make_result(content="line1\nline2\nline3", score=original_score)
        out = _rerank_results([result])
        assert len(out) == 1
        assert out[0]["score"] == pytest.approx(original_score, rel=1e-4)

    def test_empty_lines_not_counted(self):
        """Linhas em branco não contam como conteúdo."""
        original_score = 0.8
        result = _make_result(content="\n\nreal_line\n\n", score=original_score)
        out = _rerank_results([result])
        # somente 1 linha não-branca → penalidade aplicada
        assert out[0]["score"] == pytest.approx(original_score * 0.5, rel=1e-4)

    def test_one_line_gets_penalty(self):
        result = _make_result(content="single_line_content", score=0.9)
        out = _rerank_results([result])
        assert out[0]["score"] == pytest.approx(0.9 * 0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# Co-location boost
# ---------------------------------------------------------------------------

class TestColocationBoost:
    def test_two_results_same_dir_get_boost(self):
        results = [
            _make_result(file_path="src/utils/a.py", name="fn_a", score=0.8),
            _make_result(file_path="src/utils/b.py", name="fn_b", score=0.7),
        ]
        out = _rerank_results(results)
        # ambos devem ter score * 1.05
        scores = {r["name"]: r["score"] for r in out}
        assert scores["fn_a"] == pytest.approx(0.8 * 1.05, rel=1e-4)
        assert scores["fn_b"] == pytest.approx(0.7 * 1.05, rel=1e-4)

    def test_single_result_in_dir_no_boost(self):
        results = [
            _make_result(file_path="src/utils/a.py", name="fn_a", score=0.8),
            _make_result(file_path="src/other/b.py", name="fn_b", score=0.7),
        ]
        out = _rerank_results(results)
        scores = {r["name"]: r["score"] for r in out}
        assert scores["fn_a"] == pytest.approx(0.8, rel=1e-4)
        assert scores["fn_b"] == pytest.approx(0.7, rel=1e-4)

    def test_three_results_same_dir_all_boosted(self):
        results = [
            _make_result(file_path="lib/a.py", name="fn1", score=0.9),
            _make_result(file_path="lib/b.py", name="fn2", score=0.8),
            _make_result(file_path="lib/c.py", name="fn3", score=0.7),
        ]
        out = _rerank_results(results)
        scores = {r["name"]: r["score"] for r in out}
        assert scores["fn1"] == pytest.approx(0.9 * 1.05, rel=1e-4)
        assert scores["fn2"] == pytest.approx(0.8 * 1.05, rel=1e-4)
        assert scores["fn3"] == pytest.approx(0.7 * 1.05, rel=1e-4)


# ---------------------------------------------------------------------------
# Re-sort after adjustments
# ---------------------------------------------------------------------------

class TestResort:
    def test_output_sorted_by_score_descending(self):
        results = [
            _make_result(file_path="src/a.py", name="low", score=0.5),
            _make_result(file_path="src/b.py", name="high", score=0.9),
            _make_result(file_path="src/c.py", name="mid", score=0.7),
        ]
        out = _rerank_results(results)
        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_penalty_can_change_order(self):
        """Chunk com score alto mas conteúdo curto deve cair na ordenação."""
        results = [
            _make_result(file_path="src/a.py", name="short_high", score=1.0, content="x\ny"),
            _make_result(file_path="src/b.py", name="long_low", score=0.7, content="a\nb\nc\nd"),
        ]
        out = _rerank_results(results)
        # short_high: 1.0 * 0.5 = 0.5 < long_low: 0.7
        assert out[0]["name"] == "long_low"
        assert out[1]["name"] == "short_high"

    def test_empty_list_returns_empty(self):
        assert _rerank_results([]) == []


# ---------------------------------------------------------------------------
# search_exact NOT affected
# ---------------------------------------------------------------------------

class TestSearchExactNotAffected:
    """Verifica que search_exact não chama _rerank_results."""

    def _make_storage_mock(self):
        storage = MagicMock()
        storage.get_file_hashes.return_value = {"src/main.py": "abc"}
        storage.search_fts.return_value = [
            {
                "file_path": "src/main.py",
                "name": "fn1",
                "chunk_type": "function",
                "line_start": 1,
                "line_end": 10,
                "content": "line1\nline2\nline3",
                "_score": 10.0,
                "file_hash": "abc",
                "language": "python",
                "parent_name": "",
            }
        ]
        storage.is_stale.return_value = False
        storage.index_path = MagicMock()
        return storage

    @pytest.mark.asyncio
    async def test_search_exact_does_not_apply_rerank(self):
        """_rerank_results não deve ser chamado dentro de search_exact."""
        from src.server import search_exact

        storage = self._make_storage_mock()

        with (
            patch("src.server._init_components") as mock_init,
            patch("src.server._ensure_indexed", new_callable=AsyncMock) as mock_idx,
            patch("src.server.calculate_savings") as mock_savings,
            patch("src.server._build_coverage") as mock_cov,
            patch("src.server._rerank_results") as mock_rerank,
        ):
            mock_init.return_value = (MagicMock(), storage, MagicMock(), MagicMock(), MagicMock())
            mock_idx.return_value = {"status": "ready", "notice": None}
            mock_savings.return_value = {"saved": 0, "reduction_pct": 0}
            mock_cov.return_value = {}

            await search_exact(query="fn1", top_k=5)

            mock_rerank.assert_not_called()
