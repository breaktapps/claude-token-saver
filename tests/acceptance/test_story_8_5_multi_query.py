"""
Acceptance tests for Story 8.5: Multi-Query Embedding (2 perspectives).

Validates that search_semantic and search_hybrid generate technical EN reformulations
for PT-BR/abstract queries and merge results via RRF for improved recall.

SQ: SQ6 (search-quality-spec.md section 4.6)
"""

from __future__ import annotations

import asyncio
import random
from unittest.mock import MagicMock, patch

import pytest

from src.query_expansion import EXPANSION_DICT
from src.server import _expand_query_to_technical, _rrf_merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_vector(dim: int = 384, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _make_raw_chunk(
    file_path: str = "src/main.py",
    name: str = "check_plan_limit",
    chunk_type: str = "function",
    line_start: int = 1,
    line_end: int = 10,
    language: str = "python",
    distance: float = 0.1,
    seed: int | None = None,
) -> dict:
    """Raw chunk as returned by storage.search_vector (has _distance, file_hash)."""
    return {
        "file_path": file_path,
        "name": name,
        "chunk_type": chunk_type,
        "line_start": line_start,
        "line_end": line_end,
        "content": f"def {name}(): pass",
        "file_hash": "abc123",
        "language": language,
        "parent_name": "",
        "calls_json": "[]",
        "outline_json": "[]",
        "_distance": distance,
        "embedding": _random_vector(seed=seed),
    }


def _make_storage_mock(is_stale: bool = False) -> MagicMock:
    from src.storage import Storage
    storage = MagicMock(spec=Storage)
    storage.is_stale.return_value = is_stale
    storage.get_file_hashes.return_value = {"src/main.py": "abc123"}
    storage.get_all_chunks_metadata.return_value = []
    return storage


# ---------------------------------------------------------------------------
# AC: Static expansion dictionary exists and is extensible
# ---------------------------------------------------------------------------

class TestExpansionDictionary:
    """AC: Dictionary is a module-level constant dict with known terms."""

    def test_dictionary_is_dict(self):
        assert isinstance(EXPANSION_DICT, dict)

    def test_dictionary_has_core_entries(self):
        assert "plano" in EXPANSION_DICT
        assert "feature" in EXPANSION_DICT
        assert "limita" in EXPANSION_DICT
        assert "uso" in EXPANSION_DICT
        assert "autenticacao" in EXPANSION_DICT
        assert "erro" in EXPANSION_DICT

    def test_plano_expands_to_expected_terms(self):
        assert "plan" in EXPANSION_DICT["plano"]
        assert "subscription" in EXPANSION_DICT["plano"]
        assert "tier" in EXPANSION_DICT["plano"]
        assert "pricing" in EXPANSION_DICT["plano"]

    def test_feature_expands_to_expected_terms(self):
        assert "feature" in EXPANSION_DICT["feature"]
        assert "access" in EXPANSION_DICT["feature"]
        assert "gate" in EXPANSION_DICT["feature"]
        assert "permission" in EXPANSION_DICT["feature"]

    def test_limita_expands_to_expected_terms(self):
        assert "limit" in EXPANSION_DICT["limita"]
        assert "restrict" in EXPANSION_DICT["limita"]
        assert "guard" in EXPANSION_DICT["limita"]
        assert "quota" in EXPANSION_DICT["limita"]

    def test_autenticacao_expands_to_expected_terms(self):
        assert "auth" in EXPANSION_DICT["autenticacao"]
        assert "token" in EXPANSION_DICT["autenticacao"]
        assert "session" in EXPANSION_DICT["autenticacao"]

    def test_erro_expands_to_expected_terms(self):
        assert "error" in EXPANSION_DICT["erro"]
        assert "exception" in EXPANSION_DICT["erro"]
        assert "catch" in EXPANSION_DICT["erro"]

    def test_all_values_are_non_empty_lists(self):
        for key, values in EXPANSION_DICT.items():
            assert isinstance(values, list), f"{key} value is not a list"
            assert len(values) > 0, f"{key} has empty expansion list"
            for term in values:
                assert isinstance(term, str) and term, f"{key} has empty string in expansion"


# ---------------------------------------------------------------------------
# AC: _expand_query_to_technical produces correct reformulations
# ---------------------------------------------------------------------------

class TestExpandQueryToTechnical:
    """AC: Multi-query expansion extracts significant terms and maps to EN equivalents."""

    def test_query_with_known_terms_produces_reformulation(self):
        query = "como o app limita uso de features por plano?"
        result = _expand_query_to_technical(query)
        assert result is not None
        # Should contain expansions from "limita", "uso", "feature"/"features", "plano"
        assert "limit" in result or "restrict" in result or "guard" in result
        assert "plan" in result or "subscription" in result

    def test_known_example_from_spec(self):
        """Spec example: query should expand to include limit/guard/feature/plan terms."""
        query = "como o app limita uso de features por plano?"
        result = _expand_query_to_technical(query)
        assert result is not None
        terms = result.split()
        # At minimum should include some from: limit, restrict, guard, feature, access, plan, subscription, tier
        spec_terms = {"limit", "restrict", "guard", "feature", "access", "gate", "plan", "subscription", "tier"}
        found = spec_terms & set(terms)
        assert len(found) >= 3, f"Expected at least 3 spec terms, found: {found} in '{result}'"

    def test_query_with_no_dictionary_terms_returns_none(self):
        # Pure EN technical query with no PT terms
        query = "the quick brown fox"
        result = _expand_query_to_technical(query)
        assert result is None

    def test_empty_query_returns_none(self):
        assert _expand_query_to_technical("") is None

    def test_query_with_only_stop_words_returns_none(self):
        # All stop words, no dictionary match
        query = "como o que e nao"
        result = _expand_query_to_technical(query)
        assert result is None

    def test_no_duplicate_terms_in_output(self):
        # "plano" and "planos" both expand, but output should have no dupes
        query = "planos plano assinatura"
        result = _expand_query_to_technical(query)
        assert result is not None
        terms = result.split()
        assert len(terms) == len(set(terms)), f"Duplicate terms found in: {result}"

    def test_autenticacao_query(self):
        query = "como funciona autenticacao no sistema?"
        result = _expand_query_to_technical(query)
        assert result is not None
        assert "auth" in result or "authentication" in result or "token" in result

    def test_erro_query(self):
        query = "tratamento de erro na requisicao"
        result = _expand_query_to_technical(query)
        assert result is not None
        assert "error" in result or "exception" in result

    def test_punctuation_removed_before_matching(self):
        # Punctuation should not prevent matching
        query = "como limita! uso? do plano."
        result = _expand_query_to_technical(query)
        assert result is not None
        assert "limit" in result or "restrict" in result

    def test_case_insensitive_matching(self):
        query = "PLANO AUTENTICACAO ERRO"
        result = _expand_query_to_technical(query)
        assert result is not None

    def test_result_is_space_separated_english_terms(self):
        query = "autenticacao e autorizacao"
        result = _expand_query_to_technical(query)
        assert result is not None
        # Should be space-separated words, no special chars
        for term in result.split():
            assert term.replace("_", "").isalpha(), f"Unexpected chars in term: {term}"


# ---------------------------------------------------------------------------
# AC: RRF merge for dual-vector (unit)
# ---------------------------------------------------------------------------

class TestRrfMergeForDualVector:
    """AC: RRF correctly merges two vector result sets, deduplicates, returns top_k."""

    def _make_storage(self) -> MagicMock:
        from src.storage import Storage
        s = MagicMock(spec=Storage)
        s.is_stale.return_value = False
        return s

    def test_merge_two_sets_deduplicates(self):
        storage = self._make_storage()
        r1 = _make_raw_chunk(name="fn_a", line_start=1, seed=1)
        r2 = _make_raw_chunk(name="fn_b", line_start=10, seed=2)
        # Both lists contain fn_a (same key), r1 only in first, r2 only in second
        merged = _rrf_merge([r1, r2], [r1], fetch_k=4, top_k=5, storage=storage)
        names = [r["name"] for r in merged]
        assert names.count("fn_a") == 1, "fn_a should appear exactly once after dedup"

    def test_merge_respects_top_k(self):
        storage = self._make_storage()
        chunks_a = [_make_raw_chunk(name=f"fn_{i}", line_start=i, seed=i) for i in range(6)]
        chunks_b = [_make_raw_chunk(name=f"fn_{i+10}", line_start=i+10, seed=i+10) for i in range(6)]
        merged = _rrf_merge(chunks_a, chunks_b, fetch_k=12, top_k=5, storage=storage)
        assert len(merged) <= 5

    def test_item_in_both_lists_gets_higher_rrf_score(self):
        storage = self._make_storage()
        shared = _make_raw_chunk(name="shared_fn", line_start=1, seed=42)
        unique_a = _make_raw_chunk(name="unique_a", line_start=2, seed=1)
        unique_b = _make_raw_chunk(name="unique_b", line_start=3, seed=2)

        # shared is rank 0 in both lists
        merged = _rrf_merge([shared, unique_a], [shared, unique_b], fetch_k=4, top_k=3, storage=storage)
        names = [r["name"] for r in merged]
        assert names[0] == "shared_fn", "Item in both lists should rank first"

    def test_merged_results_have_expected_fields(self):
        storage = self._make_storage()
        r = _make_raw_chunk(name="check_limit", line_start=5, seed=7)
        merged = _rrf_merge([r], [], fetch_k=2, top_k=2, storage=storage)
        assert len(merged) == 1
        result = merged[0]
        for field in ("file_path", "name", "chunk_type", "line_start", "line_end", "content", "score", "language"):
            assert field in result, f"Field '{field}' missing from merged result"


# ---------------------------------------------------------------------------
# AC: search_semantic calls embed_query twice when expansion matches
# ---------------------------------------------------------------------------

class TestSearchSemanticDualVector:
    """AC: search_semantic uses dual-vector when expansion dictionary matches."""

    def test_embed_query_called_twice_when_expansion_matches(self):
        """When query has PT terms in dictionary, embed_query is called twice."""
        from src.server import search_semantic

        embed_provider = MagicMock()
        embed_provider.embed_query.return_value = _random_vector(seed=1)

        storage = _make_storage_mock()
        storage.search_vector.return_value = []
        storage.search_fts.return_value = []
        storage.get_all_chunks_metadata.return_value = []

        config = MagicMock()
        config.score_threshold = 0.0
        config.embedding_mode = "lite"

        indexer = MagicMock()

        with patch("src.server._init_components", return_value=(config, storage, indexer, embed_provider, None)), \
             patch("src.server._ensure_indexed", return_value={"status": "ready", "notice": None}):
            asyncio.run(search_semantic("como o app limita uso de features por plano?"))

        # Should have been called twice: once for original query, once for technical reformulation
        assert embed_provider.embed_query.call_count == 2

    def test_embed_query_called_once_when_no_expansion(self):
        """When query has no PT terms, only original vector is used."""
        from src.server import search_semantic

        embed_provider = MagicMock()
        embed_provider.embed_query.return_value = _random_vector(seed=1)

        storage = _make_storage_mock()
        storage.search_vector.return_value = []
        storage.get_all_chunks_metadata.return_value = []

        config = MagicMock()
        config.score_threshold = 0.0

        indexer = MagicMock()

        with patch("src.server._init_components", return_value=(config, storage, indexer, embed_provider, None)), \
             patch("src.server._ensure_indexed", return_value={"status": "ready", "notice": None}):
            asyncio.run(search_semantic("the quick brown fox"))

        assert embed_provider.embed_query.call_count == 1


# ---------------------------------------------------------------------------
# AC: search_hybrid calls embed_query twice for semantic branch when expansion matches
# ---------------------------------------------------------------------------

class TestSearchHybridDualVector:
    """AC: search_hybrid uses dual-vector for semantic branch when expansion matches."""

    def test_embed_query_called_twice_when_expansion_matches(self):
        """search_hybrid should embed original + technical when PT terms match."""
        from src.server import search_hybrid

        embed_provider = MagicMock()
        embed_provider.embed_query.return_value = _random_vector(seed=1)

        storage = _make_storage_mock()
        storage.search_vector.return_value = []
        storage.search_fts.return_value = []
        storage.get_all_chunks_metadata.return_value = []

        config = MagicMock()
        config.score_threshold = 0.0

        indexer = MagicMock()

        with patch("src.server._init_components", return_value=(config, storage, indexer, embed_provider, None)), \
             patch("src.server._ensure_indexed", return_value={"status": "ready", "notice": None}):
            asyncio.run(search_hybrid("como o app limita uso de features por plano?"))

        assert embed_provider.embed_query.call_count == 2

    def test_embed_query_called_once_when_no_expansion(self):
        """search_hybrid should embed only once when no PT terms match."""
        from src.server import search_hybrid

        embed_provider = MagicMock()
        embed_provider.embed_query.return_value = _random_vector(seed=1)

        storage = _make_storage_mock()
        storage.search_vector.return_value = []
        storage.search_fts.return_value = []
        storage.get_all_chunks_metadata.return_value = []

        config = MagicMock()
        config.score_threshold = 0.0

        indexer = MagicMock()

        with patch("src.server._init_components", return_value=(config, storage, indexer, embed_provider, None)), \
             patch("src.server._ensure_indexed", return_value={"status": "ready", "notice": None}):
            asyncio.run(search_hybrid("the quick brown fox"))

        assert embed_provider.embed_query.call_count == 1


# ---------------------------------------------------------------------------
# AC: search_exact is NOT affected by multi-query expansion
# ---------------------------------------------------------------------------

class TestSearchExactNotAffected:
    """AC: search_exact uses FTS only, never calls embed_query."""

    def test_search_exact_does_not_embed(self):
        """search_exact must never call embed_query even with PT-BR queries."""
        from src.server import search_exact

        embed_provider = MagicMock()

        storage = _make_storage_mock()
        storage.search_fts.return_value = []
        storage.get_all_chunks_metadata.return_value = []

        config = MagicMock()
        config.score_threshold = 0.0

        indexer = MagicMock()

        with patch("src.server._init_components", return_value=(config, storage, indexer, embed_provider, None)), \
             patch("src.server._ensure_indexed", return_value={"status": "ready", "notice": None}):
            asyncio.run(search_exact("limita uso plano"))

        embed_provider.embed_query.assert_not_called()


# ---------------------------------------------------------------------------
# AC: technical reformulation of spec example
# ---------------------------------------------------------------------------

class TestSpecExampleReformulation:
    """AC: The canonical spec example produces expected reformulation."""

    def test_spec_example_expansion(self):
        """
        Query: 'como o app limita uso de features por plano?'
        Expected terms from spec: limit restrict guard feature access gate plan subscription tier
        """
        query = "como o app limita uso de features por plano?"
        result = _expand_query_to_technical(query)
        assert result is not None

        terms = set(result.split())
        # Spec says expansion should include terms from: limita, uso, feature/features, plano
        from_limita = {"limit", "restrict", "guard", "check", "validate", "cap", "quota"}
        from_uso = {"usage", "access", "consumption", "utilization"}
        from_feature = {"feature", "access", "gate", "permission", "capability"}
        from_plano = {"plan", "subscription", "tier", "pricing"}

        assert bool(terms & from_limita), f"No 'limita' expansion terms found in: {result}"
        assert bool(terms & from_plano), f"No 'plano' expansion terms found in: {result}"
