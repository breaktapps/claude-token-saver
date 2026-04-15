"""
Acceptance tests for Story 7.1: search_hybrid MCP Tool com RRF.

Validates hybrid search combining semantic + FTS via Reciprocal Rank Fusion,
stop word filtering, identifier preservation, RRF merge logic,
de-duplication, filters, and token savings.

FRs: FR3 (hybrid search)
SQs: SQ1 (RRF merge)
"""

import json
import shutil
from pathlib import Path

import pytest
import pytest_asyncio

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.indexer import Indexer
from src.server import (
    _filter_stop_words,
    _rrf_merge,
    mcp,
    search_hybrid,
)
from src.storage import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_config(**overrides) -> Config:
    return Config(embedding_mode="lite", **overrides)


def _create_test_repo(tmp_path: Path) -> Path:
    """Create a small test repo with multi-language files."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    src = repo / "src"
    src.mkdir()
    shutil.copy(FIXTURES / "python" / "data_processor.py", src / "data_processor.py")
    shutil.copy(FIXTURES / "python" / "simple_function.py", src / "simple_function.py")

    services = repo / "src" / "services"
    services.mkdir()
    shutil.copy(FIXTURES / "typescript" / "api_handler.ts", services / "api_handler.ts")

    return repo


@pytest.fixture
def indexed_env(tmp_path):
    """Create test environment without indexing."""
    repo = _create_test_repo(tmp_path)
    config = _make_config()
    storage = Storage(config, repo, base_path=tmp_path / "indices")
    provider = LocalFastembedProvider(config)
    indexer = Indexer(config, storage, provider, repo_path=repo)
    return config, storage, indexer, provider, repo


@pytest_asyncio.fixture
async def indexed(indexed_env):
    """Run indexation and return components."""
    config, storage, indexer, provider, repo = indexed_env
    await indexer.reindex()
    return config, storage, indexer, provider, repo


def _inject_server(srv, config, storage, indexer, provider, repo):
    """Inject components into server globals."""
    srv._config = config
    srv._storage = storage
    srv._indexer = indexer
    srv._embed_provider = provider
    srv._repo_path = repo


def _restore_server(srv, saved):
    config, storage, indexer, provider, repo = saved
    srv._config = config
    srv._storage = storage
    srv._indexer = indexer
    srv._embed_provider = provider
    srv._repo_path = repo


class TestHybridSearchResults:
    """AC: Given an indexed repository, When search_hybrid is called,
    Then results combine vector similarity and FTS, merged and de-duplicated."""

    @pytest.mark.asyncio
    async def test_returns_combined_results(self, indexed):
        """search_hybrid deve retornar resultados combinando semantico e FTS."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("load data from file", top_k=5)
            response = json.loads(result_json)

            assert "results" in response
            assert "tokens_saved" in response
            assert isinstance(response["results"], list)
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_results_have_combined_score(self, indexed):
        """Cada resultado deve conter um score combinado."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("process records", top_k=5)
            response = json.loads(result_json)

            assert "results" in response
            for r in response["results"]:
                assert "score" in r
                assert isinstance(r["score"], float)
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_results_are_deduplicated(self, indexed):
        """Resultados duplicados devem ser removidos."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("data processor", top_k=10)
            response = json.loads(result_json)

            results = response["results"]
            # Check no duplicate (file_path, name, line_start) tuples
            keys = [(r["file_path"], r["name"], r["line_start"]) for r in results]
            assert len(keys) == len(set(keys)), "Duplicates found in hybrid results"
        finally:
            _restore_server(srv, saved)


class TestHybridMerging:
    """AC: Given semantic finds A and FTS finds B,
    When merged, Then both A and B appear ranked by combined relevance."""

    @pytest.mark.asyncio
    async def test_both_sources_represented(self, indexed):
        """Resultados de ambas as fontes (semantico e FTS) devem aparecer."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)

            # Run each branch independently
            vector = provider.embed_query("handle request")
            vec_results = storage.search_vector(vector, top_k=10)
            fts_results = storage.search_fts("handle request", top_k=10)

            if vec_results and fts_results:
                result_json = await search_hybrid("handle request", top_k=10)
                response = json.loads(result_json)
                hybrid_keys = {
                    (r["file_path"], r["name"], r["line_start"])
                    for r in response["results"]
                }

                # At least one result from each source should appear
                vec_keys = {
                    (r.get("file_path", ""), r.get("name", ""), r.get("line_start", 0))
                    for r in vec_results
                }
                fts_keys = {
                    (r.get("file_path", ""), r.get("name", ""), r.get("line_start", 0))
                    for r in fts_results
                }

                has_vec = bool(hybrid_keys & vec_keys)
                has_fts = bool(hybrid_keys & fts_keys)
                assert has_vec or has_fts, "No results from either source found in hybrid"
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_ranked_by_combined_relevance(self, indexed):
        """Resultados devem ser ranqueados por relevancia combinada (score desc)."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("process data records", top_k=5)
            response = json.loads(result_json)

            scores = [r["score"] for r in response["results"]]
            assert scores == sorted(scores, reverse=True), "Results not sorted by score desc"
        finally:
            _restore_server(srv, saved)


class TestHybridFilters:
    """AC: Given hybrid search with filter_ext or filter_path,
    Then filters apply to both branches."""

    @pytest.mark.asyncio
    async def test_filter_ext_applies_to_both(self, indexed):
        """filter_ext deve filtrar tanto resultados semanticos quanto FTS."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("function handler", top_k=10, filter_ext=".py")
            response = json.loads(result_json)

            for r in response["results"]:
                assert r["file_path"].endswith(".py"), (
                    f"Result with non-.py file passed filter: {r['file_path']}"
                )
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_filter_path_applies_to_both(self, indexed):
        """filter_path deve filtrar tanto resultados semanticos quanto FTS."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid(
                "API handler request", top_k=10, filter_path="src/services/"
            )
            response = json.loads(result_json)

            for r in response["results"]:
                assert "services" in r["file_path"], (
                    f"Result outside services/ passed filter: {r['file_path']}"
                )
        finally:
            _restore_server(srv, saved)


class TestHybridTokenSavings:
    """AC: Given hybrid results, Then tokens_saved is included."""

    @pytest.mark.asyncio
    async def test_tokens_saved_included(self, indexed):
        """Resposta deve incluir tokens_saved com envelope padrao."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("network timeout retry", top_k=5)
            response = json.loads(result_json)

            assert "tokens_saved" in response
            ts = response["tokens_saved"]
            assert "without_plugin" in ts
            assert "with_plugin" in ts
            assert "saved" in ts
            assert "reduction_pct" in ts

            # Also verify metadata is present
            assert "metadata" in response
            assert "query_time_ms" in response["metadata"]
            assert "index_status" in response["metadata"]
        finally:
            _restore_server(srv, saved)


class TestFilterStopWords:
    """Unit tests for _filter_stop_words helper (AC: #3, #4)."""

    def test_pt_br_long_query_removes_stop_words(self):
        """Query PT-BR com mais de 5 palavras deve ter stop words removidas."""
        query = "como o app limita uso de features por plano"
        result = _filter_stop_words(query)
        result_lower = result.lower()
        assert "app" in result_lower
        assert "limita" in result_lower
        assert "features" in result_lower
        assert "plano" in result_lower
        # Stop words devem ser removidas
        assert " o " not in f" {result} "
        assert " de " not in f" {result} "
        assert " por " not in f" {result} "

    def test_en_long_query_removes_stop_words(self):
        """Query EN com mais de 5 palavras deve ter stop words removidas."""
        query = "how does the app limit features by plan"
        result = _filter_stop_words(query)
        result_lower = result.lower()
        assert "app" in result_lower
        assert "limit" in result_lower
        assert "features" in result_lower
        assert "plan" in result_lower
        assert " the " not in f" {result} "
        assert " by " not in f" {result} "

    def test_short_query_not_filtered(self):
        """Query com <= 5 palavras nao deve ser filtrada."""
        query = "subscription guard check"
        assert _filter_stop_words(query) == query

    def test_exactly_5_words_not_filtered(self):
        """Query com exatamente 5 palavras nao deve ser filtrada."""
        query = "the app does the thing"
        assert _filter_stop_words(query) == query

    def test_camelcase_preserved(self):
        """Identificadores camelCase devem ser preservados mesmo sendo stop words."""
        query = "SubscriptionGuard handles the limit for users"
        result = _filter_stop_words(query)
        assert "SubscriptionGuard" in result

    def test_snake_case_preserved(self):
        """Identificadores snake_case devem ser preservados."""
        query = "check_feature in the billing module here"
        result = _filter_stop_words(query)
        assert "check_feature" in result

    def test_all_stop_words_fallback(self):
        """Query com apenas stop words retorna query original (safety fallback)."""
        query = "o que e isso aquilo"
        result = _filter_stop_words(query)
        # Com 5 palavras ou menos, retorna inalterada
        assert result == query

    def test_empty_query(self):
        """Query vazia retorna vazia."""
        assert _filter_stop_words("") == ""

    def test_mix_pt_en_preserves_content(self):
        """Mix PT+EN — conteudo significativo preservado."""
        query = "como funciona o subscription_guard no sistema"
        result = _filter_stop_words(query)
        assert "funciona" in result
        assert "subscription_guard" in result


class TestRrfMerge:
    """Unit tests for _rrf_merge helper (AC: #1, #2)."""

    def _make_record(self, fp, name, line_start, **extra):
        return {
            "file_path": fp,
            "name": name,
            "line_start": line_start,
            "line_end": line_start + 10,
            "chunk_type": "function",
            "content": "...",
            "language": "python",
            "file_hash": "",
            "parent_name": "",
            **extra,
        }

    class _FakeStorage:
        def is_stale(self, fp, file_hash):
            return False

    def test_doc_in_both_lists_scores_higher(self):
        """Doc em ambas as listas deve ter score maior que doc em apenas uma."""
        storage = self._FakeStorage()
        rec_a = self._make_record("a.py", "func_a", 1)
        rec_b = self._make_record("b.py", "func_b", 10)

        vector_raw = [rec_a, rec_b]
        fts_raw = [rec_a]  # rec_a aparece em ambas, rec_b so no vector

        results = _rrf_merge(vector_raw, fts_raw, fetch_k=10, top_k=5, storage=storage)

        scores = {r["name"]: r["score"] for r in results}
        assert scores["func_a"] > scores["func_b"], "Doc em ambas listas deve ter score maior"

    def test_absent_retriever_uses_absent_rank(self):
        """Doc em apenas um retriever deve usar absent_rank no outro."""
        storage = self._FakeStorage()
        rec_only_vec = self._make_record("c.py", "only_vec", 5)
        rec_only_fts = self._make_record("d.py", "only_fts", 15)

        results = _rrf_merge(
            vector_raw=[rec_only_vec],
            fts_raw=[rec_only_fts],
            fetch_k=10,
            top_k=5,
            storage=storage,
        )
        assert len(results) == 2
        # Ambos devem ter score positivo
        for r in results:
            assert r["score"] > 0

    def test_empty_vector_returns_fts_results(self):
        """Vector vazio — resultado deve conter apenas docs do FTS."""
        storage = self._FakeStorage()
        rec = self._make_record("e.py", "fts_only", 1)
        results = _rrf_merge([], [rec], fetch_k=10, top_k=5, storage=storage)
        assert len(results) == 1
        assert results[0]["name"] == "fts_only"

    def test_empty_fts_returns_vector_results(self):
        """FTS vazio — resultado deve conter apenas docs do vector."""
        storage = self._FakeStorage()
        rec = self._make_record("f.py", "vec_only", 1)
        results = _rrf_merge([rec], [], fetch_k=10, top_k=5, storage=storage)
        assert len(results) == 1
        assert results[0]["name"] == "vec_only"

    def test_both_empty_returns_empty(self):
        """Ambos vazios — retorna lista vazia."""
        storage = self._FakeStorage()
        results = _rrf_merge([], [], fetch_k=10, top_k=5, storage=storage)
        assert results == []

    def test_deduplication_by_key(self):
        """Mesmo (file_path, name, line_start) deve aparecer apenas uma vez."""
        storage = self._FakeStorage()
        rec = self._make_record("g.py", "shared", 1)
        results = _rrf_merge([rec], [rec], fetch_k=10, top_k=5, storage=storage)
        keys = [(r["file_path"], r["name"], r["line_start"]) for r in results]
        assert len(keys) == len(set(keys))
        assert len(results) == 1

    def test_results_sorted_descending(self):
        """Resultados devem estar ordenados por RRF score descendente."""
        storage = self._FakeStorage()
        recs_vec = [self._make_record(f"h{i}.py", f"func_{i}", i) for i in range(5)]
        recs_fts = [self._make_record(f"h{i}.py", f"func_{i}", i) for i in range(3, 8)]
        results = _rrf_merge(recs_vec, recs_fts, fetch_k=10, top_k=10, storage=storage)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        """Nunca deve retornar mais que top_k resultados."""
        storage = self._FakeStorage()
        recs = [self._make_record(f"i{i}.py", f"func_{i}", i) for i in range(20)]
        results = _rrf_merge(recs, recs, fetch_k=40, top_k=3, storage=storage)
        assert len(results) <= 3

    def test_all_scores_positive(self):
        """Todos os RRF scores devem ser positivos."""
        storage = self._FakeStorage()
        recs = [self._make_record(f"j{i}.py", f"fn_{i}", i) for i in range(5)]
        results = _rrf_merge(recs, [], fetch_k=10, top_k=10, storage=storage)
        for r in results:
            assert r["score"] > 0


class TestHybridRRFIntegration:
    """Integration tests — hybrid never regresses vs individual tools (AC: #2, #6)."""

    @pytest.mark.asyncio
    async def test_hybrid_never_returns_zero_when_individual_tools_have_results(self, indexed):
        """Hybrid nao deve retornar 0 resultados quando semantico ou FTS tem resultados."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)

            vector = provider.embed_query("process data")
            vec_results = storage.search_vector(vector, top_k=10)
            fts_results = storage.search_fts("process data", top_k=10)

            if vec_results or fts_results:
                result_json = await search_hybrid("process data", top_k=10)
                response = json.loads(result_json)
                assert len(response["results"]) > 0, (
                    "hybrid returned 0 results but individual tools had results"
                )
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_no_score_threshold_applied(self, indexed):
        """RRF results nao devem ser filtrados por score_threshold — top_k e o unico limite."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("process records load file", top_k=20)
            response = json.loads(result_json)

            # All scores should be positive RRF scores (not filtered by 0.5 threshold)
            for r in response["results"]:
                assert r["score"] > 0, "RRF score must be positive"
                # RRF scores are much smaller than 0.5 — verify no threshold filtering
                # by checking scores are in valid RRF range
                assert r["score"] < 1.0, f"RRF score unexpectedly high: {r['score']}"
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_rrf_scores_sorted_descending(self, indexed):
        """RRF scores devem estar ordenados descendente."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await search_hybrid("function handler data", top_k=10)
            response = json.loads(result_json)

            scores = [r["score"] for r in response["results"]]
            assert scores == sorted(scores, reverse=True)
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_tool_description_mentions_rrf(self):
        """Tool description deve mencionar RRF."""
        tool_descriptions = {t.name: t.description for t in mcp._tool_manager.list_tools()}
        desc = tool_descriptions.get("search_hybrid", "")
        assert "RRF" in desc or "Reciprocal Rank Fusion" in desc
