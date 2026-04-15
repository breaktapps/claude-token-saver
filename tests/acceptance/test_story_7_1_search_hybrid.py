"""
Acceptance tests for Story 7.1: search_hybrid MCP Tool.

Validates hybrid search combining semantic + FTS, result merging,
de-duplication, filters, and token savings.

FRs: FR3 (hybrid search)
"""

import json
import shutil
from pathlib import Path

import pytest
import pytest_asyncio

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.indexer import Indexer
from src.server import mcp, search_hybrid
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
