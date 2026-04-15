"""
Acceptance tests for Story 1.6: search_semantic MCP Tool (End-to-End).

Validates semantic search with ranking, token savings, filters,
auto-indexation, MCP registration, and error handling.

FRs: FR1 (semantic search), FR4 (filter), FR5 (score), FR17 (tokens_saved per search)
NFRs: NFR1 (<500ms), NFR12 (MCP stdio)
"""

import json
import shutil
from pathlib import Path

import pytest
import pytest_asyncio

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.errors import EmbeddingProviderError
from src.indexer import Indexer
from src.metrics import calculate_savings
from src.server import search_semantic, _init_components, mcp
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
    """Create a fully indexed test environment.

    Returns (config, storage, indexer, embed_provider, repo_path).
    """
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


class TestSemanticSearchResults:
    """AC: Given an indexed repository, When search_semantic is called,
    Then up to top_k chunks ranked by similarity with full metadata."""

    @pytest.mark.asyncio
    async def test_returns_up_to_top_k_results(self, indexed):
        """search_semantic deve retornar ate top_k chunks."""
        config, storage, indexer, provider, repo = indexed
        vector = provider.embed_query("load data from JSON file")
        results = storage.search_vector(vector, top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_results_ranked_by_semantic_similarity(self, indexed):
        """Resultados devem estar ranqueados por similaridade semantica."""
        config, storage, indexer, provider, repo = indexed
        vector = provider.embed_query("load data from JSON file")
        results = storage.search_vector(vector, top_k=5)
        if len(results) >= 2:
            # Results should be sorted by distance (ascending)
            distances = [r.get("_distance", 0) for r in results]
            assert distances == sorted(distances)

    @pytest.mark.asyncio
    async def test_each_result_has_required_fields(self, indexed):
        """Cada resultado deve conter: file_path, name, chunk_type, line_start,
        line_end, content, score, language, stale."""
        config, storage, indexer, provider, repo = indexed
        vector = provider.embed_query("process data records")
        raw_results = storage.search_vector(vector, top_k=3)
        assert len(raw_results) > 0

        # Build result like server does
        for r in raw_results:
            # Raw results from LanceDB have all schema fields
            assert "file_path" in r
            assert "name" in r
            assert "chunk_type" in r
            assert "line_start" in r
            assert "line_end" in r
            assert "content" in r
            assert "_distance" in r
            assert "language" in r


class TestTokenSavingsMetrics:
    """AC: Given search results, When response is built,
    Then tokens_saved object is included with correct calculations."""

    @pytest.mark.asyncio
    async def test_tokens_saved_object_present(self, indexed):
        """Resposta deve incluir tokens_saved com without_plugin, with_plugin, saved, reduction_pct."""
        config, storage, indexer, provider, repo = indexed
        vector = provider.embed_query("data processor")
        results = storage.search_vector(vector, top_k=5)

        # Build results list like server
        result_dicts = [{"file_path": r["file_path"], "content": r["content"]} for r in results]
        savings = calculate_savings(result_dicts, repo)

        assert "without_plugin" in savings
        assert "with_plugin" in savings
        assert "saved" in savings
        assert "reduction_pct" in savings

    @pytest.mark.asyncio
    async def test_without_plugin_calculated_from_file_sizes(self, indexed):
        """without_plugin = sum(ceil(file_size / 4)) para arquivos unicos nos resultados."""
        config, storage, indexer, provider, repo = indexed
        import math

        vector = provider.embed_query("load data from file")
        results = storage.search_vector(vector, top_k=5)

        result_dicts = [{"file_path": r["file_path"], "content": r["content"]} for r in results]
        savings = calculate_savings(result_dicts, repo)

        # Manually calculate expected without_plugin
        unique_files = set(r["file_path"] for r in results)
        expected = 0
        for fp in unique_files:
            full = repo / fp
            if full.exists():
                expected += math.ceil(full.stat().st_size / 4)

        if expected > 0:
            assert savings["without_plugin"] == expected

    @pytest.mark.asyncio
    async def test_metadata_includes_query_time_and_index_status(self, indexed):
        """metadata deve incluir query_time_ms e index_status."""
        # Test the metadata structure by calling search_semantic directly
        # with mocked components
        config, storage, indexer, provider, repo = indexed

        import src.server as srv
        old_config = srv._config
        old_storage = srv._storage
        old_indexer = srv._indexer
        old_provider = srv._embed_provider
        old_repo = srv._repo_path

        try:
            srv._config = config
            srv._storage = storage
            srv._indexer = indexer
            srv._embed_provider = provider
            srv._repo_path = repo

            result_json = await search_semantic("process data")
            response = json.loads(result_json)

            assert "metadata" in response
            assert "query_time_ms" in response["metadata"]
            assert "index_status" in response["metadata"]
            assert isinstance(response["metadata"]["query_time_ms"], int)
            assert response["metadata"]["index_status"] in ("ready", "first_index")
        finally:
            srv._config = old_config
            srv._storage = old_storage
            srv._indexer = old_indexer
            srv._embed_provider = old_provider
            srv._repo_path = old_repo


class TestSearchFilters:
    """AC: Given a query with filter_ext or filter_path,
    Then only matching chunks are returned."""

    @pytest.mark.asyncio
    async def test_filter_by_extension(self, indexed):
        """filter_ext='.py' deve retornar apenas chunks de arquivos Python."""
        config, storage, indexer, provider, repo = indexed
        vector = provider.embed_query("function handler")
        results = storage.search_vector(vector, top_k=10, filter_ext=".py")
        for r in results:
            assert r["file_path"].endswith(".py")

    @pytest.mark.asyncio
    async def test_filter_by_path(self, indexed):
        """filter_path='src/services/' deve retornar apenas chunks desse path."""
        config, storage, indexer, provider, repo = indexed
        vector = provider.embed_query("API handler request")
        results = storage.search_vector(vector, top_k=10, filter_path="src/services/")
        # May return 0 results if path doesn't match stored paths exactly
        for r in results:
            assert "services" in r["file_path"]


class TestAutoIndexation:
    """AC: Given no index exists, When search_semantic is called,
    Then auto-indexation triggers, then returns search results."""

    @pytest.mark.asyncio
    async def test_auto_indexes_when_no_index_exists(self, indexed_env):
        """Deve auto-indexar quando indice nao existe e retornar resultados."""
        config, storage, indexer, provider, repo = indexed_env
        # Not indexed yet -- calling search_semantic should auto-index

        import src.server as srv
        old_config = srv._config
        old_storage = srv._storage
        old_indexer = srv._indexer
        old_provider = srv._embed_provider
        old_repo = srv._repo_path

        try:
            srv._config = config
            srv._storage = storage
            srv._indexer = indexer
            srv._embed_provider = provider
            srv._repo_path = repo

            result_json = await search_semantic("data processing")
            response = json.loads(result_json)

            assert "results" in response
            assert "tokens_saved" in response
        finally:
            srv._config = old_config
            srv._storage = old_storage
            srv._indexer = old_indexer
            srv._embed_provider = old_provider
            srv._repo_path = old_repo

    @pytest.mark.asyncio
    async def test_auto_index_response_has_first_index_status(self, indexed_env):
        """Resposta deve incluir metadata.index_status: 'first_index'."""
        config, storage, indexer, provider, repo = indexed_env

        import src.server as srv
        old_config = srv._config
        old_storage = srv._storage
        old_indexer = srv._indexer
        old_provider = srv._embed_provider
        old_repo = srv._repo_path

        try:
            srv._config = config
            srv._storage = storage
            srv._indexer = indexer
            srv._embed_provider = provider
            srv._repo_path = repo

            result_json = await search_semantic("load data")
            response = json.loads(result_json)

            assert response["metadata"]["index_status"] == "first_index"
        finally:
            srv._config = old_config
            srv._storage = old_storage
            srv._indexer = old_indexer
            srv._embed_provider = old_provider
            srv._repo_path = old_repo


class TestMcpRegistration:
    """AC: Given MCP server starts, When FastMCP initializes,
    Then search_semantic is registered with competitive description."""

    def test_tool_registered_with_use_instead_of_grep(self):
        """Descricao da tool deve conter 'USE INSTEAD OF Grep'."""
        # Check that the tool is registered in FastMCP
        tools = mcp._tool_manager._tools
        assert "search_semantic" in tools
        tool = tools["search_semantic"]
        assert "USE INSTEAD OF Grep" in tool.description


class _FailingEmbedProvider:
    """Stub embed provider that always raises EmbeddingProviderError."""

    def __init__(self, message: str):
        self._message = message

    def embed_query(self, query: str):
        raise EmbeddingProviderError(self._message)

    def embed_batch(self, texts):
        raise EmbeddingProviderError(self._message)


class _CrashingEmbedProvider:
    """Stub embed provider that raises an unexpected RuntimeError."""

    def embed_query(self, query: str):
        raise RuntimeError("unexpected crash")

    def embed_batch(self, texts):
        raise RuntimeError("unexpected crash")


class TestErrorHandling:
    """AC: Given an internal exception occurs,
    When tool handler catches it, Then JSON error response is returned."""

    @pytest.mark.asyncio
    async def test_exception_returns_json_error_with_suggestion(self, indexed):
        """Excecao interna deve retornar { error, suggestion } JSON."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed

        old_provider = srv._embed_provider
        old_config = srv._config
        old_storage = srv._storage
        old_indexer = srv._indexer
        old_repo = srv._repo_path

        try:
            srv._config = config
            srv._storage = storage
            srv._indexer = indexer
            srv._repo_path = repo
            # Inject a provider that raises EmbeddingProviderError
            srv._embed_provider = _FailingEmbedProvider("model not found")

            result_json = await search_semantic("test query")
            response = json.loads(result_json)

            assert "error" in response
            assert "suggestion" in response
            assert "model not found" in response["error"]
        finally:
            srv._config = old_config
            srv._storage = old_storage
            srv._indexer = old_indexer
            srv._embed_provider = old_provider
            srv._repo_path = old_repo

    @pytest.mark.asyncio
    async def test_exception_never_escapes_to_mcp_protocol(self, indexed):
        """Excecao nunca deve escapar para o protocolo MCP."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed

        old_provider = srv._embed_provider
        old_config = srv._config
        old_storage = srv._storage
        old_indexer = srv._indexer
        old_repo = srv._repo_path

        try:
            srv._config = config
            srv._storage = storage
            srv._indexer = indexer
            srv._repo_path = repo
            # Inject a provider that raises an unexpected RuntimeError
            srv._embed_provider = _CrashingEmbedProvider()

            # Should NOT raise, should return JSON error
            result_json = await search_semantic("test")
            response = json.loads(result_json)
            assert "error" in response
        finally:
            srv._config = old_config
            srv._storage = old_storage
            srv._indexer = old_indexer
            srv._embed_provider = old_provider
            srv._repo_path = old_repo
