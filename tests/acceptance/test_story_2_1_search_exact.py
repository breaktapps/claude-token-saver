"""
Acceptance tests for Story 2.1: search_exact MCP Tool.

Validates exact name search via FTS with enriched results, parent_name linking,
filters, token savings, performance, and MCP registration.

FRs: FR2 (exact search with metadata)
NFRs: NFR2 (<100ms)
"""

import json
import shutil
import time
from pathlib import Path

import pytest
import pytest_asyncio

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.indexer import Indexer
from src.server import search_exact, mcp
from src.storage import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_config(**overrides) -> Config:
    return Config(embedding_mode="lite", **overrides)


def _create_test_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    src = repo / "src"
    src.mkdir()
    shutil.copy(FIXTURES / "python" / "data_processor.py", src / "data_processor.py")

    dart = repo / "lib"
    dart.mkdir()
    shutil.copy(FIXTURES / "dart" / "audio_service.dart", dart / "audio_service.dart")

    return repo


@pytest_asyncio.fixture
async def indexed_env(tmp_path):
    """Create a fully indexed test environment."""
    repo = _create_test_repo(tmp_path)
    config = _make_config()
    storage = Storage(config, repo, base_path=tmp_path / "indices")
    provider = LocalFastembedProvider(config)
    indexer = Indexer(config, storage, provider, repo_path=repo)
    await indexer.reindex()

    import src.server as srv
    old = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)
    srv._config = config
    srv._storage = storage
    srv._indexer = indexer
    srv._embed_provider = provider
    srv._repo_path = repo

    yield config, storage, indexer, provider, repo

    srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path = old


class TestExactSearchResults:
    """AC: Given an indexed repository with a function named parse_audio_stream,
    When search_exact is called, Then matching chunks are returned via FTS."""

    @pytest.mark.asyncio
    async def test_exact_search_finds_function_by_name(self, indexed_env):
        """search_exact deve encontrar funcao pelo nome exato."""
        result_json = await search_exact("load_data")
        response = json.loads(result_json)
        assert "results" in response
        assert len(response["results"]) > 0
        names = [r["name"] for r in response["results"]]
        assert any("load_data" in n for n in names)

    @pytest.mark.asyncio
    async def test_results_have_required_fields(self, indexed_env):
        """Cada resultado deve conter: file_path, name, chunk_type, line_start,
        line_end, content, score, language, stale."""
        result_json = await search_exact("DataProcessor")
        response = json.loads(result_json)
        assert len(response["results"]) > 0

        required = ["file_path", "name", "chunk_type", "line_start", "line_end",
                     "content", "score", "language", "stale"]
        for r in response["results"]:
            for field in required:
                assert field in r, f"Missing field: {field}"


class TestClassAndMethodOrdering:
    """AC: Given search for AudioService with class and methods,
    Then class outline appears first, methods follow with parent_name."""

    @pytest.mark.asyncio
    async def test_class_outline_appears_first(self, indexed_env):
        """Chunk outline da classe deve aparecer primeiro nos resultados."""
        result_json = await search_exact("AudioService")
        response = json.loads(result_json)
        results = response["results"]
        if len(results) >= 2:
            # First result should be class type
            class_results = [r for r in results if r["chunk_type"] == "class"]
            if class_results:
                # Find index of first class result
                first_class_idx = next(i for i, r in enumerate(results) if r["chunk_type"] == "class")
                method_indices = [i for i, r in enumerate(results) if r["chunk_type"] == "method"]
                if method_indices:
                    assert first_class_idx < min(method_indices)

    @pytest.mark.asyncio
    async def test_methods_include_parent_name(self, indexed_env):
        """Chunks de metodo devem incluir parent_name linkando a classe."""
        result_json = await search_exact("DataProcessor")
        response = json.loads(result_json)
        method_results = [r for r in response["results"] if r["chunk_type"] == "method"]
        for mr in method_results:
            assert "parent_name" in mr


class TestExactSearchFilters:
    """AC: Given search_exact with filter_ext='.dart',
    Then only .dart chunks are returned."""

    @pytest.mark.asyncio
    async def test_filter_by_extension(self, indexed_env):
        """filter_ext='.py' deve retornar apenas chunks de arquivos Python."""
        result_json = await search_exact("def", filter_ext=".py")
        response = json.loads(result_json)
        for r in response["results"]:
            assert r["file_path"].endswith(".py")


class TestExactSearchTokenSavings:
    """AC: Given results are returned, Then tokens_saved is included."""

    @pytest.mark.asyncio
    async def test_tokens_saved_with_file_size_estimation(self, indexed_env):
        """Resposta deve incluir tokens_saved com estimativa baseada em file size."""
        result_json = await search_exact("load_data")
        response = json.loads(result_json)
        assert "tokens_saved" in response
        ts = response["tokens_saved"]
        assert "without_plugin" in ts
        assert "with_plugin" in ts
        assert "saved" in ts
        assert "reduction_pct" in ts

    @pytest.mark.asyncio
    async def test_metadata_includes_query_time(self, indexed_env):
        """metadata deve incluir query_time_ms."""
        result_json = await search_exact("transform_records")
        response = json.loads(result_json)
        assert "metadata" in response
        assert "query_time_ms" in response["metadata"]
        assert isinstance(response["metadata"]["query_time_ms"], int)


class TestExactSearchPerformance:
    """AC: Given an indexed repository, When search_exact executes,
    Then response time is < 100ms."""

    @pytest.mark.asyncio
    async def test_response_under_100ms(self, indexed_env):
        """search_exact deve responder em < 100ms (FTS sem embedding)."""
        start = time.perf_counter()
        result_json = await search_exact("load_data")
        elapsed_ms = (time.perf_counter() - start) * 1000

        response = json.loads(result_json)
        assert "results" in response
        assert elapsed_ms < 100, f"search_exact took {elapsed_ms:.1f}ms, expected < 100ms"


class TestExactSearchMcpRegistration:
    """AC: Given MCP server is initialized, Then search_exact has
    description containing 'USE INSTEAD OF Grep'."""

    def test_tool_description_contains_use_instead_of_grep(self):
        """Descricao deve conter 'USE INSTEAD OF Grep' para exact name lookups."""
        tools = mcp._tool_manager._tools
        assert "search_exact" in tools
        tool = tools["search_exact"]
        assert "USE INSTEAD OF Grep" in tool.description
