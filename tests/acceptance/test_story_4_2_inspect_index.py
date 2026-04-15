"""
Acceptance tests for Story 4.2: inspect_index MCP Tool.

Validates global stats, per-file inspection, stale detection,
cumulative metrics display, no-index error, and MCP registration.

FRs: FR20 (global index stats), FR21 (per-file chunk inspection), FR22 (stale indicator)
"""

import contextlib
import hashlib
import json
import re
from pathlib import Path

import pytest

import src.server as server_module
from src.metrics import _save_metrics
from src.storage import Storage


@contextlib.contextmanager
def _inject_server_state(config, storage, repo_path):
    """Context manager: inject real components into server globals, restore after."""
    old_config = server_module._config
    old_storage = server_module._storage
    old_indexer = server_module._indexer
    old_provider = server_module._embed_provider
    old_repo = server_module._repo_path

    server_module._config = config
    server_module._storage = storage
    server_module._indexer = None
    server_module._embed_provider = None
    server_module._repo_path = repo_path
    try:
        yield
    finally:
        server_module._config = old_config
        server_module._storage = old_storage
        server_module._indexer = old_indexer
        server_module._embed_provider = old_provider
        server_module._repo_path = old_repo


def _make_storage_with_chunks(tmp_path, chunks: list[dict], config=None):
    """Helper: create Storage backed by tmp_path and upsert chunks."""
    if config is None:
        from src.config import Config
        config = Config()
    storage = Storage(config, tmp_path / "repo", base_path=tmp_path / "indices")
    if chunks:
        storage.upsert(chunks)
    return storage


def _fake_chunk(file_path="src/a.py", language="python", chunk_type="function",
                name="foo", line_start=1, line_end=10, file_hash="abc123"):
    from src.config import Config
    dim = 384
    return {
        "file_path": file_path,
        "chunk_type": chunk_type,
        "name": name,
        "line_start": line_start,
        "line_end": line_end,
        "content": f"def {name}(): pass",
        "embedding": [0.0] * dim,
        "file_hash": file_hash,
        "language": language,
        "parent_name": "",
        "calls_json": "[]",
        "outline_json": "[]",
    }


@pytest.fixture()
def mock_components(tmp_path):
    """Inject real Storage and Config into server globals for controlled testing."""
    from src.config import Config
    config = Config()
    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    storage = _make_storage_with_chunks(tmp_path, [
        _fake_chunk("src/a.py", language="python", name="foo", file_hash="hash_a"),
        _fake_chunk("src/b.py", language="dart", name="bar", file_hash="hash_b"),
    ])

    with _inject_server_state(config, storage, repo_path):
        yield storage, repo_path


class TestGlobalStats:
    """AC: Given an indexed repository, When inspect_index() without args,
    Then global stats are returned as human-readable text."""

    @pytest.mark.asyncio
    async def test_returns_total_files(self, mock_components):
        """Deve incluir contagem de arquivos na resposta."""
        result = await server_module.inspect_index()
        assert "2 files" in result

    @pytest.mark.asyncio
    async def test_returns_total_chunks(self, mock_components):
        """Deve incluir contagem de chunks na resposta."""
        result = await server_module.inspect_index()
        assert "2 chunks" in result

    @pytest.mark.asyncio
    async def test_returns_languages_with_counts(self, mock_components):
        """Deve incluir contagem de linguagens na resposta."""
        result = await server_module.inspect_index()
        assert "python" in result.lower()
        assert "dart" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_index_size_bytes(self, mock_components):
        """Deve incluir tamanho do índice na resposta."""
        result = await server_module.inspect_index()
        # Format is like "0.1MB" or similar size indicator
        assert "MB" in result or "KB" in result or "0.0MB" in result

    @pytest.mark.asyncio
    async def test_returns_tokens_saved_total(self, mock_components):
        """Deve incluir tokens salvos na resposta."""
        result = await server_module.inspect_index()
        assert "Tokens saved" in result or "tokens saved" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_total_queries(self, mock_components):
        """Deve incluir total de queries na resposta."""
        result = await server_module.inspect_index()
        assert "queries" in result.lower() or "0 queries" in result

    @pytest.mark.asyncio
    async def test_returns_stale_files_count(self, mock_components):
        """Deve incluir contagem de arquivos stale na resposta."""
        result = await server_module.inspect_index()
        assert "Stale files:" in result or "stale" in result.lower()


class TestPerFileInspection:
    """AC: Given inspect_index(file_path=...), Then lists all chunks
    for that file with metadata."""

    @pytest.mark.asyncio
    async def test_lists_chunks_for_specific_file(self, mock_components):
        """Deve listar chunks do arquivo com name, chunk_type, line ranges."""
        result = await server_module.inspect_index(file_path="src/a.py")
        assert "src/a.py" in result
        assert "foo" in result
        # Should contain line range info like L1-10
        assert re.search(r"L\d+-\d+", result)


class TestStaleFilesReporting:
    """AC: Given 5 files modified since indexation,
    Then stale_files_count: 5 and each stale file listed."""

    @pytest.mark.asyncio
    async def test_stale_count_matches_modified_files(self, tmp_path):
        """stale_files_count deve corresponder ao numero de arquivos modificados."""
        from src.config import Config
        config = Config()
        repo_path = tmp_path / "repo"
        repo_path.mkdir(parents=True)

        # Index files with hash "old_hash"
        chunks = [
            _fake_chunk(f"src/file{i}.py", file_hash="old_hash", name=f"fn{i}")
            for i in range(3)
        ]
        storage = _make_storage_with_chunks(tmp_path, chunks)

        with _inject_server_state(config, storage, repo_path):
            result = await server_module.inspect_index()

        # All 3 files are stale because "old_hash" != current hash (files don't exist)
        assert "Stale files: 3" in result

    @pytest.mark.asyncio
    async def test_stale_files_listed_with_paths(self, tmp_path):
        """Cada arquivo stale deve ser listado com seu path."""
        from src.config import Config
        config = Config()
        repo_path = tmp_path / "repo"
        repo_path.mkdir(parents=True)

        chunks = [
            _fake_chunk("src/stale1.py", file_hash="old_hash1", name="fn1"),
            _fake_chunk("src/stale2.py", file_hash="old_hash2", name="fn2"),
        ]
        storage = _make_storage_with_chunks(tmp_path, chunks)

        with _inject_server_state(config, storage, repo_path):
            result = await server_module.inspect_index()

        assert "src/stale1.py" in result
        assert "src/stale2.py" in result


class TestCumulativeMetricsDisplay:
    """AC: Given cumulative metrics in metrics.json,
    Then inspect_index includes them."""

    @pytest.mark.asyncio
    async def test_includes_persisted_metrics(self, tmp_path):
        """tokens_saved_total e total_queries devem vir de metrics.json."""
        from src.config import Config
        config = Config()
        repo_path = tmp_path / "repo"
        repo_path.mkdir(parents=True)

        storage = _make_storage_with_chunks(tmp_path, [
            _fake_chunk("src/a.py", name="foo"),
        ])

        # Write known metrics to the index
        _save_metrics(storage.index_path, {
            "total_saved": 99999,
            "total_queries": 77,
            "last_updated": "2026-04-14T12:00:00",
        })

        with _inject_server_state(config, storage, repo_path):
            result = await server_module.inspect_index()

        assert "99,999" in result or "99999" in result
        assert "77" in result


class TestNoIndexError:
    """AC: Given no index exists, When inspect_index called,
    Then error with suggestion is returned."""

    @pytest.mark.asyncio
    async def test_no_index_returns_error(self, tmp_path):
        """Deve retornar { error: 'No index found', suggestion: '...' }."""
        from src.config import Config
        config = Config()
        repo_path = tmp_path / "repo"
        repo_path.mkdir(parents=True)

        # Storage with NO chunks
        storage = _make_storage_with_chunks(tmp_path, [])

        with _inject_server_state(config, storage, repo_path):
            result = json.loads(await server_module.inspect_index())

        assert "error" in result
        assert result["error"] == "No index found"
        assert "suggestion" in result


class TestInspectIndexMcpRegistration:
    """AC: Given MCP server initialized, Then inspect_index description
    explains both modes."""

    def test_description_explains_global_and_per_file(self):
        """Descricao deve explicar modos global stats e per-file inspection."""
        tool_fn = server_module.inspect_index
        # FastMCP stores description on the wrapped function or its metadata
        # Check docstring and module-level description
        description = getattr(tool_fn, "__doc__", "") or ""
        # The description is set at @mcp.tool(description=...) level
        # Verify the function exists and is callable
        import asyncio
        assert callable(tool_fn)
        # Verify the mcp tools include inspect_index
        tools = server_module.mcp._tool_manager._tools
        assert "inspect_index" in tools
        tool_desc = tools["inspect_index"].description
        assert "global" in tool_desc.lower() or "stats" in tool_desc.lower()
        assert "file_path" in tool_desc or "per-file" in tool_desc.lower() or "file" in tool_desc.lower()
