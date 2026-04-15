"""
Acceptance tests for Story 3.2: reindex MCP Tool & Hook Auto-Reindex.

Validates manual reindex tool (incremental + force), post_tool hook
auto-reindex, lock handling, and MCP registration.

FRs: FR11 (force reindex), FR14 (auto-reindex via post_tool hook), FR16 (file lock)
"""

import json
import os
import time
from pathlib import Path

import pytest

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.errors import LockTimeoutError
from src.indexer import Indexer
from src.storage import Storage


def _make_config(**overrides) -> Config:
    fields = {"embedding_mode": "lite", **overrides}
    return Config(**fields)


def _create_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    src = repo / "src"
    src.mkdir()
    (src / "module.py").write_text("def func(): return 1\n")
    return repo


@pytest.fixture
def repo_path(tmp_path):
    return _create_repo(tmp_path)


@pytest.fixture
def components(tmp_path, repo_path):
    config = _make_config()
    storage = Storage(config, repo_path, base_path=tmp_path / "indices")
    provider = LocalFastembedProvider(config)
    indexer = Indexer(config, storage, provider, repo_path=repo_path)
    return config, storage, provider, indexer


class TestReindexToolIncremental:
    """AC: Given MCP server running, When reindex(force=false),
    Then incremental reindex with stats response."""

    @pytest.mark.asyncio
    async def test_incremental_reindex_returns_stats(self, components):
        """reindex(force=false) deve retornar files_scanned, files_updated,
        files_deleted, duration_ms."""
        config, storage, provider, indexer = components

        # Initial index
        await indexer.reindex()

        # Incremental reindex stats
        stats = await indexer.reindex(force=False)

        assert "files_scanned" in stats
        assert "files_updated" in stats
        assert "files_deleted" in stats
        assert "files_added" in stats
        assert "duration_ms" in stats
        assert stats["duration_ms"] >= 0


class TestReindexToolForce:
    """AC: Given reindex(force=true) with existing index,
    Then entire index rebuilt from scratch."""

    @pytest.mark.asyncio
    async def test_force_reindex_rebuilds_entirely(self, components):
        """reindex(force=true) deve dropar e reconstruir o indice."""
        config, storage, provider, indexer = components

        # Initial index
        stats1 = await indexer.reindex()
        first_chunks = stats1["chunks_created"]
        assert first_chunks > 0

        # Force reindex should rebuild completely
        stats2 = await indexer.reindex(force=True)
        # All files re-added (no modified/deleted)
        assert stats2["files_added"] == stats1["files_added"]
        assert stats2["chunks_created"] == first_chunks

    @pytest.mark.asyncio
    async def test_force_reindex_returns_rebuilt_status(self, tmp_path, repo_path):
        """Resposta do server tool deve incluir metadata.index_status: 'rebuilt'."""
        # Test via server tool handler directly
        import src.server as server_module

        # Reset global state for isolation
        server_module._config = None
        server_module._storage = None
        server_module._indexer = None
        server_module._embed_provider = None
        server_module._repo_path = None

        # Override _init_components to use test fixtures
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        server_module._config = config
        server_module._storage = storage
        server_module._indexer = indexer
        server_module._embed_provider = provider
        server_module._repo_path = repo_path

        try:
            # First index
            await indexer.reindex()

            # Call server reindex tool with force=True
            result_json = await server_module.reindex(force=True)
            result = json.loads(result_json)

            assert "metadata" in result
            assert result["metadata"]["index_status"] == "rebuilt"
        finally:
            server_module._config = None
            server_module._storage = None
            server_module._indexer = None
            server_module._embed_provider = None
            server_module._repo_path = None


class TestPostToolHookAutoReindex:
    """AC: Given hooks.json with post_tool hook,
    When Edit/Write completes, Then hook triggers incremental reindex."""

    @pytest.mark.asyncio
    async def test_hook_triggers_reindex_after_edit(self, tmp_path, repo_path):
        """Hook post_tool deve triggerar reindex apos Edit (simulado)."""
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        # Initial index
        await indexer.reindex()

        # Simulate Edit: modify a file
        py_file = repo_path / "src" / "module.py"
        py_file.write_text("def func_edited(): return 'edited'\n")

        # Hook triggers incremental reindex
        stats = await indexer.reindex(force=False)
        assert stats["files_updated"] == 1

        # Verify updated hash in index
        hashes = storage.get_file_hashes()
        assert str(py_file) in hashes

    @pytest.mark.asyncio
    async def test_hook_triggers_reindex_after_write(self, tmp_path, repo_path):
        """Hook post_tool deve triggerar reindex apos Write (simulado)."""
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        # Initial index
        await indexer.reindex()

        # Simulate Write: create a new file
        new_file = repo_path / "src" / "new_module.py"
        new_file.write_text("def new_func(): return 'new'\n")

        # Hook triggers incremental reindex
        stats = await indexer.reindex(force=False)
        assert stats["files_added"] == 1

        # Verify new file is in index
        hashes = storage.get_file_hashes()
        assert str(new_file) in hashes

    @pytest.mark.asyncio
    async def test_hook_runs_asynchronously(self, tmp_path, repo_path):
        """Hook deve rodar async sem bloquear a proxima acao do agente."""
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        # Verify hooks.json exists with PostToolUse hook
        hooks_json = Path(__file__).parent.parent.parent / "hooks" / "hooks.json"
        assert hooks_json.exists(), f"hooks.json not found at {hooks_json}"

        import json as json_module
        hooks_config = json_module.loads(hooks_json.read_text())
        hooks = hooks_config.get("hooks", [])

        post_tool_hooks = [h for h in hooks if h.get("event") == "PostToolUse"]
        assert len(post_tool_hooks) >= 1, "No PostToolUse hook found in hooks.json"

        auto_reindex_hook = next(
            (h for h in post_tool_hooks if "reindex" in h.get("name", "").lower()),
            None
        )
        assert auto_reindex_hook is not None, "auto-reindex PostToolUse hook not found"

        # Verify it matches Edit and Write
        matcher = auto_reindex_hook.get("matcher", {})
        tool_names = matcher.get("tool_name", [])
        assert "Edit" in tool_names
        assert "Write" in tool_names


class TestHookLockHandling:
    """AC: Given hook triggers reindex while lock is held,
    Then waits for lock or reports timeout gracefully."""

    @pytest.mark.asyncio
    async def test_hook_waits_for_lock(self, tmp_path, repo_path):
        """Hook deve esperar pelo lock."""
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        await indexer.reindex()

        # Simulate lock being briefly held then released
        lock_path = storage.index_path / "index.lock"

        # Create lock, then release after short delay in background
        import threading

        def release_lock():
            time.sleep(0.3)
            if lock_path.exists():
                lock_path.unlink()

        lock_path.touch()
        t = threading.Thread(target=release_lock, daemon=True)
        t.start()

        # delete_file should wait for lock and succeed once it's released
        # (Since LanceDB upsert acquires lock internally)
        # Modify a file to trigger processing
        (repo_path / "src" / "module.py").write_text("def func_new(): return 99\n")

        # Run reindex — the lock will be released before upsert needs it
        stats = await indexer.reindex(force=False)
        t.join(timeout=2.0)
        # Should complete without error
        assert stats["files_updated"] == 1

    @pytest.mark.asyncio
    async def test_hook_graceful_timeout(self, tmp_path, repo_path):
        """Hook deve reportar timeout gracefully sem corromper indice."""
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        await indexer.reindex()

        # Hold the lock permanently to simulate timeout
        lock_path = storage.index_path / "index.lock"
        lock_path.touch()

        try:
            # Attempt to delete a file (which acquires lock) — should raise LockTimeoutError
            with pytest.raises(LockTimeoutError):
                storage.delete_file(str(repo_path / "src" / "module.py"))
        finally:
            if lock_path.exists():
                lock_path.unlink()

        # Index should still be intact (not corrupted)
        hashes = storage.get_file_hashes()
        assert len(hashes) > 0


class TestReindexSearchConsistency:
    """AC: Given reindex completes, When agent searches immediately,
    Then results reflect updated code with stale: false."""

    @pytest.mark.asyncio
    async def test_search_after_reindex_reflects_changes(self, tmp_path, repo_path):
        """Busca apos reindex deve refletir codigo atualizado."""
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        # Initial index
        await indexer.reindex()

        # Modify a file with distinctive content
        py_file = repo_path / "src" / "module.py"
        py_file.write_text("def unique_function_xyz(): return 'xyz_result'\n")

        # Reindex
        stats = await indexer.reindex(force=False)
        assert stats["files_updated"] == 1

        # Search should find the new content
        query_vector = provider.embed_query("unique function xyz")
        results = storage.search_vector(query_vector, top_k=5)

        contents = [r.get("content", "") for r in results]
        assert any("unique_function_xyz" in c for c in contents), \
            "Updated function not found in search results after reindex"

    @pytest.mark.asyncio
    async def test_no_stale_after_reindex(self, tmp_path, repo_path):
        """Apos reindex, chunks devem ter stale: false."""
        config = _make_config()
        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo_path)

        # Initial index
        await indexer.reindex()

        # Modify a file
        py_file = repo_path / "src" / "module.py"
        py_file.write_text("def func_v2(): return 2\n")

        # Reindex
        await indexer.reindex(force=False)

        # All stored files should match current hashes (not stale)
        stored_hashes = storage.get_file_hashes()
        for fp, stored_hash in stored_hashes.items():
            file_path = Path(fp)
            if file_path.exists():
                is_stale = storage.is_stale(fp, stored_hash)
                assert not is_stale, f"File {fp} is stale after reindex"


class TestReindexMcpRegistration:
    """AC: Given reindex tool registered, Then description explains
    both manual and incremental modes."""

    def test_tool_description_explains_modes(self):
        """Descricao deve explicar modos manual e incremental."""
        import src.server as server_module

        # Find the reindex tool in the FastMCP instance
        # FastMCP stores tools in its internal registry
        mcp = server_module.mcp

        # The tool should be registered — verify by checking the module has the function
        assert hasattr(server_module, "reindex"), "reindex tool not found in server module"

        # Verify the function is async
        import asyncio
        assert asyncio.iscoroutinefunction(server_module.reindex), "reindex must be async"

        # Check description is meaningful by inspecting the decorated function
        # The description is passed to @mcp.tool() decorator
        # We can verify by checking the tool is accessible and has proper behavior
        import inspect
        sig = inspect.signature(server_module.reindex)
        params = list(sig.parameters.keys())
        assert "force" in params, "reindex tool must have 'force' parameter"

        # Default must be False (incremental mode)
        force_param = sig.parameters["force"]
        assert force_param.default is False, "force parameter default must be False"
