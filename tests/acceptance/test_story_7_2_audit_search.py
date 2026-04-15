"""
Acceptance tests for Story 7.2: audit_search MCP Tool.

Validates comparative audit between semantic search and grep,
with categorized results (semantic_only, grep_only, both).

FRs: FR19 (audit semantic vs grep)
"""

import json
import shutil
from pathlib import Path

import pytest
import pytest_asyncio

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.indexer import Indexer
from src.server import audit_search, mcp
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


@pytest_asyncio.fixture
async def indexed(tmp_path):
    """Create and index a test environment."""
    repo = _create_test_repo(tmp_path)
    config = _make_config()
    storage = Storage(config, repo, base_path=tmp_path / "indices")
    provider = LocalFastembedProvider(config)
    indexer = Indexer(config, storage, provider, repo_path=repo)
    await indexer.reindex()
    return config, storage, indexer, provider, repo


def _inject_server(srv, config, storage, indexer, provider, repo):
    srv._config = config
    srv._storage = storage
    srv._indexer = indexer
    srv._embed_provider = provider
    srv._repo_path = repo


def _restore_server(srv, saved):
    srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path = saved


class TestAuditSearchExecution:
    """AC: Given an indexed repository, When audit_search is called,
    Then both semantic and grep-equivalent searches are run."""

    @pytest.mark.asyncio
    async def test_runs_both_semantic_and_grep(self, indexed):
        """audit_search deve executar busca semantica E grep-equivalent."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await audit_search("process data")
            response = json.loads(result_json)

            # Both keys must be present — even if empty lists
            assert "semantic_only" in response
            assert "grep_only" in response
            assert "both" in response
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_returns_categorized_results(self, indexed):
        """Resposta deve incluir semantic_only, grep_only, both."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await audit_search("load data")
            response = json.loads(result_json)

            assert isinstance(response["semantic_only"], list)
            assert isinstance(response["grep_only"], list)
            assert isinstance(response["both"], list)

            # Every item must have file_path and name
            for category in ("semantic_only", "grep_only", "both"):
                for item in response[category]:
                    assert "file_path" in item
                    assert "name" in item
        finally:
            _restore_server(srv, saved)


class TestAuditResultCategorization:
    """AC: Given semantic finds 5, grep finds 3, 2 overlap,
    Then semantic_only: 3, grep_only: 1, both: 2."""

    @pytest.mark.asyncio
    async def test_correct_semantic_only_count(self, indexed):
        """semantic_only deve contar resultados exclusivos do semantico."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await audit_search("data processor")
            response = json.loads(result_json)

            counts = response["counts"]
            assert "semantic_only" in counts
            assert counts["semantic_only"] == len(response["semantic_only"])
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_correct_grep_only_count(self, indexed):
        """grep_only deve contar resultados exclusivos do grep."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await audit_search("data processor")
            response = json.loads(result_json)

            counts = response["counts"]
            assert "grep_only" in counts
            assert counts["grep_only"] == len(response["grep_only"])
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_correct_both_count(self, indexed):
        """both deve contar resultados encontrados por ambos."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await audit_search("data processor")
            response = json.loads(result_json)

            counts = response["counts"]
            assert "both" in counts
            assert counts["both"] == len(response["both"])

            # Verify no item can appear in both 'both' and any exclusive category
            both_keys = {(r["file_path"], r["name"]) for r in response["both"]}
            sem_only_keys = {(r["file_path"], r["name"]) for r in response["semantic_only"]}
            grep_only_keys = {(r["file_path"], r["name"]) for r in response["grep_only"]}

            assert not (both_keys & sem_only_keys), "Item in 'both' also in 'semantic_only'"
            assert not (both_keys & grep_only_keys), "Item in 'both' also in 'grep_only'"
            assert not (sem_only_keys & grep_only_keys), "Item in 'semantic_only' also in 'grep_only'"
        finally:
            _restore_server(srv, saved)


class TestAuditTokenSavings:
    """AC: Given audit results, Then tokens_saved compares both approaches."""

    @pytest.mark.asyncio
    async def test_tokens_saved_compares_approaches(self, indexed):
        """tokens_saved deve comparar ambas as abordagens."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await audit_search("network error")
            response = json.loads(result_json)

            assert "tokens_saved" in response
            ts = response["tokens_saved"]
            assert "without_plugin" in ts
            assert "with_plugin" in ts
            assert "saved" in ts
            assert "reduction_pct" in ts
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_match_counts_per_category(self, indexed):
        """Resposta deve incluir contagem de matches por categoria."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result_json = await audit_search("load records process")
            response = json.loads(result_json)

            assert "counts" in response
            counts = response["counts"]
            assert "semantic_only" in counts
            assert "grep_only" in counts
            assert "both" in counts

            # All counts must be non-negative integers
            for key in ("semantic_only", "grep_only", "both"):
                assert isinstance(counts[key], int)
                assert counts[key] >= 0
        finally:
            _restore_server(srv, saved)
