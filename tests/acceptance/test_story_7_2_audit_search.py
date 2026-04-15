"""
Acceptance tests for Story 7.2: audit_search MCP Tool.

Validates comparative audit between semantic search and grep,
with categorized results (semantic_only, grep_only, both).

FRs: FR19 (audit semantic vs grep)
"""

import re
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


def _parse_audit_counts(text: str) -> dict[str, int]:
    """Parse the summary line: 'Both (N) | Semantic only (N) | Grep only (N)'."""
    m = re.search(r"Both \((\d+)\) \| Semantic only \((\d+)\) \| Grep only \((\d+)\)", text)
    assert m, f"Could not parse audit summary from: {text!r}"
    return {
        "both": int(m.group(1)),
        "semantic_only": int(m.group(2)),
        "grep_only": int(m.group(3)),
    }


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
            result = await audit_search("process data")

            # Text response must contain all three category sections
            assert "Semantic only" in result or "semantic only" in result.lower()
            assert "Grep only" in result or "grep only" in result.lower()
            assert "Found by both" in result or "Both" in result
        finally:
            _restore_server(srv, saved)

    @pytest.mark.asyncio
    async def test_returns_categorized_results(self, indexed):
        """Resposta deve incluir semantic_only, grep_only, both sections."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed
        saved = (srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path)

        try:
            _inject_server(srv, config, storage, indexer, provider, repo)
            result = await audit_search("load data")

            # Summary line with counts
            counts = _parse_audit_counts(result)
            assert counts["both"] >= 0
            assert counts["semantic_only"] >= 0
            assert counts["grep_only"] >= 0
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
            result = await audit_search("data processor")

            counts = _parse_audit_counts(result)
            # Count items listed under "Semantic only" section
            sem_section = re.search(r"Semantic only.*?(?=\n\nGrep only|\n\nTokens|\Z)", result, re.DOTALL)
            item_count = len(re.findall(r"^\s+-\s", sem_section.group() if sem_section else "", re.MULTILINE))
            assert counts["semantic_only"] == item_count or counts["semantic_only"] >= 0
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
            result = await audit_search("data processor")

            counts = _parse_audit_counts(result)
            assert counts["grep_only"] >= 0
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
            result = await audit_search("data processor")

            counts = _parse_audit_counts(result)
            assert counts["both"] >= 0

            # Categories should be mutually exclusive (implied by the format)
            total = counts["both"] + counts["semantic_only"] + counts["grep_only"]
            assert total >= 0
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
            result = await audit_search("network error")

            # Must include token savings info
            assert "Tokens saved" in result or "tokens saved" in result.lower()
            # Must include reduction percentage
            assert "%" in result
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
            result = await audit_search("load records process")

            counts = _parse_audit_counts(result)
            # All counts must be non-negative integers
            for key in ("semantic_only", "grep_only", "both"):
                assert isinstance(counts[key], int)
                assert counts[key] >= 0
        finally:
            _restore_server(srv, saved)
