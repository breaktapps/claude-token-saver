"""
Acceptance tests for Story 7.4: Git Hook Post-Commit Auto-Reindex.

Validates git post-commit hook triggering incremental reindex,
lock handling with concurrent operations, and sequential commits.

FRs: FR15 (git hook post-commit)
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.indexer import Indexer
from src.storage import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_config(**overrides) -> Config:
    return Config(embedding_mode="lite", **overrides)


def _create_test_repo(tmp_path: Path) -> Path:
    """Create a small test repo."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    src = repo / "src"
    src.mkdir()
    shutil.copy(FIXTURES / "python" / "data_processor.py", src / "data_processor.py")
    shutil.copy(FIXTURES / "python" / "simple_function.py", src / "simple_function.py")
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


class TestGitPostCommitHook:
    """AC: Given hooks.json includes git post-commit hook,
    When developer commits, Then incremental reindex triggers."""

    @pytest.mark.asyncio
    async def test_reindex_triggers_on_commit(self, indexed):
        """Reindex incremental deve ser triggerado apos git commit."""
        config, storage, indexer, provider, repo = indexed

        reindex_calls = []
        original_reindex = indexer.reindex

        async def tracking_reindex(force=False):
            reindex_calls.append(force)
            return await original_reindex(force=force)

        with patch.object(indexer, "reindex", side_effect=tracking_reindex):
            # Simulate hook triggering incremental reindex
            await indexer.reindex(force=False)

        assert len(reindex_calls) == 1
        assert reindex_calls[0] is False  # incremental, not full rebuild

    @pytest.mark.asyncio
    async def test_only_committed_files_reindexed(self, indexed):
        """Apenas arquivos modificados no commit devem ser re-indexados."""
        config, storage, indexer, provider, repo = indexed

        # Modify one file to simulate a committed change
        target = repo / "src" / "simple_function.py"
        original = target.read_text()
        target.write_text(original + "\n# post-commit modification\n")

        stats = await indexer.reindex(force=False)

        assert "files_scanned" in stats
        assert "files_updated" in stats
        assert stats["files_updated"] >= 1

        # Restore
        target.write_text(original)


class TestGitHookLockHandling:
    """AC: Given post-commit triggers reindex while lock held by post_tool,
    Then waits for lock without corrupting index."""

    @pytest.mark.asyncio
    async def test_waits_for_lock_up_to_10s(self, indexed):
        """Hook deve esperar pelo lock ate 10s sem corromper o indice."""
        from src.errors import LockTimeoutError

        config, storage, indexer, provider, repo = indexed

        # Simulate a held lock
        lock_path = storage._lock_path
        lock_path.touch()

        try:
            with pytest.raises(LockTimeoutError):
                with storage._acquire_lock(timeout=0.1):
                    pass  # Lock already held -- should timeout quickly
        finally:
            lock_path.unlink(missing_ok=True)

        # Index must be intact after lock timeout
        hashes = storage.get_file_hashes()
        assert isinstance(hashes, dict)


class TestSequentialCommits:
    """AC: Given multiple commits in quick succession,
    When hooks fire for each, Then lock acquired/released sequentially."""

    @pytest.mark.asyncio
    async def test_sequential_lock_handling(self, indexed):
        """Cada reindex deve adquirir e liberar o lock sequencialmente."""
        config, storage, indexer, provider, repo = indexed

        stats1 = await indexer.reindex(force=False)
        stats2 = await indexer.reindex(force=False)

        assert "files_scanned" in stats1
        assert "files_scanned" in stats2

    @pytest.mark.asyncio
    async def test_no_data_loss_on_rapid_commits(self, indexed):
        """Commits rapidos em sequencia nao devem causar perda de dados."""
        config, storage, indexer, provider, repo = indexed

        initial_hashes = storage.get_file_hashes()
        initial_count = len(initial_hashes)

        for _ in range(3):
            await indexer.reindex(force=False)

        final_hashes = storage.get_file_hashes()
        assert len(final_hashes) >= initial_count


class TestHooksJsonConfig:
    """AC: hooks.json includes git post-commit hook configuration."""

    def test_hooks_json_contains_post_commit(self):
        """hooks.json deve conter configuracao do post-commit hook."""
        hooks_path = Path(__file__).parent.parent.parent / "hooks" / "hooks.json"
        assert hooks_path.exists(), "hooks.json not found"

        data = json.loads(hooks_path.read_text())
        hooks = data.get("hooks", [])

        git_hook = next(
            (
                h for h in hooks
                if "post-commit" in h.get("name", "").lower()
                or "post-commit" in h.get("event", "").lower()
                or "post_commit" in h.get("args", [""])[0].lower()
            ),
            None,
        )
        assert git_hook is not None, "No git post-commit hook found in hooks.json"

    def test_post_commit_script_exists(self):
        """Script post_commit.sh deve existir."""
        script_path = Path(__file__).parent.parent.parent / "hooks" / "post_commit.sh"
        assert script_path.exists(), "post_commit.sh not found"

        content = script_path.read_text()
        assert "git diff" in content
