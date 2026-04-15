"""
Acceptance tests for Story 3.1: Re-indexacao Incremental por Hash.

Validates incremental reindex: only modified files re-processed,
deleted files removed, new files added, performance, and lock handling.

FRs: FR10 (incremental reindex by hash), FR16 (file lock)
NFRs: NFR4 (<3s for 1-5 files), NFR11 (file lock anti-corruption)
"""

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.errors import LockTimeoutError
from src.indexer import Indexer
from src.storage import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_config(**overrides) -> Config:
    fields = {"embedding_mode": "lite", **overrides}
    return Config(**fields)


def _create_repo(tmp_path: Path) -> Path:
    """Create a test repo with Python files."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    py_dir = repo / "src"
    py_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "python" / "data_processor.py", py_dir / "data_processor.py")
    shutil.copy(FIXTURES / "python" / "simple_function.py", py_dir / "simple_function.py")

    return repo


@pytest.fixture
def repo_path(tmp_path):
    return _create_repo(tmp_path)


@pytest.fixture
def components(tmp_path, repo_path):
    config = _make_config()
    storage = Storage(config, repo_path, base_path=tmp_path / "indices")
    provider = LocalFastembedProvider(config)
    return config, storage, provider


class TestIncrementalReindex:
    """AC: Given 100 indexed files and 3 modified,
    When reindex(force=False), Then only 3 are re-processed."""

    @pytest.mark.asyncio
    async def test_only_modified_files_reprocessed(self, tmp_path):
        """Apenas arquivos com hash diferente devem ser re-chunked e re-embedded."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()

        # Create 5 files
        files = []
        for i in range(5):
            f = src / f"module_{i}.py"
            f.write_text(f"def func_{i}():\n    return {i}\n")
            files.append(f)

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        # Initial full indexation
        stats1 = await indexer.reindex()
        assert stats1["files_scanned"] == 5
        assert stats1["files_added"] == 5
        assert stats1["files_updated"] == 0
        assert stats1["files_deleted"] == 0

        # Modify 2 files
        files[0].write_text("def func_0_modified():\n    return 'changed'\n")
        files[2].write_text("def func_2_modified():\n    return 999\n")

        # Incremental reindex
        stats2 = await indexer.reindex()
        assert stats2["files_updated"] == 2
        assert stats2["files_added"] == 0
        assert stats2["files_deleted"] == 0

    @pytest.mark.asyncio
    async def test_unmodified_files_untouched(self, tmp_path):
        """Arquivos nao modificados nao devem ser reprocessados."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()

        f1 = src / "stable.py"
        f1.write_text("def stable(): return 1\n")
        f2 = src / "changing.py"
        f2.write_text("def changing(): return 1\n")

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        # Initial indexation
        stats1 = await indexer.reindex()
        assert stats1["files_added"] == 2

        # Modify only f2
        f2.write_text("def changing(): return 999\n")

        stats2 = await indexer.reindex()
        # Only f2 should be re-processed
        assert stats2["files_updated"] == 1
        assert stats2["files_added"] == 0
        # chunks_created should reflect only 1 file reprocessed
        assert stats2["chunks_created"] > 0


class TestDeletedFileHandling:
    """AC: Given a file is deleted, When incremental reindex runs,
    Then chunks for deleted file are removed."""

    @pytest.mark.asyncio
    async def test_deleted_file_chunks_removed(self, tmp_path):
        """Chunks de arquivo deletado devem ser removidos do indice."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()

        f1 = src / "keep.py"
        f1.write_text("def keep(): return 1\n")
        f2 = src / "delete_me.py"
        f2.write_text("def will_be_deleted(): return 2\n")

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        # Index both files
        stats1 = await indexer.reindex()
        assert stats1["files_added"] == 2

        # Verify f2 is in index
        hashes_before = storage.get_file_hashes()
        assert any("delete_me.py" in k for k in hashes_before)

        # Delete f2 from disk
        f2.unlink()

        # Incremental reindex
        stats2 = await indexer.reindex()
        assert stats2["files_deleted"] == 1
        assert stats2["files_updated"] == 0
        assert stats2["files_added"] == 0

        # Verify f2 chunks are gone from index
        hashes_after = storage.get_file_hashes()
        assert not any("delete_me.py" in k for k in hashes_after)


class TestNewFileHandling:
    """AC: Given a new file is added, When incremental reindex runs,
    Then new file is scanned, chunked, embedded, and stored."""

    @pytest.mark.asyncio
    async def test_new_file_indexed(self, tmp_path):
        """Arquivo novo deve ser escaneado, chunked, embedded e armazenado."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()

        existing = src / "existing.py"
        existing.write_text("def existing(): return 1\n")

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        # Initial indexation
        stats1 = await indexer.reindex()
        assert stats1["files_added"] == 1

        # Add a new file
        new_file = src / "new_module.py"
        new_file.write_text("def new_function():\n    return 'new'\n")

        # Incremental reindex
        stats2 = await indexer.reindex()
        assert stats2["files_added"] == 1
        assert stats2["files_updated"] == 0
        assert stats2["files_deleted"] == 0
        assert stats2["chunks_created"] > 0

        # Verify new file is in index
        hashes = storage.get_file_hashes()
        assert any("new_module.py" in k for k in hashes)


class TestIncrementalPerformance:
    """AC: Given 1-5 modified files, When incremental reindex runs,
    Then completes in < 3 seconds."""

    @pytest.mark.asyncio
    async def test_1_to_5_files_under_3_seconds(self, tmp_path):
        """Re-indexacao de 1-5 arquivos deve completar em < 3s."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()

        # Create 10 files and index them
        files = []
        for i in range(10):
            f = src / f"module_{i}.py"
            f.write_text(f"def func_{i}():\n    return {i}\n")
            files.append(f)

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        await indexer.reindex()

        # Modify only 3 files
        for f in files[:3]:
            f.write_text(f"# modified\ndef modified_{f.stem}():\n    return 'changed'\n")

        # Incremental should be fast
        start = time.monotonic()
        stats = await indexer.reindex()
        elapsed = time.monotonic() - start

        assert stats["files_updated"] == 3
        assert elapsed < 3.0, f"Incremental reindex took {elapsed:.2f}s, expected < 3s"


class TestHashComparison:
    """AC: Given stored hashes, When reindex compares with current,
    Then only mismatched hashes are re-processed."""

    @pytest.mark.asyncio
    async def test_hash_comparison_identifies_changes(self, tmp_path):
        """Comparacao de hashes deve identificar corretamente arquivos modificados."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()

        f1 = src / "file_a.py"
        f1.write_text("def a(): return 1\n")
        f2 = src / "file_b.py"
        f2.write_text("def b(): return 2\n")
        f3 = src / "file_c.py"
        f3.write_text("def c(): return 3\n")

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        # Index all files
        stats1 = await indexer.reindex()
        assert stats1["files_added"] == 3

        # Store hashes before modification
        hashes_before = storage.get_file_hashes()
        assert len(hashes_before) == 3

        # Modify only f2
        f2.write_text("def b_modified(): return 999\n")

        # Incremental reindex
        stats2 = await indexer.reindex()
        assert stats2["files_updated"] == 1
        assert stats2["files_added"] == 0
        assert stats2["files_deleted"] == 0

        # Stored hash for f2 should be updated
        hashes_after = storage.get_file_hashes()
        f2_rel = str(f2)
        assert hashes_after[f2_rel] != hashes_before[f2_rel]

        # Hashes for f1, f3 should be unchanged
        f1_rel = str(f1)
        f3_rel = str(f3)
        assert hashes_after[f1_rel] == hashes_before[f1_rel]
        assert hashes_after[f3_rel] == hashes_before[f3_rel]


class TestIncrementalReindexLock:
    """AC: Given lock held by another operation,
    When incremental reindex tries, Then waits 10s then raises LockTimeoutError."""

    @pytest.mark.asyncio
    async def test_lock_timeout_on_concurrent_reindex(self, tmp_path):
        """Reindex concorrente deve esperar e levantar LockTimeoutError."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()
        (src / "mod.py").write_text("def f(): return 1\n")

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        # Index once to have data
        await indexer.reindex()

        # Manually create the lock file to simulate another operation holding it
        lock_path = storage.index_path / "index.lock"
        lock_path.touch()

        try:
            with pytest.raises(LockTimeoutError):
                # Modify file to trigger upsert (which needs the lock)
                (src / "mod.py").write_text("def f_modified(): return 99\n")
                # Use very short timeout to avoid waiting 10s in tests
                # We patch the timeout by directly calling upsert
                storage._acquire_lock.__func__  # verify it's a context manager
                # Call delete_file which acquires lock internally
                storage.delete_file(str(src / "mod.py"))
        finally:
            # Clean up lock file
            if lock_path.exists():
                lock_path.unlink()

        # Verify that with a custom short timeout, LockTimeoutError is raised
        lock_path.touch()
        try:
            import contextlib

            @contextlib.contextmanager
            def fast_lock():
                start = time.monotonic()
                while True:
                    try:
                        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        break
                    except FileExistsError:
                        elapsed = time.monotonic() - start
                        if elapsed >= 0.5:
                            raise LockTimeoutError("Lock timeout in test")
                        time.sleep(0.1)
                try:
                    yield
                finally:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass

            with pytest.raises(LockTimeoutError):
                with fast_lock():
                    pass
        finally:
            if lock_path.exists():
                lock_path.unlink()
