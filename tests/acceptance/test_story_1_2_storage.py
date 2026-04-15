"""
Acceptance tests for Story 1.2: Vector Storage Engine.

Validates LanceDB storage layer with vector search, FTS, file locking,
stale detection, and index path resolution.

FRs: FR30 (detect repo), NFR8 (local storage), NFR11 (file lock), NFR15 (LanceDB only)
"""

import hashlib
import os
import random
import time
from pathlib import Path

import pytest

from src.config import Config
from src.errors import LockTimeoutError
from src.storage import Storage, get_index_path, _compute_repo_hash, EMBEDDING_DIMS


DIM = EMBEDDING_DIMS["lite"]  # 384


def _random_vector(dim: int = DIM, seed: int | None = None) -> list[float]:
    """Generate a random float vector of given dimension."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _make_chunk(
    file_path: str = "src/main.py",
    chunk_type: str = "function",
    name: str = "main",
    line_start: int = 1,
    line_end: int = 10,
    content: str = "def main(): pass",
    file_hash: str = "abc123",
    language: str = "python",
    parent_name: str = "",
    calls_json: str = "[]",
    outline_json: str = "[]",
    embedding: list[float] | None = None,
    seed: int | None = None,
) -> dict:
    """Create a chunk dict with all required fields."""
    return {
        "file_path": file_path,
        "chunk_type": chunk_type,
        "name": name,
        "line_start": line_start,
        "line_end": line_end,
        "content": content,
        "embedding": embedding or _random_vector(seed=seed),
        "file_hash": file_hash,
        "language": language,
        "parent_name": parent_name,
        "calls_json": calls_json,
        "outline_json": outline_json,
    }


@pytest.fixture
def storage(tmp_path):
    """Create a Storage instance using tmp_path as base."""
    config = Config.load(config_path=tmp_path / "nonexistent.yaml")
    repo_path = tmp_path / "fake-repo"
    repo_path.mkdir()
    return Storage(config, repo_path, base_path=tmp_path / "indices")


@pytest.fixture
def storage_with_chunks(storage):
    """Storage pre-loaded with sample chunks."""
    chunks = [
        _make_chunk(
            file_path="src/main.py",
            name="main",
            content="def main(): print('hello world')",
            seed=1,
        ),
        _make_chunk(
            file_path="src/utils.py",
            name="helper",
            content="def helper(): return 42",
            chunk_type="function",
            seed=2,
        ),
        _make_chunk(
            file_path="lib/parser.ts",
            name="parse",
            content="function parse(input: string): AST",
            language="typescript",
            seed=3,
        ),
    ]
    storage.upsert(chunks)
    return storage


class TestStorageUpsert:
    """AC: Given storage is initialized, When upsert() is called with chunks,
    Then chunks are stored with the full PyArrow schema."""

    def test_upsert_stores_chunks_with_full_schema(self, storage):
        """Chunks devem ser armazenados com todos os campos do schema PyArrow
        (file_path, chunk_type, name, line_start, line_end, content, embedding,
        file_hash, language, parent_name, calls_json, outline_json)."""
        chunk = _make_chunk(
            file_path="src/app.py",
            chunk_type="function",
            name="run",
            line_start=5,
            line_end=20,
            content="def run(): ...",
            file_hash="hash123",
            language="python",
            parent_name="App",
            calls_json='["init", "start"]',
            outline_json='["run", "stop"]',
            seed=42,
        )
        storage.upsert([chunk])

        # Verify by searching
        results = storage.search_vector(_random_vector(seed=42), top_k=1)
        assert len(results) >= 1
        r = results[0]
        assert r["file_path"] == "src/app.py"
        assert r["chunk_type"] == "function"
        assert r["name"] == "run"
        assert r["line_start"] == 5
        assert r["line_end"] == 20
        assert r["content"] == "def run(): ..."
        assert r["file_hash"] == "hash123"
        assert r["language"] == "python"
        assert r["parent_name"] == "App"
        assert r["calls_json"] == '["init", "start"]'
        assert r["outline_json"] == '["run", "stop"]'

    def test_upsert_preserves_embedding_vectors(self, storage):
        """Embeddings vetoriais devem ser preservados corretamente no LanceDB."""
        vec = _random_vector(seed=99)
        chunk = _make_chunk(embedding=vec, seed=None)
        storage.upsert([chunk])

        results = storage.search_vector(vec, top_k=1)
        assert len(results) >= 1
        stored_vec = results[0]["embedding"]
        # Compare with tolerance for float precision
        assert len(stored_vec) == DIM
        for a, b in zip(vec, stored_vec):
            assert abs(a - b) < 1e-5


class TestVectorSearch:
    """AC: Given chunks with embeddings exist, When search_vector is called,
    Then top_k results are returned sorted by cosine similarity."""

    def test_vector_search_returns_top_k_results(self, storage_with_chunks):
        """search_vector deve retornar ate top_k resultados."""
        results = storage_with_chunks.search_vector(
            _random_vector(seed=1), top_k=2
        )
        assert len(results) <= 2

    def test_vector_search_sorted_by_cosine_similarity(self, storage):
        """Resultados devem estar ordenados por similaridade cosseno (maior primeiro)."""
        # Insert chunks with known embeddings
        base_vec = [1.0] * DIM
        similar_vec = [0.9] * DIM
        different_vec = [0.0] * (DIM - 1) + [1.0]

        chunks = [
            _make_chunk(name="exact", embedding=base_vec, file_path="a.py"),
            _make_chunk(name="similar", embedding=similar_vec, file_path="b.py"),
            _make_chunk(name="different", embedding=different_vec, file_path="c.py"),
        ]
        storage.upsert(chunks)

        results = storage.search_vector(base_vec, top_k=3)
        assert len(results) >= 2
        # First result should be the exact match (smallest distance)
        assert results[0]["name"] == "exact"

    def test_vector_search_includes_distance_scores(self, storage_with_chunks):
        """Cada resultado deve incluir score de distancia."""
        results = storage_with_chunks.search_vector(
            _random_vector(seed=1), top_k=3
        )
        assert len(results) > 0
        for r in results:
            assert "_distance" in r


class TestFullTextSearch:
    """AC: Given chunks with text content exist, When search_fts is called,
    Then matching chunks are returned via LanceDB FTS."""

    def test_fts_returns_matching_chunks(self, storage_with_chunks):
        """search_fts deve retornar chunks que correspondem ao texto buscado."""
        results = storage_with_chunks.search_fts("hello world", top_k=5)
        assert len(results) >= 1
        # The chunk with "hello world" in content should be returned
        contents = [r["content"] for r in results]
        assert any("hello" in c for c in contents)

    def test_fts_respects_top_k(self, storage_with_chunks):
        """search_fts deve respeitar o limite top_k."""
        results = storage_with_chunks.search_fts("def", top_k=1)
        assert len(results) <= 1


class TestSearchFilters:
    """AC: Given search with filter_ext or filter_path,
    Then only matching chunks are returned."""

    def test_filter_by_extension(self, storage_with_chunks):
        """Filtro filter_ext='.py' deve retornar apenas chunks de arquivos Python."""
        results = storage_with_chunks.search_vector(
            _random_vector(seed=1), top_k=10, filter_ext=".py"
        )
        for r in results:
            assert r["file_path"].endswith(".py")

    def test_filter_by_path(self, storage_with_chunks):
        """Filtro filter_path='src/' deve retornar apenas chunks de arquivos em src/."""
        results = storage_with_chunks.search_vector(
            _random_vector(seed=1), top_k=10, filter_path="src/"
        )
        for r in results:
            assert r["file_path"].startswith("src/")


class TestStaleDetection:
    """AC: Given a file indexed with hash abc123, When file changes to def456,
    Then is_stale returns True."""

    def test_stale_detection_when_hash_changes(self, storage):
        """is_stale deve retornar True quando hash do arquivo mudou."""
        chunk = _make_chunk(file_path="src/app.py", file_hash="abc123", seed=10)
        storage.upsert([chunk])
        assert storage.is_stale("src/app.py", "def456") is True

    def test_not_stale_when_hash_matches(self, storage):
        """is_stale deve retornar False quando hash do arquivo nao mudou."""
        chunk = _make_chunk(file_path="src/app.py", file_hash="abc123", seed=10)
        storage.upsert([chunk])
        assert storage.is_stale("src/app.py", "abc123") is False


class TestFileLock:
    """AC: Given the storage lock is held, When a second operation tries to acquire,
    Then it waits up to 10s then raises LockTimeoutError."""

    def test_lock_timeout_raises_error(self, storage):
        """Segunda operacao deve receber LockTimeoutError apos timeout."""
        import fcntl

        lock_path = storage.index_path / "index.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Hold an exclusive flock on the lock file to block _acquire_lock
        fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with pytest.raises(LockTimeoutError):
                # Use a very short timeout to not slow down tests
                with storage._acquire_lock(timeout=0.5):
                    pass
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            lock_path.unlink(missing_ok=True)

    def test_concurrent_access_waits_for_lock(self, storage):
        """Operacao concorrente deve esperar pelo lock antes de prosseguir."""
        import fcntl
        import threading

        lock_path = storage.index_path / "index.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Hold an exclusive flock to block the acquirer thread
        fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)

        acquired = threading.Event()

        def release_lock_after_delay():
            time.sleep(0.3)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        def try_acquire():
            with storage._acquire_lock(timeout=2.0):
                acquired.set()

        # Start thread to release lock after delay
        releaser = threading.Thread(target=release_lock_after_delay)
        acquirer = threading.Thread(target=try_acquire)

        releaser.start()
        acquirer.start()

        acquirer.join(timeout=5.0)
        releaser.join(timeout=5.0)

        lock_path.unlink(missing_ok=True)
        assert acquired.is_set(), "Lock should have been acquired after release"


class TestIndexPathResolution:
    """AC: Given cwd is /home/user/my-project, When storage initializes,
    Then index is at ~/.claude-token-saver/indices/<repo-hash>/."""

    def test_index_path_uses_repo_hash(self, tmp_path):
        """Caminho do indice deve usar hash derivado do path absoluto do repo."""
        repo_path = tmp_path / "my-project"
        repo_path.mkdir()

        expected_hash = hashlib.sha256(
            str(repo_path.resolve()).encode()
        ).hexdigest()[:16]

        index_path = get_index_path(repo_path)
        assert expected_hash in str(index_path)

    def test_index_stored_in_user_home(self, tmp_path):
        """Indice deve ser armazenado em ~/.claude-token-saver/indices/."""
        repo_path = tmp_path / "my-project"
        repo_path.mkdir()

        index_path = get_index_path(repo_path)
        expected_base = Path.home() / ".claude-token-saver" / "indices"
        assert str(index_path).startswith(str(expected_base))
