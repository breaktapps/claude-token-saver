"""LanceDB storage layer for claude-token-saver.

Handles vector storage, search (vector + FTS), file locking, stale detection,
and index path resolution. LanceDB is the only storage backend.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from .config import Config
from .embeddings import LITE_DIMENSION, QUALITY_DIMENSION
from .errors import LockTimeoutError

logger = logging.getLogger("cts.storage")

# Default lock timeout in seconds
LOCK_TIMEOUT = 10
LOCK_POLL_INTERVAL = 0.5


def _escape_sql(value: str) -> str:
    """Escape a string value for safe use in SQL WHERE clauses.

    Replaces single quotes with two single quotes (SQL standard escaping).
    """
    return value.replace("'", "''")


def _compute_repo_hash(repo_path: Path) -> str:
    """Generate a deterministic hash from the absolute repo path."""
    return hashlib.sha256(str(repo_path.resolve()).encode()).hexdigest()[:16]


def get_index_path(repo_path: Path) -> Path:
    """Return the index directory path for a given repo."""
    repo_hash = _compute_repo_hash(repo_path)
    return Path.home() / ".claude-token-saver" / "indices" / repo_hash


def _build_schema(dim: int) -> pa.Schema:
    """Build the PyArrow schema for the chunks table."""
    return pa.schema([
        pa.field("file_path", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("name", pa.string()),
        pa.field("line_start", pa.int32()),
        pa.field("line_end", pa.int32()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), dim)),
        pa.field("file_hash", pa.string()),
        pa.field("language", pa.string()),
        pa.field("parent_name", pa.string()),
        pa.field("calls_json", pa.string()),
        pa.field("outline_json", pa.string()),
        pa.field("imports", pa.list_(pa.string())),
    ])


class Storage:
    """LanceDB storage interface for code chunks."""

    def __init__(self, config: Config, repo_path: Path, *, base_path: Path | None = None):
        """Initialize storage for a repository.

        Args:
            config: Plugin configuration.
            repo_path: Path to the repository root.
            base_path: Override for index base path (for testing).
        """
        self._config = config
        self._repo_path = repo_path.resolve()

        if base_path is not None:
            repo_hash = _compute_repo_hash(repo_path)
            self._index_path = base_path / repo_hash
        else:
            self._index_path = get_index_path(repo_path)

        self._index_path.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._index_path / "index.lock"

        _EMBEDDING_DIMS = {"lite": LITE_DIMENSION, "quality": QUALITY_DIMENSION}
        dim = _EMBEDDING_DIMS.get(config.embedding_mode, LITE_DIMENSION)
        self._schema = _build_schema(dim)
        self._dim = dim

        self._db = lancedb.connect(str(self._index_path / "index.lance"))
        self._table = None
        self._fts_created = False

    @property
    def index_path(self) -> Path:
        """Return the index directory path."""
        return self._index_path

    def _get_or_create_table(self) -> Any:
        """Get or create the chunks table."""
        if self._table is not None:
            return self._table

        try:
            self._table = self._db.open_table("chunks")
        except Exception:
            # Table doesn't exist, create it
            empty = pa.table(
                {field.name: pa.array([], type=field.type) for field in self._schema},
                schema=self._schema,
            )
            self._table = self._db.create_table("chunks", data=empty, schema=self._schema)

        return self._table

    def _ensure_fts_index(self) -> None:
        """Create FTS index on content, name, and file_path fields."""
        if self._fts_created:
            return
        table = self._get_or_create_table()
        try:
            table.create_fts_index("content", replace=True)
            table.create_fts_index("name", replace=True)
            table.create_fts_index("file_path", replace=True)
            self._fts_created = True
        except Exception:
            # FTS index creation can fail on empty tables
            pass

    @contextmanager
    def _acquire_lock(self, timeout: float = LOCK_TIMEOUT):
        """Acquire file lock with timeout using fcntl.flock().

        Unlike O_CREAT|O_EXCL, flock locks are released automatically by the OS
        when the process dies, preventing orphaned lock files.

        Raises:
            LockTimeoutError: If lock cannot be acquired within timeout.
        """
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
        start = time.monotonic()
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    elapsed = time.monotonic() - start
                    if elapsed >= timeout:
                        raise LockTimeoutError(
                            f"Could not acquire lock after {timeout}s. "
                            "Another operation may be in progress."
                        )
                    time.sleep(LOCK_POLL_INTERVAL)
            logger.debug("Acquired lock")
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                logger.debug("Released lock")
        finally:
            os.close(fd)

    def upsert(self, chunks: list[dict]) -> None:
        """Insert or update chunks in the table.

        Strategy: delete existing chunks for each file_path, then insert new ones.
        """
        if not chunks:
            return

        with self._acquire_lock():
            table = self._get_or_create_table()

            # Group chunks by file_path for efficient delete
            file_paths = {c["file_path"] for c in chunks}
            for fp in file_paths:
                try:
                    table.delete(f"file_path = '{_escape_sql(fp)}'")
                except Exception:
                    pass  # Table may be empty or file_path not found

            # Prepare data for insertion
            arrays = {}
            for field in self._schema:
                values = [c.get(field.name) for c in chunks]
                arrays[field.name] = pa.array(values, type=field.type)

            batch = pa.table(arrays, schema=self._schema)
            table.add(batch)
            logger.info("Upserted %d chunks for %d file(s)", len(chunks), len(file_paths))

            # Recreate FTS index after data changes
            self._fts_created = False
            self._ensure_fts_index()

    def search_vector(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_ext: str | None = None,
        filter_path: str | None = None,
    ) -> list[dict]:
        """Search by vector similarity (cosine).

        Returns list of dicts with chunk fields and _distance score.
        Raises ValueError if vector dimension does not match index dimension.
        """
        if len(vector) != self._dim:
            raise ValueError(
                f"Vector dimension mismatch: provider returned {len(vector)}-dim vector "
                f"but index was built with {self._dim}-dim embeddings. "
                "Run reindex with the new embedding provider to rebuild the index."
            )

        table = self._get_or_create_table()

        query = table.search(vector, query_type="vector").metric("cosine").limit(top_k)

        where_clauses = self._build_where(filter_ext, filter_path)
        if where_clauses:
            query = query.where(where_clauses)

        try:
            results = query.to_list()
        except Exception:
            return []

        return results

    def search_fts(
        self,
        query_text: str,
        top_k: int = 5,
        filter_ext: str | None = None,
        filter_path: str | None = None,
    ) -> list[dict]:
        """Search by full-text search on content field.

        Returns list of dicts with chunk fields.
        """
        self._ensure_fts_index()
        table = self._get_or_create_table()

        query = table.search(query_text, query_type="fts").limit(top_k)

        where_clauses = self._build_where(filter_ext, filter_path)
        if where_clauses:
            query = query.where(where_clauses)

        try:
            results = query.to_list()
        except Exception:
            return []

        return results

    def is_stale(self, file_path: str, current_hash: str) -> bool:
        """Check if a file's indexed version is stale.

        Returns True if the stored hash differs from current_hash.
        """
        table = self._get_or_create_table()
        try:
            results = (
                table.search()
                .where(f"file_path = '{_escape_sql(file_path)}'")
                .select(["file_hash"])
                .limit(1)
                .to_list()
            )
        except Exception:
            return True

        if not results:
            return True

        stored_hash = results[0].get("file_hash", "")
        return stored_hash != current_hash

    def get_file_hashes(self) -> dict[str, str]:
        """Return mapping of file_path -> file_hash for all indexed files."""
        table = self._get_or_create_table()
        try:
            # Full scan intentional — needed for complete stale detection
            results = table.search().select(["file_path", "file_hash"]).to_list()
        except Exception:
            return {}

        hashes = {}
        for r in results:
            fp = r.get("file_path", "")
            fh = r.get("file_hash", "")
            if fp and fp not in hashes:
                hashes[fp] = fh
        return hashes

    def get_all_chunks_metadata(self) -> list[dict]:
        """Return minimal metadata for all chunks: file_path, file_hash, language, chunk_type."""
        table = self._get_or_create_table()
        try:
            # Full scan intentional — needed for complete stale detection
            results = table.search().select(["file_path", "file_hash", "language", "chunk_type"]).to_list()
        except Exception:
            return []
        return results

    def get_chunks_for_file(self, file_path: str) -> list[dict]:
        """Return all chunks for a specific file_path with chunk metadata."""
        table = self._get_or_create_table()
        try:
            results = (
                table.search()
                .where(f"file_path = '{_escape_sql(file_path)}'")
                .select(["name", "chunk_type", "line_start", "line_end", "file_hash"])
                .to_list()
            )
        except Exception:
            return []
        return results

    def get_chunks_by_file(self, file_path: str) -> list[dict]:
        """Return all chunks for a specific file_path with full fields needed by get_file tool."""
        table = self._get_or_create_table()
        try:
            results = (
                table.search()
                .where(f"file_path = '{_escape_sql(file_path)}'")
                .select([
                    "file_path", "chunk_type", "name", "line_start", "line_end",
                    "content", "language", "file_hash", "parent_name", "calls_json",
                ])
                .to_list()
            )
        except Exception:
            return []
        return results

    def search_by_name(self, name: str) -> list[dict]:
        """Look up chunks by exact name match.

        Returns list of dicts with full chunk fields. Returns empty list if not found.
        """
        table = self._get_or_create_table()
        try:
            results = (
                table.search()
                .where(f"name = '{_escape_sql(name)}'")
                .select([
                    "file_path", "chunk_type", "name", "line_start", "line_end",
                    "content", "language", "file_hash", "parent_name", "calls_json",
                ])
                .limit(1)
                .to_list()
            )
        except Exception:
            return []
        return results

    def delete_file(self, file_path: str) -> None:
        """Delete all chunks for a given file."""
        with self._acquire_lock():
            table = self._get_or_create_table()
            try:
                table.delete(f"file_path = '{_escape_sql(file_path)}'")
            except Exception:
                pass

    @staticmethod
    def _build_where(filter_ext: str | None, filter_path: str | None) -> str | None:
        """Build SQL WHERE clause from filters."""
        clauses = []
        if filter_ext:
            ext = _escape_sql(filter_ext.lstrip("."))
            clauses.append(f"file_path LIKE '%.{ext}'")
        if filter_path:
            clauses.append(f"file_path LIKE '{_escape_sql(filter_path)}%'")
        return " AND ".join(clauses) if clauses else None
