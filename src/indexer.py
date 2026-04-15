"""Repository indexation pipeline for claude-token-saver.

Orchestrates scan -> chunk -> embed -> store pipeline.
The Indexer is a facade that coordinates chunker, embeddings, and storage.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

logger = logging.getLogger("cts.indexer")

from .chunker import detect_language, parse, scan_files
from .config import Config
from .embeddings import EmbeddingProvider
from .storage import Storage


def _find_repo_root(start: Path) -> Path:
    """Find repository root by looking for .git directory."""
    current = start.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # No .git found, use start directory
    return start.resolve()


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


class Indexer:
    """Repository indexation facade.

    Coordinates: scan files -> parse chunks -> embed -> store in LanceDB.
    """

    def __init__(
        self,
        config: Config,
        storage: Storage,
        embed_provider: EmbeddingProvider,
        *,
        repo_path: Path | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._embed_provider = embed_provider
        self._repo_path = repo_path or _find_repo_root(Path.cwd())

    @property
    def repo_path(self) -> Path:
        return self._repo_path

    async def reindex(self, *, force: bool = False, progress_callback=None) -> dict:
        """Run the full indexation pipeline without blocking the event loop.

        Args:
            force: If True, delete all existing chunks and rebuild from scratch.
                   If False (default), only re-process files whose hash changed,
                   add new files, and remove deleted files (incremental mode).
            progress_callback: Optional callable(stage, current, total, detail)
                   for reporting progress. Called with:
                   - ("scanning", current, total, file_path)
                   - ("chunking", current, total, file_path)
                   - ("embedding", current, total, batch_info)
                   - ("storing", 0, 0, "")
                   - ("done", 0, 0, summary)

        Returns:
            Stats dict with files_scanned, files_indexed, chunks_created,
            files_updated, files_deleted, files_added, duration_ms, languages.
        """
        import asyncio
        from functools import partial

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._reindex_sync, force, progress_callback))

    def _reindex_sync(self, force: bool = False, progress_callback=None) -> dict:
        """Synchronous implementation of the full indexation pipeline."""
        start_time = time.monotonic()

        def _progress(stage, current=0, total=0, detail=""):
            if progress_callback:
                try:
                    progress_callback(stage, current, total, detail)
                except Exception:
                    pass  # Never let progress reporting break indexation

        # Step 1: Scan current files on disk
        _progress("scanning", 0, 0, str(self._repo_path))
        files = scan_files(self._repo_path, self._config.extra_ignore)

        # Step 2: Compute current hashes for all discovered files
        current_hashes: dict[str, str] = {}
        for i, file_path in enumerate(files):
            language = detect_language(file_path)
            if language is None:
                continue
            rel_path = str(file_path)
            current_hashes[rel_path] = _compute_file_hash(file_path)
            _progress("scanning", i + 1, len(files), rel_path)

        # Step 3: Get stored hashes from index
        if force:
            stored_hashes: dict[str, str] = {}
        else:
            stored_hashes = self._storage.get_file_hashes()

        # Step 4: Compute diff
        modified = [p for p in current_hashes if p in stored_hashes and current_hashes[p] != stored_hashes[p]]
        new_files = [p for p in current_hashes if p not in stored_hashes]
        deleted = [p for p in stored_hashes if p not in current_hashes]

        files_to_process = modified + new_files

        # Step 5: Remove deleted files from index (no lock needed for reads above)
        for fp in deleted:
            self._storage.delete_file(fp)

        # Step 6: Chunk + embed files that need processing
        all_chunks: list[dict] = []
        languages: dict[str, int] = {}

        total_to_process = len(files_to_process)
        for idx, rel_path in enumerate(files_to_process):
            file_path = Path(rel_path)
            language = detect_language(file_path)
            if language is None:
                continue

            languages[language] = languages.get(language, 0) + 1
            file_hash = current_hashes[rel_path]

            _progress("chunking", idx + 1, total_to_process, rel_path)

            chunks = parse(
                file_path,
                language,
                max_chunk_lines=self._config.max_chunk_lines,
            )

            for chunk in chunks:
                chunk["file_hash"] = file_hash
                chunk["file_path"] = rel_path

            all_chunks.extend(chunks)

        # Also count unchanged files' languages for stats
        unchanged = [p for p in current_hashes if p not in modified and p not in new_files]
        for rel_path in unchanged:
            file_path = Path(rel_path)
            language = detect_language(file_path)
            if language is not None:
                languages[language] = languages.get(language, 0) + 1

        # Step 7: Embed in batches
        if all_chunks:
            contents = [f"# {c['name']} ({c['chunk_type']}) in {c.get('file_path', '')}\n{c['content']}" for c in all_chunks]
            batch_size = self._config.batch_size

            all_embeddings: list[list[float]] = []
            total_chunks = len(contents)
            for i in range(0, total_chunks, batch_size):
                batch = contents[i: i + batch_size]
                _progress("embedding", min(i + batch_size, total_chunks), total_chunks,
                          f"batch {i // batch_size + 1}")
                embeddings = self._embed_provider.embed_texts(batch)
                all_embeddings.extend(embeddings)

            for chunk, embedding in zip(all_chunks, all_embeddings):
                chunk["embedding"] = embedding

            # Step 8: Store (upsert handles delete-then-insert per file)
            _progress("storing", 0, 0, f"{len(all_chunks)} chunks")
            self._storage.upsert(all_chunks)

        duration_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "Reindex: %d scanned, %d updated, %d added, %d deleted, %d chunks in %dms",
            len(files),
            len(modified),
            len(new_files),
            len(deleted),
            len(all_chunks),
            duration_ms,
        )

        _progress("done", 0, 0,
                  f"{len(all_chunks)} chunks from {len(files_to_process)} files in {duration_ms}ms")

        return {
            "files_scanned": len(files),
            "files_indexed": len(languages),
            "chunks_created": len(all_chunks),
            "files_updated": len(modified),
            "files_deleted": len(deleted),
            "files_added": len(new_files),
            "duration_ms": duration_ms,
            "languages": languages,
        }
