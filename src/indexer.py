"""Repository indexation pipeline for claude-token-saver.

Orchestrates scan -> chunk -> embed -> store pipeline.
The Indexer is a facade that coordinates chunker, embeddings, and storage.
"""

from __future__ import annotations

import hashlib
import json
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
        t_scan = time.monotonic()
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
        logger.debug("Scan+hash: %.2fs (%d files)", time.monotonic() - t_scan, len(current_hashes))

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

        t_chunk = time.monotonic()
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
        logger.debug("Chunking: %.2fs (%d chunks from %d files)", time.monotonic() - t_chunk, len(all_chunks), total_to_process)

        # Also count unchanged files' languages for stats
        unchanged = [p for p in current_hashes if p not in modified and p not in new_files]
        for rel_path in unchanged:
            file_path = Path(rel_path)
            language = detect_language(file_path)
            if language is not None:
                languages[language] = languages.get(language, 0) + 1

        # Step 7: Embed all chunks — provider manages internal batching via config.batch_size
        if all_chunks:
            contents = [c["content"] for c in all_chunks]
            total_chunks = len(contents)
            _progress("embedding", 0, total_chunks, f"0/{total_chunks} chunks")
            t_embed = time.monotonic()
            all_embeddings = self._embed_provider.embed_texts(contents)
            logger.debug("Embedding: %.2fs (%d chunks, batch_size=%d)", time.monotonic() - t_embed, total_chunks, self._config.batch_size)
            _progress("embedding", total_chunks, total_chunks, f"{total_chunks}/{total_chunks} chunks")

            for chunk, embedding in zip(all_chunks, all_embeddings):
                chunk["embedding"] = embedding

            # Step 8: Store (upsert handles delete-then-insert per file)
            _progress("storing", 0, 0, f"{len(all_chunks)} chunks")
            t_store = time.monotonic()
            self._storage.upsert(all_chunks)
            logger.debug("Storage upsert: %.2fs", time.monotonic() - t_store)

        # Step 9: Build reverse caller index (callers.json) over all indexed chunks
        t_callers = time.monotonic()
        self._build_callers_index()
        logger.debug("Callers index: %.2fs", time.monotonic() - t_callers)

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

    def _build_callers_index(self) -> None:
        """Build inverted caller index from all indexed chunks and save as callers.json.

        Iterates all chunks, parses calls_json for each, and builds a dict:
            {called_name: [caller_chunk_name, ...]}

        The result is saved as callers.json next to the LanceDB index files.
        Also invalidates the in-memory cache on the storage instance.
        """
        all_chunks = self._storage.get_all_chunks_with_calls()

        callers: dict[str, list[str]] = {}
        for chunk in all_chunks:
            caller_name = chunk.get("name", "")
            if not caller_name:
                continue
            calls_json_str = chunk.get("calls_json", "[]") or "[]"
            try:
                called_names = json.loads(calls_json_str)
            except (json.JSONDecodeError, TypeError):
                called_names = []
            for called in called_names:
                if called:
                    callers.setdefault(called, []).append(caller_name)

        callers_path = self._storage.index_path / "callers.json"
        try:
            callers_path.write_text(json.dumps(callers, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write callers.json: %s", exc)

        # Invalidate the in-memory cache so the next get_callers() reloads the file
        self._storage.invalidate_callers_cache()
        logger.info("Built callers index: %d entries", len(callers))
