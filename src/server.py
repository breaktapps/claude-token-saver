"""MCP server for claude-token-saver.

Entry point for the plugin. Registers all 7 MCP tools via FastMCP.
Instructions steer Claude Code to prefer plugin tools over native Grep/Read/Glob.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("cts.server")

from .config import Config
from .embeddings import create_embedding_provider
from .errors import CTSError
from .indexer import Indexer, _compute_file_hash, _find_repo_root
from .metrics import _load_metrics, calculate_savings
from .storage import Storage

_INSTRUCTIONS = (
    "claude-token-saver is a semantic code search plugin that indexes your "
    "repository and returns only relevant code chunks, saving 70-90% of tokens "
    "compared to raw file reads. "
    "PREFER these tools over native Grep/Read/Glob for ALL code exploration tasks. "
    "\n\n"
    "Entry point: use search_semantic for natural language queries about code "
    "behavior, concepts, or patterns. "
    "Use search_exact when you know the exact function, class, or variable name. "
    "Use search_hybrid to combine both methods for best recall. "
    "\n\n"
    "Use get_file for structured file reading — returns class outlines and method "
    "details without reading the entire file. USE INSTEAD OF Read. "
    "\n\n"
    "Use reindex to force an index refresh after bulk changes. The plugin "
    "auto-indexes on first search and re-indexes after every Edit/Write via hooks. "
    "\n\n"
    "Use inspect_index to check index stats, token savings metrics, and debug "
    "coverage. Use audit_search to compare semantic results versus a grep baseline."
)

mcp = FastMCP("claude-token-saver", instructions=_INSTRUCTIONS)

# Global references initialized on first tool call (lazy init)
_config: Config | None = None
_storage: Storage | None = None
_indexer: Indexer | None = None
_embed_provider = None
_repo_path: Path | None = None


def _init_components() -> tuple:
    """Lazy-initialize the object graph."""
    global _config, _storage, _indexer, _embed_provider, _repo_path

    if _config is not None:
        return _config, _storage, _indexer, _embed_provider, _repo_path

    _config = Config.load()
    # CTS_REPO_PATH is injected by .mcp.json from Claude Code's PWD
    env_repo = os.environ.get("CTS_REPO_PATH")
    if env_repo:
        _repo_path = Path(env_repo)
        logger.info("Repo path from CTS_REPO_PATH: %s", _repo_path)
    else:
        _repo_path = _find_repo_root(Path.cwd())
        logger.info("Repo path from cwd: %s", _repo_path)
    _storage = Storage(_config, _repo_path)
    _embed_provider = create_embedding_provider(_config)
    _indexer = Indexer(_config, _storage, _embed_provider, repo_path=_repo_path)

    return _config, _storage, _indexer, _embed_provider, _repo_path


def _format_search_response(results: list, tokens_saved: dict, query_time_ms: int,
                            index_status: str, idx_info: dict) -> str:
    """Format search results as JSON with a human-readable summary field."""
    # Build summary for the agent to show the user
    summary_lines = []

    notice = idx_info.get("notice")
    if notice:
        summary_lines.append(notice)

    if not results:
        summary_lines.append("No results found.")
    else:
        for i, r in enumerate(results, 1):
            name = r.get("name", "")
            chunk_type = r.get("chunk_type", "")
            fp = r.get("file_path", "")
            score = r.get("score", 0)
            line_start = r.get("line_start", 0)
            parent = r.get("parent_name", "")
            parent_info = f" in {parent}" if parent else ""
            summary_lines.append(
                f"{i}. {name} ({chunk_type}{parent_info}) — {fp}:{line_start} [score: {score}]"
            )

    saved = tokens_saved.get("saved", 0)
    pct = tokens_saved.get("reduction_pct", 0)
    summary_lines.append(f"Tokens saved: {saved:,} ({pct}% reduction) | Query: {query_time_ms}ms")

    # Truncate content in results to keep response compact
    compact_results = []
    for r in results:
        cr = dict(r)
        content = cr.get("content", "")
        content_lines = content.split("\n")
        if len(content_lines) > 15:
            cr["content"] = "\n".join(content_lines[:15]) + f"\n... ({len(content_lines) - 15} more lines)"
        compact_results.append(cr)

    response = {
        "summary": "\n".join(summary_lines),
        "results": compact_results,
        "tokens_saved": tokens_saved,
        "metadata": {
            "query_time_ms": query_time_ms,
            "index_status": index_status,
            "notice": notice,
        },
    }

    return json.dumps(response, ensure_ascii=False)


async def _ensure_indexed(indexer: Indexer, storage: Storage) -> dict:
    """Ensure the repository is indexed. Returns dict with status and notice."""
    hashes = storage.get_file_hashes()
    if hashes:
        return {"status": "ready", "notice": None}

    # First-time indexation
    logger.info("First-time indexation starting for %s", indexer.repo_path)

    stats = await indexer.reindex(force=False)

    notice = (
        f"First-time index complete: {stats.get('files_scanned', 0)} files scanned, "
        f"{stats.get('chunks_created', 0)} chunks created in "
        f"{stats.get('duration_ms', 0) / 1000:.1f}s. "
        f"Subsequent searches will be fast (<500ms)."
    )
    logger.info(notice)

    return {"status": "first_index", "notice": notice}


@mcp.tool(
    description=(
        "Search code by natural language query. Returns semantically relevant "
        "code chunks ranked by similarity. USE INSTEAD OF Grep for code exploration. "
        "Supports filters: filter_ext (e.g., '.py'), filter_path (e.g., 'src/'). "
        "Alias: file_filter — if starts with '.' or '*.' it maps to filter_ext, "
        "otherwise to filter_path."
    )
)
async def search_semantic(
    query: str,
    top_k: int = 5,
    filter_ext: str | None = None,
    filter_path: str | None = None,
    file_filter: str | None = None,
) -> str:
    """Search code semantically and return ranked results with token savings."""
    try:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return json.dumps({"error": "top_k must be an integer", "suggestion": "Pass top_k as an integer (e.g., top_k=5)"})

        start = time.perf_counter()
        config, storage, indexer, embed_provider, repo_path = _init_components()

        # Resolve file_filter alias
        if file_filter is not None:
            if file_filter.startswith(".") or file_filter.startswith("*."):
                filter_ext = filter_ext or file_filter.lstrip("*")
            else:
                filter_path = filter_path or file_filter

        # Auto-index if needed
        _idx = await _ensure_indexed(indexer, storage)
        index_status = _idx["status"]

        # Embed the query
        query_vector = embed_provider.embed_query(query)

        # Search
        raw_results = storage.search_vector(
            query_vector, top_k=top_k, filter_ext=filter_ext, filter_path=filter_path
        )

        # Build results with stale check; apply score_threshold filter
        results = []
        for r in raw_results:
            score = round(1.0 - r.get("_distance", 0.0), 4)
            if score < config.score_threshold:
                continue

            fp = r.get("file_path", "")
            file_hash = r.get("file_hash", "")
            stale = storage.is_stale(fp, file_hash) if fp and file_hash else False

            results.append({
                "file_path": fp,
                "name": r.get("name", ""),
                "chunk_type": r.get("chunk_type", ""),
                "line_start": r.get("line_start", 0),
                "line_end": r.get("line_end", 0),
                "content": r.get("content", ""),
                "score": score,
                "language": r.get("language", ""),
                "stale": stale,
                "parent_name": r.get("parent_name", ""),
            })

        # Calculate token savings (persists cumulative metrics)
        tokens_saved = calculate_savings(results, repo_path, index_path=storage.index_path)

        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info("search_semantic: query=%r, results=%d, time=%dms", query, len(results), query_time_ms)

        return _format_search_response(results, tokens_saved, query_time_ms, index_status, _idx)

    except CTSError as e:
        return json.dumps({
            "error": str(e),
            "suggestion": _error_suggestion(e),
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please report this issue. Try running reindex.",
        })


@mcp.tool(
    description=(
        "Re-index the repository to keep search up to date. "
        "IMPORTANT: Tell the user indexing is in progress before calling — "
        "first-time indexing takes 30-60s, incremental takes 1-3s. "
        "force=false (default): incremental, only changed files. "
        "force=true: full rebuild. "
        "Returns: files_scanned, files_updated, files_deleted, files_added, chunks_created, duration_ms."
    )
)
async def reindex(force: bool = False) -> str:
    """Re-index the repository (incremental by default, full rebuild with force=true)."""
    try:
        config, storage, indexer, embed_provider, repo_path = _init_components()

        stats = await indexer.reindex(force=force)

        index_status = "rebuilt" if force else "updated"
        logger.info(
            "reindex: force=%s, scanned=%d, updated=%d, added=%d, deleted=%d, time=%dms",
            force,
            stats["files_scanned"],
            stats.get("files_updated", 0),
            stats.get("files_added", 0),
            stats.get("files_deleted", 0),
            stats["duration_ms"],
        )

        scanned = stats["files_scanned"]
        updated = stats.get("files_updated", 0)
        added = stats.get("files_added", 0)
        deleted = stats.get("files_deleted", 0)
        chunks = stats["chunks_created"]
        duration = stats["duration_ms"]
        secs = duration / 1000

        return (
            f"Reindex complete ({index_status}): "
            f"{scanned} files scanned, {added} added, {updated} updated, {deleted} deleted. "
            f"{chunks} chunks created in {secs:.1f}s."
        )

    except CTSError as e:
        return json.dumps({
            "error": str(e),
            "suggestion": _error_suggestion(e),
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please report this issue.",
        })


@mcp.tool(
    description=(
        "Search code by exact name (function, class, variable). Returns enriched "
        "chunks via full-text search. USE INSTEAD OF Grep for exact name lookups. "
        "Supports filters: filter_ext (e.g., '.dart'), filter_path (e.g., 'src/'). "
        "Alias: file_filter — if starts with '.' or '*.' it maps to filter_ext, "
        "otherwise to filter_path."
    )
)
async def search_exact(
    query: str,
    top_k: int = 5,
    filter_ext: str | None = None,
    filter_path: str | None = None,
    file_filter: str | None = None,
) -> str:
    """Search code by exact name via FTS (no embedding needed)."""
    try:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return json.dumps({"error": "top_k must be an integer", "suggestion": "Pass top_k as an integer (e.g., top_k=5)"})

        start = time.perf_counter()
        config, storage, indexer, embed_provider, repo_path = _init_components()

        # Resolve file_filter alias
        if file_filter is not None:
            if file_filter.startswith(".") or file_filter.startswith("*."):
                filter_ext = filter_ext or file_filter.lstrip("*")
            else:
                filter_path = filter_path or file_filter

        # Auto-index if needed
        _idx = await _ensure_indexed(indexer, storage)
        index_status = _idx["status"]

        # FTS search -- no embedding
        raw_results = storage.search_fts(
            query, top_k=top_k, filter_ext=filter_ext, filter_path=filter_path
        )

        # Build results with stale check; apply score_threshold filter
        results = []
        for r in raw_results:
            score = r.get("_score", r.get("score", 0.0))
            if score < config.score_threshold:
                continue

            fp = r.get("file_path", "")
            file_hash = r.get("file_hash", "")
            stale = storage.is_stale(fp, file_hash) if fp and file_hash else False

            results.append({
                "file_path": fp,
                "name": r.get("name", ""),
                "chunk_type": r.get("chunk_type", ""),
                "line_start": r.get("line_start", 0),
                "line_end": r.get("line_end", 0),
                "content": r.get("content", ""),
                "score": score,
                "language": r.get("language", ""),
                "stale": stale,
                "parent_name": r.get("parent_name", ""),
            })

        # Sort: class outlines first, then methods/functions
        _TYPE_ORDER = {"class": 0, "method": 1, "function": 2}
        results.sort(key=lambda x: _TYPE_ORDER.get(x["chunk_type"], 3))

        # Calculate token savings (persists cumulative metrics)
        tokens_saved = calculate_savings(results, repo_path, index_path=storage.index_path)

        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info("search_exact: query=%r, results=%d, time=%dms", query, len(results), query_time_ms)

        return _format_search_response(results, tokens_saved, query_time_ms, index_status, _idx)

    except CTSError as e:
        return json.dumps({
            "error": str(e),
            "suggestion": _error_suggestion(e),
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please report this issue. Try running reindex.",
        })


@mcp.tool(
    description=(
        "Inspect the semantic index. "
        "Without arguments: returns global stats (total_files, total_chunks, languages, "
        "index_size_bytes, tokens_saved_total, total_queries, stale_files_count, stale_files). "
        "With file_path: lists all chunks for that file (name, chunk_type, line_start, line_end, stale)."
    )
)
async def inspect_index(file_path: str | None = None) -> str:
    """Inspect index stats or per-file chunks."""
    try:
        config, storage, indexer, embed_provider, repo_path = _init_components()

        # Check if index has any data
        hashes = storage.get_file_hashes()
        if not hashes:
            return json.dumps({
                "error": "No index found",
                "suggestion": "Run a search or reindex to create the index",
            })

        if file_path is not None:
            # Per-file mode
            chunks = storage.get_chunks_for_file(file_path)
            result_chunks = []
            current_hash = _current_file_hash(file_path, repo_path)
            stale = storage.is_stale(file_path, current_hash)
            for c in chunks:
                result_chunks.append({
                    "name": c.get("name", ""),
                    "chunk_type": c.get("chunk_type", ""),
                    "line_start": c.get("line_start", 0),
                    "line_end": c.get("line_end", 0),
                    "stale": stale,
                })
            lines = [f"File: {file_path} ({'STALE' if stale else 'up-to-date'})"]
            for c in result_chunks:
                lines.append(f"  {c['chunk_type']:10s} {c['name']:30s} L{c['line_start']}-{c['line_end']}")
            return "\n".join(lines)

        # Global stats mode
        all_meta = storage.get_all_chunks_metadata()

        # Count unique files and chunks
        total_chunks = len(all_meta)
        unique_files: set[str] = set()
        lang_counts: dict[str, int] = {}
        for m in all_meta:
            fp = m.get("file_path", "")
            if fp:
                unique_files.add(fp)
            lang = m.get("language", "")
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        total_files = len(unique_files)

        # Index size: directory size of index.lance
        index_size_bytes = 0
        lance_dir = storage.index_path / "index.lance"
        if lance_dir.exists():
            for f in lance_dir.rglob("*"):
                if f.is_file():
                    try:
                        index_size_bytes += f.stat().st_size
                    except OSError:
                        pass

        # Cumulative metrics from metrics.json
        metrics = _load_metrics(storage.index_path)
        tokens_saved_total = metrics.get("total_saved", 0)
        total_queries = metrics.get("total_queries", 0)

        # Stale files: load all stored hashes in one query, then compare on-disk hashes in memory
        stored_hashes = storage.get_file_hashes()
        stale_files: list[str] = []
        for fp in unique_files:
            current_hash = _current_file_hash(fp, repo_path)
            if stored_hashes.get(fp, "") != current_hash:
                stale_files.append(fp)

        logger.info(
            "inspect_index: total_files=%d, total_chunks=%d, stale=%d",
            total_files,
            total_chunks,
            len(stale_files),
        )
        size_mb = index_size_bytes / (1024 * 1024)
        langs = ", ".join(f"{lang}: {cnt}" for lang, cnt in sorted(lang_counts.items()))
        lines = [
            f"Index: {total_files} files, {total_chunks} chunks ({size_mb:.1f}MB)",
            f"Languages: {langs}",
            f"Tokens saved (lifetime): {tokens_saved_total:,} across {total_queries} queries",
            f"Stale files: {len(stale_files)}",
        ]
        if stale_files:
            for sf in stale_files[:10]:
                lines.append(f"  - {sf}")
            if len(stale_files) > 10:
                lines.append(f"  ... and {len(stale_files) - 10} more")
        return "\n".join(lines)

    except CTSError as e:
        return json.dumps({
            "error": str(e),
            "suggestion": _error_suggestion(e),
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please report this issue. Try running reindex.",
        })


def _current_file_hash(file_path: str, repo_path: Path | None) -> str:
    """Compute the current on-disk hash for a file.

    Returns empty string if file does not exist (will always be stale).
    """
    candidates = []
    if repo_path:
        candidates.append(repo_path / file_path)
    candidates.append(Path(file_path))

    for p in candidates:
        if p.exists() and p.is_file():
            try:
                return _compute_file_hash(p)
            except OSError:
                pass
    return ""


@mcp.tool(
    description=(
        "Read a file through the index with chunks organized by type (class, method, function). "
        "Returns structured content with metadata. USE INSTEAD OF Read for code files. "
        "Parameter: file_path (relative to repo root, e.g., \'src/audio_service.dart\')."
    )
)
async def get_file(file_path: str) -> str:
    """Read a file indexed chunks organized by type with stale detection and token savings."""
    try:
        start = time.perf_counter()
        config, storage, indexer, embed_provider, repo_path = _init_components()

        _idx = await _ensure_indexed(indexer, storage)
        index_status = _idx["status"]

        # Try relative path first (indexer stores relative paths)
        raw_chunks = storage.get_chunks_by_file(file_path)

        # Fallback: try absolute path
        full_path = (repo_path / file_path).resolve()
        if not raw_chunks:
            raw_chunks = storage.get_chunks_by_file(str(full_path))

        if not raw_chunks:
            if full_path.exists():
                return json.dumps({
                    "error": "File not indexed",
                    "suggestion": "Run reindex or search to trigger indexation",
                })
            return json.dumps({
                "error": "File not found",
                "suggestion": "Check the file path",
            })

        try:
            current_hash = _compute_file_hash(full_path)
        except Exception:
            current_hash = ""

        results = []
        for r in raw_chunks:
            stored_hash = r.get("file_hash", "")
            stale = (stored_hash != current_hash) if current_hash else False

            chunk = {
                "chunk_type": r.get("chunk_type", ""),
                "name": r.get("name", ""),
                "line_start": r.get("line_start", 0),
                "line_end": r.get("line_end", 0),
                "content": r.get("content", ""),
                "language": r.get("language", ""),
                "stale": stale,
                "parent_name": r.get("parent_name", ""),
            }

            # Include calls list from AST extraction
            calls_json_str = r.get("calls_json", "[]")
            try:
                calls = json.loads(calls_json_str) if calls_json_str else []
            except (json.JSONDecodeError, TypeError):
                calls = []
            chunk["calls"] = calls

            results.append(chunk)

        _TYPE_ORDER = {"class": 0, "method": 1, "function": 2}
        results.sort(key=lambda x: (_TYPE_ORDER.get(x["chunk_type"], 3), x["line_start"]))

        tokens_saved = calculate_savings(
            [{"file_path": str(full_path), "content": r["content"]} for r in results],
            None,  # file_path is already absolute
            index_path=storage.index_path,
        )

        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info("get_file: path=%r, chunks=%d, time=%dms", file_path, len(results), query_time_ms)

        # Format as readable text
        lines = [f"File: {file_path} ({len(results)} chunks)"]
        for r in results:
            name = r.get("name", "")
            chunk_type = r.get("chunk_type", "")
            line_start = r.get("line_start", 0)
            line_end = r.get("line_end", 0)
            lang = r.get("language", "")
            content = r.get("content", "")
            stale = r.get("stale", False)
            stale_tag = " [STALE]" if stale else ""

            lines.append(f"\n### {name} ({chunk_type}) L{line_start}-{line_end}{stale_tag}")
            content_lines = content.split("\n")
            if len(content_lines) > 20:
                lines.append(f"```{lang}")
                lines.append("\n".join(content_lines[:20]))
                lines.append(f"... ({len(content_lines) - 20} more lines)")
                lines.append("```")
            else:
                lines.append(f"```{lang}")
                lines.append(content)
                lines.append("```")

        saved = tokens_saved.get("saved", 0)
        pct = tokens_saved.get("reduction_pct", 0)
        lines.append(f"\n---\nTokens saved: {saved:,} ({pct}% reduction) | Query: {query_time_ms}ms")

        return "\n".join(lines)

    except CTSError as e:
        return json.dumps({
            "error": str(e),
            "suggestion": _error_suggestion(e),
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please report this issue. Try running reindex.",
        })


@mcp.tool(
    description=(
        "Hybrid search combining semantic (vector) and exact (FTS) approaches "
        "for maximum recall. Use when you want both conceptual matches AND exact "
        "name/token matches in one query. Returns merged, de-duplicated results "
        "ranked by combined relevance score. USE INSTEAD OF running search_semantic "
        "and search_exact separately. "
        "Supports filters: filter_ext (e.g., '.py'), filter_path (e.g., 'src/'). "
        "Alias: file_filter — if starts with '.' or '*.' it maps to filter_ext, "
        "otherwise to filter_path."
    )
)
async def search_hybrid(
    query: str,
    top_k: int = 5,
    filter_ext: str | None = None,
    filter_path: str | None = None,
    file_filter: str | None = None,
) -> str:
    """Combine vector search and FTS, merge and de-duplicate results."""
    try:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return json.dumps({"error": "top_k must be an integer", "suggestion": "Pass top_k as an integer (e.g., top_k=5)"})

        start = time.perf_counter()
        config, storage, indexer, embed_provider, repo_path = _init_components()

        # Resolve file_filter alias
        if file_filter is not None:
            if file_filter.startswith(".") or file_filter.startswith("*."):
                filter_ext = filter_ext or file_filter.lstrip("*")
            else:
                filter_path = filter_path or file_filter

        # Auto-index if needed
        _idx = await _ensure_indexed(indexer, storage)
        index_status = _idx["status"]

        # Run both branches in parallel; fetch top_k*2 to allow for overlap
        fetch_k = top_k * 2

        def _run_vector():
            query_vector = embed_provider.embed_query(query)
            return storage.search_vector(
                query_vector, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path
            )

        def _run_fts():
            return storage.search_fts(
                query, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path
            )

        loop = asyncio.get_running_loop()
        vector_raw, fts_raw = await asyncio.gather(
            loop.run_in_executor(None, _run_vector),
            loop.run_in_executor(None, _run_fts),
        )

        # Index vector results by dedup key → (record, vector_score)
        def _dedup_key(r: dict) -> tuple:
            return (r.get("file_path", ""), r.get("name", ""), r.get("line_start", 0))

        vec_map: dict[tuple, tuple[dict, float]] = {}
        for r in vector_raw:
            key = _dedup_key(r)
            score = round(1.0 - r.get("_distance", 0.0), 4)
            vec_map[key] = (r, score)

        fts_map: dict[tuple, tuple[dict, float]] = {}
        for r in fts_raw:
            key = _dedup_key(r)
            score = r.get("_score", r.get("score", 0.0))
            fts_map[key] = (r, score)

        # Merge all unique keys
        all_keys = set(vec_map) | set(fts_map)
        merged: list[dict] = []
        for key in all_keys:
            in_vec = key in vec_map
            in_fts = key in fts_map

            if in_vec and in_fts:
                rec, v_score = vec_map[key]
                _, f_score = fts_map[key]
                # 1.2x boost for items found by both methods; capped at 1.0
                combined_score = round(min(max(v_score, f_score) * 1.2, 1.0), 4)
            elif in_vec:
                rec, v_score = vec_map[key]
                combined_score = v_score
            else:
                rec, f_score = fts_map[key]
                combined_score = f_score

            fp = rec.get("file_path", "")
            file_hash = rec.get("file_hash", "")
            stale = storage.is_stale(fp, file_hash) if fp and file_hash else False

            merged.append({
                "file_path": fp,
                "name": rec.get("name", ""),
                "chunk_type": rec.get("chunk_type", ""),
                "line_start": rec.get("line_start", 0),
                "line_end": rec.get("line_end", 0),
                "content": rec.get("content", ""),
                "score": combined_score,
                "language": rec.get("language", ""),
                "stale": stale,
                "parent_name": rec.get("parent_name", ""),
            })

        # Sort by combined score descending, apply score_threshold, and limit to top_k
        merged.sort(key=lambda x: x["score"], reverse=True)
        results = [r for r in merged if r["score"] >= config.score_threshold][:top_k]

        tokens_saved = calculate_savings(results, repo_path, index_path=storage.index_path)
        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info("search_hybrid: query=%r, results=%d, time=%dms", query, len(results), query_time_ms)

        return _format_search_response(results, tokens_saved, query_time_ms, index_status, _idx)

    except CTSError as e:
        return json.dumps({
            "error": str(e),
            "suggestion": _error_suggestion(e),
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please report this issue. Try running reindex.",
        })


@mcp.tool(
    description=(
        "Audit semantic search vs grep-equivalent for the same query. "
        "Runs both approaches and categorizes results as: "
        "'semantic_only' (found by vector but not grep), "
        "'grep_only' (found by grep but not vector), "
        "'both' (found by both). "
        "Use to validate plugin coverage, find semantic gaps, and compare approaches. "
        "Parameter: top_k (default 10) — how many results to fetch per method. "
        "Includes match counts per category and tokens_saved comparison."
    )
)
async def audit_search(query: str, top_k: int = 10) -> str:
    """Compare semantic search vs FTS (grep-equivalent), categorize results."""
    try:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return json.dumps({"error": "top_k must be an integer", "suggestion": "Pass top_k as an integer (e.g., top_k=10)"})

        start = time.perf_counter()
        config, storage, indexer, embed_provider, repo_path = _init_components()

        _idx = await _ensure_indexed(indexer, storage)
        index_status = _idx["status"]

        fetch_k = top_k

        def _run_vector():
            query_vector = embed_provider.embed_query(query)
            return storage.search_vector(query_vector, top_k=fetch_k)

        def _run_fts():
            return storage.search_fts(query, top_k=fetch_k)

        loop = asyncio.get_running_loop()
        vector_raw, fts_raw = await asyncio.gather(
            loop.run_in_executor(None, _run_vector),
            loop.run_in_executor(None, _run_fts),
        )

        def _audit_key(r: dict) -> tuple:
            return (r.get("file_path", ""), r.get("name", ""))

        def _to_item(r: dict) -> dict:
            return {
                "file_path": r.get("file_path", ""),
                "name": r.get("name", ""),
                "chunk_type": r.get("chunk_type", ""),
                "line_start": r.get("line_start", 0),
            }

        vec_map: dict[tuple, dict] = {_audit_key(r): _to_item(r) for r in vector_raw}
        fts_map: dict[tuple, dict] = {_audit_key(r): _to_item(r) for r in fts_raw}

        vec_keys = set(vec_map)
        fts_keys = set(fts_map)

        both_keys = vec_keys & fts_keys
        semantic_only_keys = vec_keys - fts_keys
        grep_only_keys = fts_keys - vec_keys

        semantic_only = [vec_map[k] for k in semantic_only_keys]
        grep_only = [fts_map[k] for k in grep_only_keys]
        both = [vec_map[k] for k in both_keys]

        tokens_saved = calculate_savings(
            [{"file_path": r.get("file_path", ""), "content": r.get("content", "")} for r in vector_raw],
            repo_path,
            index_path=storage.index_path,
        )

        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "audit_search: query=%r, both=%d, semantic_only=%d, grep_only=%d, time=%dms",
            query,
            len(both),
            len(semantic_only),
            len(grep_only),
            query_time_ms,
        )

        def _fmt_items(items, label):
            if not items:
                return [f"{label}: (none)"]
            out = [f"{label} ({len(items)}):"]
            for item in items:
                out.append(f"  - {item.get('name', '?')} ({item.get('chunk_type', '?')}) — {item.get('file_path', '')}:{item.get('line_start', 0)}")
            return out

        saved = tokens_saved.get("saved", 0)
        pct = tokens_saved.get("reduction_pct", 0)

        lines = [
            f"Audit: semantic vs grep for query",
            f"Both ({len(both)}) | Semantic only ({len(semantic_only)}) | Grep only ({len(grep_only)})",
            "",
        ]
        lines.extend(_fmt_items(both, "Found by both"))
        lines.extend(_fmt_items(semantic_only, "Semantic only (missed by grep)"))
        lines.extend(_fmt_items(grep_only, "Grep only (missed by semantic)"))
        lines.append(f"\nTokens saved: {saved:,} ({pct}% reduction) | Query: {query_time_ms}ms")

        return "\n".join(lines)

    except CTSError as e:
        return json.dumps({
            "error": str(e),
            "suggestion": _error_suggestion(e),
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "suggestion": "Please report this issue. Try running reindex.",
        })


def _error_suggestion(error: CTSError) -> str:
    """Map error types to helpful suggestions."""
    from .errors import (
        ConfigValidationError,
        EmbeddingProviderError,
        IndexNotFoundError,
        LockTimeoutError,
    )

    if isinstance(error, IndexNotFoundError):
        return "Run reindex to create the index."
    elif isinstance(error, EmbeddingProviderError):
        return "Check fastembed installation or switch embedding_mode in config."
    elif isinstance(error, LockTimeoutError):
        return "Another indexing operation may be in progress. Wait and retry."
    elif isinstance(error, ConfigValidationError):
        return "Check your config at ~/.claude-token-saver/config.yaml"
    return "Check plugin configuration and try again."


if __name__ == "__main__":
    mcp.run(transport="stdio")
