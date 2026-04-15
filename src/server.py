"""MCP server for claude-token-saver.

Entry point for the plugin. Registers all 7 MCP tools via FastMCP.
Instructions steer Claude Code to prefer plugin tools over native Grep/Read/Glob.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("cts.server")

from .config import Config
from .embeddings import create_embedding_provider
from .errors import CTSError
from .indexer import Indexer, _compute_file_hash, _find_repo_root
from .metrics import _load_metrics, calculate_savings
from .query_expansion import EXPANSION_DICT
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

# Boost multipliers applied to search_semantic scores by chunk type (post-scoring, pre-sort).
# Not applied to search_exact (user wants specific match) or search_hybrid (RRF handles ranking).
_TYPE_BOOST: dict[str, float] = {"class": 1.15, "function": 1.00, "method": 0.95}


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
                            index_status: str, idx_info: dict,
                            coverage: dict | None = None) -> str:
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

    if coverage is not None:
        response["coverage"] = coverage

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


def _expand_via_calls(
    results: list[dict],
    raw_results: list[dict],
    top_k: int,
    storage: Storage,
) -> list[dict]:
    """Expand results by following calls_json one hop.

    For each result, parses calls_json from the corresponding raw record, looks up
    each called function by exact name, and returns expanded chunks (not already in
    results) with score = parent_score * 0.8 and expanded=True.

    Limits total expansions to int(top_k * 0.5).
    """
    max_expansions = int(top_k * 0.5)
    existing_names = {r.get("name", "") for r in results}
    expanded: list[dict] = []

    for i, raw in enumerate(raw_results):
        if len(expanded) >= max_expansions:
            break

        calls_json_str = raw.get("calls_json", "[]")
        try:
            called_names = json.loads(calls_json_str) if calls_json_str else []
        except (json.JSONDecodeError, TypeError):
            called_names = []

        if not called_names:
            continue

        parent_score = results[i].get("score", 0.0) if i < len(results) else 0.0

        for name in called_names:
            if len(expanded) >= max_expansions:
                break
            if not name or name in existing_names:
                continue

            found = storage.search_by_name(name)
            if not found:
                continue

            rec = found[0]
            fp = rec.get("file_path", "")
            file_hash = rec.get("file_hash", "")
            stale = storage.is_stale(fp, file_hash) if fp and file_hash else False

            expanded.append({
                "file_path": fp,
                "name": rec.get("name", ""),
                "chunk_type": rec.get("chunk_type", ""),
                "line_start": rec.get("line_start", 0),
                "line_end": rec.get("line_end", 0),
                "content": rec.get("content", ""),
                "score": round(parent_score * 0.8, 6),
                "language": rec.get("language", ""),
                "stale": stale,
                "parent_name": rec.get("parent_name", ""),
                "expanded": True,
            })
            existing_names.add(name)

    return expanded


def _expand_via_callers(
    results: list[dict],
    top_k: int,
    storage: Storage,
) -> list[dict]:
    """Expand results with reverse-index: find who calls each result.

    For each result, looks up callers via the pre-built callers.json index.
    Returns new chunks (not already in results) with score = parent_score * 0.75
    and caller_of=<name>.

    Limits:
    - Total caller expansions: 10
    - Skips names that have 50+ callers (too generic, e.g. 'init', 'build')
    """
    max_callers = 10
    existing_names = {r.get("name", "") for r in results}
    expanded: list[dict] = []

    generic_cache: dict[str, bool] = {}

    def _is_generic(name: str) -> bool:
        if name in generic_cache:
            return generic_cache[name]
        caller_list = storage.get_callers(name)
        result = len(caller_list) >= 50
        generic_cache[name] = result
        return result

    for res in results:
        if len(expanded) >= max_callers:
            break

        res_name = res.get("name", "")
        if not res_name:
            continue

        if _is_generic(res_name):
            continue

        parent_score = res.get("score", 0.0)
        caller_names = storage.get_callers(res_name)

        for caller_name in caller_names:
            if len(expanded) >= max_callers:
                break
            if not caller_name or caller_name in existing_names:
                continue

            found = storage.search_by_name(caller_name)
            if not found:
                continue

            rec = found[0]
            fp = rec.get("file_path", "")
            file_hash = rec.get("file_hash", "")
            stale = storage.is_stale(fp, file_hash) if fp and file_hash else False

            expanded.append({
                "file_path": fp,
                "name": rec.get("name", ""),
                "chunk_type": rec.get("chunk_type", ""),
                "line_start": rec.get("line_start", 0),
                "line_end": rec.get("line_end", 0),
                "content": rec.get("content", ""),
                "score": round(parent_score * 0.75, 6),
                "language": rec.get("language", ""),
                "stale": stale,
                "parent_name": rec.get("parent_name", ""),
                "expanded": True,
                "caller_of": res_name,
            })
            existing_names.add(caller_name)

    return expanded


def _expand_module_siblings(
    results: list[dict],
    top_k: int,
    storage: Storage,
    filter_ext: str | None = None,
) -> list[dict]:
    """Add co-located chunks from directories that have 2+ results (strong signal).

    For each directory with 2+ results, queries the index for OTHER chunks in
    that same directory not already in results. Adds them with module_match=True
    and score = max_result_score * 0.7.

    Limits: max 5 co-located files added total; prefers class chunks over others.
    filter_ext: if provided, only siblings matching that extension are added.
    """
    # Count results per directory
    dir_counts: dict[str, int] = {}
    for r in results:
        fp = r.get("file_path", "")
        parent = str(Path(fp).parent) if fp else ""
        dir_counts[parent] = dir_counts.get(parent, 0) + 1

    # Directories with strong signal (2+ results)
    strong_dirs = {d for d, cnt in dir_counts.items() if cnt >= 2}
    if not strong_dirs:
        return []

    existing_files = {r.get("file_path", "") for r in results}

    max_result_score = max((r.get("score", 0.0) for r in results), default=0.0)
    sibling_score = round(max_result_score * 0.7, 6)

    # Normalize extension for filtering (ensure it starts with '.')
    ext_filter = None
    if filter_ext:
        ext_filter = filter_ext if filter_ext.startswith(".") else f".{filter_ext}"

    added_files: set[str] = set()
    siblings: list[dict] = []

    for directory in strong_dirs:
        if len(added_files) >= 5:
            break

        candidates = storage.get_chunks_by_directory(directory)
        # Filter out already-present files; keep one chunk per file
        new_by_file: dict[str, dict] = {}
        for c in candidates:
            fp = c.get("file_path", "")
            if fp in existing_files or fp in added_files:
                continue
            # Respect filter_ext if provided
            if ext_filter and not fp.endswith(ext_filter):
                continue
            # Prefer class chunks; only store one representative chunk per file
            existing = new_by_file.get(fp)
            if existing is None:
                new_by_file[fp] = c
            elif c.get("chunk_type") == "class" and existing.get("chunk_type") != "class":
                new_by_file[fp] = c

        # Sort: class first, then others
        candidates_sorted = sorted(
            new_by_file.values(),
            key=lambda c: (0 if c.get("chunk_type") == "class" else 1),
        )

        for c in candidates_sorted:
            if len(added_files) >= 5:
                break
            fp = c.get("file_path", "")
            file_hash = c.get("file_hash", "")
            stale = storage.is_stale(fp, file_hash) if fp and file_hash else False
            siblings.append({
                "file_path": fp,
                "name": c.get("name", ""),
                "chunk_type": c.get("chunk_type", ""),
                "line_start": c.get("line_start", 0),
                "line_end": c.get("line_end", 0),
                "content": c.get("content", ""),
                "score": sibling_score,
                "language": c.get("language", ""),
                "stale": stale,
                "parent_name": c.get("parent_name", ""),
                "module_match": True,
            })
            added_files.add(fp)

    return siblings


_CROSS_LAYER_SUFFIXES = re.compile(
    r'(Service|Controller|Provider|Widget|State|Screen|Handler|Guard)$',
    re.IGNORECASE,
)

_CAMEL_TO_SNAKE_RE = re.compile(r'(?<=[a-z0-9])([A-Z])|(?<=[A-Z])([A-Z][a-z])')


def _normalize_name(name: str) -> str:
    """Normalize a symbol name for cross-layer matching.

    1. Convert CamelCase to snake_case.
    2. Strip one common architectural suffix from the end (single pass).
    3. Return lowercase with underscores.
    """
    s = _CAMEL_TO_SNAKE_RE.sub(lambda m: '_' + (m.group(1) or m.group(2)), name)
    s = s.lower().replace('-', '_')
    # Strip suffix exactly once â only the outermost suffix is architectural
    stripped = _CROSS_LAYER_SUFFIXES.sub('', s, count=1).rstrip('_')
    if stripped:
        s = stripped
    return s


def _expand_cross_layer(results: list[dict], storage: Storage) -> list[dict]:
    """Expand results with matching symbols from other languages (cross-layer).

    For each result, normalizes the name and searches for chunks with the same
    normalized name in OTHER languages. Found matches are added with
    cross_layer=True and score = original_score * 0.7.

    Deduplicates: skips if already present in results.
    """
    existing_keys: set[tuple] = {
        (r.get("file_path", ""), r.get("name", "")) for r in results
    }

    cross_layer: list[dict] = []

    for result in results:
        source_lang = result.get("language", "")
        source_name = result.get("name", "")
        source_score = result.get("score", 0.0)

        if not source_name:
            continue

        normalized = _normalize_name(source_name)
        if not normalized:
            continue

        # Search by exact source name, normalized name, PascalCase, and FTS on normalized term
        # FTS on the normalized term finds variants like SubscriptionService when searching "subscription"
        candidates_to_try: list[str] = [source_name]
        if normalized != source_name.lower():
            candidates_to_try.append(normalized)
        pascal = ''.join(w.capitalize() for w in normalized.split('_') if w)
        if pascal and pascal != source_name:
            candidates_to_try.append(pascal)

        found_chunks: list[dict] = []
        seen_candidate: set[str] = set()
        for candidate in candidates_to_try:
            if candidate in seen_candidate:
                continue
            seen_candidate.add(candidate)
            found_chunks.extend(storage.search_by_name(candidate))
        # Also search FTS with normalized base term to catch suffix variants
        fts_hits = storage.search_fts(normalized, top_k=10)
        found_chunks.extend(fts_hits)

        for rec in found_chunks:
            rec_lang = rec.get("language", "")
            rec_name = rec.get("name", "")
            rec_fp = rec.get("file_path", "")

            # Only cross-layer: skip same language
            if rec_lang == source_lang:
                continue

            key = (rec_fp, rec_name)
            if key in existing_keys:
                continue


            # Verify normalized names match (exact or prefix relationship)
            rec_normalized = _normalize_name(rec_name)
            # Allow match if one normalized form is a prefix of the other (min 5 chars)
            n1, n2 = normalized, rec_normalized
            n_match = (
                n1 == n2
                or (len(n1) >= 5 and n2.startswith(n1))
                or (len(n2) >= 5 and n1.startswith(n2))
            )
            if not n_match:
                continue

            existing_keys.add(key)
            cross_layer.append({
                "file_path": rec_fp,
                "name": rec_name,
                "chunk_type": rec.get("chunk_type", ""),
                "line_start": rec.get("line_start", 0),
                "line_end": rec.get("line_end", 0),
                "content": rec.get("content", ""),
                "score": round(source_score * 0.7, 6),
                "language": rec_lang,
                "stale": False,
                "parent_name": rec.get("parent_name", ""),
                "cross_layer": True,
            })

    return cross_layer


def _expand_test_files(results: list[dict], storage: Storage) -> list[dict]:
    """Expand results with associated test files.

    For each result, derives the base filename and searches for test file patterns:
    {name}_test, test_{name}, {name}.test, {name}.spec.

    Found test chunks are added with test_for=original_file_path,
    score = source_score * 0.6, at the END of results.

    Deduplicates: skips if already present in results.
    """
    existing_fps: set[str] = {r.get("file_path", "") for r in results}
    test_results: list[dict] = []
    seen_test_fps: set[str] = set()

    for result in results:
        fp = result.get("file_path", "")
        source_score = result.get("score", 0.0)

        if not fp:
            continue

        base = Path(fp).stem
        base_lower = base.lower()
        patterns = [
            f"{base}_test",
            f"test_{base}",
            f"{base}.test",
            f"{base}.spec",
        ]

        for pattern in patterns:
            found = storage.search_fts(pattern, top_k=3)
            for rec in found:
                rec_fp = rec.get("file_path", "")
                if not rec_fp:
                    continue
                if rec_fp in existing_fps or rec_fp in seen_test_fps:
                    continue

                rec_stem = Path(rec_fp).stem.lower()
                is_test_match = (
                    rec_stem == f"{base_lower}_test"
                    or rec_stem == f"test_{base_lower}"
                    or rec_stem == f"{base_lower}.test"
                    or rec_stem == f"{base_lower}.spec"
                )
                if not is_test_match:
                    continue

                seen_test_fps.add(rec_fp)
                test_results.append({
                    "file_path": rec_fp,
                    "name": rec.get("name", ""),
                    "chunk_type": rec.get("chunk_type", ""),
                    "line_start": rec.get("line_start", 0),
                    "line_end": rec.get("line_end", 0),
                    "content": rec.get("content", ""),
                    "score": round(source_score * 0.6, 6),
                    "language": rec.get("language", ""),
                    "stale": False,
                    "parent_name": rec.get("parent_name", ""),
                    "test_for": fp,
                })

    return test_results


def _rerank_results(results: list[dict]) -> list[dict]:
    """Post-process results with diversity and quality heuristics.

    Rules applied in order:
    1. Short chunk penalty: chunks with < 3 lines get score * 0.5.
    2. Co-location boost: if 2+ results share the same parent directory,
       all results from that directory get score * 1.05.
    3. File diversity: max 2 chunks per file; extras (lowest score) are dropped.
    4. Re-sort by score descending.
    """
    if not results:
        return results

    # Rule 1: short chunk penalty
    for r in results:
        content = r.get("content", "")
        lines = [ln for ln in content.split("\n") if ln.strip()]
        if len(lines) < 3:
            r["score"] = round(r["score"] * 0.5, 6)

    # Rule 2: co-location boost — count results per parent directory
    dir_counts: dict[str, int] = {}
    for r in results:
        fp = r.get("file_path", "")
        parent = str(Path(fp).parent) if fp else ""
        dir_counts[parent] = dir_counts.get(parent, 0) + 1

    for r in results:
        fp = r.get("file_path", "")
        parent = str(Path(fp).parent) if fp else ""
        if dir_counts.get(parent, 0) >= 2:
            r["score"] = round(r["score"] * 1.05, 6)

    # Rule 3: file diversity — keep top 2 per file by score
    results.sort(key=lambda x: x["score"], reverse=True)
    file_counts: dict[str, int] = {}
    kept = []
    for r in results:
        fp = r.get("file_path", "")
        count = file_counts.get(fp, 0)
        if count < 2:
            kept.append(r)
            file_counts[fp] = count + 1

    # Rule 4: re-sort
    kept.sort(key=lambda x: x["score"], reverse=True)
    return kept


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

        # Embed the query — optionally dual-vector via technical expansion
        technical_query = _expand_query_to_technical(query)

        fetch_k = top_k * 2 if technical_query else top_k

        def _run_original_vector():
            vec = embed_provider.embed_query(query)
            return storage.search_vector(vec, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path)

        if technical_query:
            def _run_technical_vector():
                vec = embed_provider.embed_query(technical_query)
                return storage.search_vector(vec, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path)

            loop = asyncio.get_running_loop()
            original_raw, technical_raw = await asyncio.gather(
                loop.run_in_executor(None, _run_original_vector),
                loop.run_in_executor(None, _run_technical_vector),
            )
            raw_merged = _rrf_merge(original_raw, technical_raw, fetch_k, top_k, storage)
            # _rrf_merge already builds dicts; extract accepted_raw for graph expansion
            accepted_raw = raw_merged
            results = raw_merged
        else:
            raw_results = _run_original_vector()

            # Build results with stale check; apply score_threshold filter (0.45 for cross-lingual queries)
            results = []
            accepted_raw = []
            for r in raw_results:
                score = round(1.0 - r.get("_distance", 0.0), 4)
                if score < 0.45:
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
                accepted_raw.append(r)

            # Apply chunk type boost and re-sort by boosted score
            for res in results:
                res["score"] = round(res["score"] * _TYPE_BOOST.get(res["chunk_type"], 1.0), 4)
            results.sort(key=lambda x: x["score"], reverse=True)

        # Graph expansion: follow calls_json one hop
        expanded = _expand_via_calls(results, accepted_raw, top_k, storage)
        results = results + expanded

        # Reverse expansion: find callers of each result (1 hop)
        callers_expanded = _expand_via_callers(results, top_k, storage)
        results = results + callers_expanded

        # Module clustering: add co-located chunks from directories with 2+ results
        siblings = _expand_module_siblings(results, top_k, storage, filter_ext=filter_ext)
        results = results + siblings

        # Cross-layer boost: find matching symbols in other languages
        cross_layer = _expand_cross_layer(results, storage)
        results = results + cross_layer

        # Test file association: find test files for source files in results
        test_files = _expand_test_files(results, storage)

        # Re-ranking heuristic: diversity + quality adjustments
        results = _rerank_results(results)

        # Test files appended at the END, after reranking
        results = results + test_files

        # Calculate token savings (persists cumulative metrics)
        tokens_saved = calculate_savings(results, repo_path, index_path=storage.index_path)

        coverage = _build_coverage(results, storage, query)
        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info("search_semantic: query=%r, results=%d, time=%dms", query, len(results), query_time_ms)

        return _format_search_response(results, tokens_saved, query_time_ms, index_status, _idx, coverage)

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

        # Build results with stale check; no score threshold (FTS relevance is by rank)
        results = []
        for r in raw_results:
            score = r.get("_score", r.get("score", 0.0))

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

        coverage = _build_coverage(results, storage, query)
        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info("search_exact: query=%r, results=%d, time=%dms", query, len(results), query_time_ms)

        return _format_search_response(results, tokens_saved, query_time_ms, index_status, _idx, coverage)

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


_STOP_WORDS_PT = frozenset({
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sob", "sobre",
    "e", "ou", "mas", "que", "se", "como", "quando",
    "ao", "aos", "pelo", "pela", "pelos", "pelas",
    "este", "esta", "esse", "essa", "aquele", "aquela",
    "isto", "isso", "aquilo",
    "eu", "tu", "ele", "ela", "nos", "vos", "eles", "elas",
    "meu", "minha", "seu", "sua", "nosso", "nossa",
    "ser", "estar", "ter", "haver", "fazer", "ir",
    "nao", "mais", "muito", "tambem", "ja", "ainda",
})

_STOP_WORDS_EN = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "if", "then", "than",
    "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "our", "their",
    "what", "which", "who", "how", "when", "where", "why",
})

_IDENTIFIER_RE = re.compile(r'[a-z][a-zA-Z0-9]*[A-Z]|[a-zA-Z_]\w*_\w+|[A-Z][a-z]+[A-Z]')

_STOP_WORDS = _STOP_WORDS_PT | _STOP_WORDS_EN


def _filter_stop_words(query: str) -> str:
    if not query:
        return query
    tokens = query.split()
    if len(tokens) <= 5:
        return query
    filtered = []
    for token in tokens:
        if _IDENTIFIER_RE.search(token):
            filtered.append(token)
        elif token.lower() not in _STOP_WORDS:
            filtered.append(token)
    if not filtered:
        return query
    return " ".join(filtered)


_RRF_K = 60


def _expand_query_to_technical(query: str) -> str | None:
    """Expand a query to an English technical reformulation using a static dictionary.

    Extracts significant terms (after stop-word removal), maps each to its
    English code equivalents, and concatenates them into a reformulation string.

    Returns None if no dictionary terms matched (caller should skip dual-vector search).
    """
    tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
    expanded_terms: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if token in _STOP_WORDS:
            continue
        expansions = EXPANSION_DICT.get(token)
        if expansions:
            for term in expansions:
                if term not in seen:
                    seen.add(term)
                    expanded_terms.append(term)

    if not expanded_terms:
        return None

    return " ".join(expanded_terms)


def _dedup_key(r: dict) -> tuple:
    return (r.get("file_path", ""), r.get("name", ""), r.get("line_start", 0))


def _rrf_merge(
    vector_raw: list[dict],
    fts_raw: list[dict],
    fetch_k: int,
    top_k: int,
    storage: Storage,
) -> list[dict]:
    absent_rank = fetch_k + 1

    vec_rank: dict[tuple, tuple[int, dict]] = {}
    for rank, r in enumerate(vector_raw):
        key = _dedup_key(r)
        if key not in vec_rank:
            vec_rank[key] = (rank, r)

    fts_rank: dict[tuple, tuple[int, dict]] = {}
    for rank, r in enumerate(fts_raw):
        key = _dedup_key(r)
        if key not in fts_rank:
            fts_rank[key] = (rank, r)

    all_keys = set(vec_rank) | set(fts_rank)
    scored: list[tuple[float, dict]] = []

    for key in all_keys:
        v_rank = vec_rank[key][0] if key in vec_rank else absent_rank
        f_rank = fts_rank[key][0] if key in fts_rank else absent_rank
        rrf_score = 1.0 / (_RRF_K + v_rank) + 1.0 / (_RRF_K + f_rank)

        if key in vec_rank:
            rec = vec_rank[key][1]
        else:
            rec = fts_rank[key][1]

        fp = rec.get("file_path", "")
        file_hash = rec.get("file_hash", "")
        stale = storage.is_stale(fp, file_hash) if fp and file_hash else False

        scored.append((rrf_score, {
            "file_path": fp,
            "name": rec.get("name", ""),
            "chunk_type": rec.get("chunk_type", ""),
            "line_start": rec.get("line_start", 0),
            "line_end": rec.get("line_end", 0),
            "content": rec.get("content", ""),
            "score": round(rrf_score, 6),
            "language": rec.get("language", ""),
            "stale": stale,
            "parent_name": rec.get("parent_name", ""),
            "calls_json": rec.get("calls_json", "[]"),
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


@mcp.tool(
    description=(
        "Hybrid search combining semantic (vector) and exact (FTS) approaches "
        "via Reciprocal Rank Fusion (RRF) for maximum recall. "
        "USE INSTEAD OF Grep for maximum recall — combines conceptual matches "
        "AND exact name/token matches in one query. Returns merged, de-duplicated "
        "results ranked by RRF score. USE INSTEAD OF running search_semantic "
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

        fts_query = _filter_stop_words(query)
        technical_query = _expand_query_to_technical(query)

        def _run_fts():
            return storage.search_fts(
                fts_query, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path
            )

        loop = asyncio.get_running_loop()

        if technical_query:
            def _run_vector_original():
                vec = embed_provider.embed_query(query)
                return storage.search_vector(vec, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path)

            def _run_vector_technical():
                vec = embed_provider.embed_query(technical_query)
                return storage.search_vector(vec, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path)

            vec_original_raw, vec_technical_raw, fts_raw = await asyncio.gather(
                loop.run_in_executor(None, _run_vector_original),
                loop.run_in_executor(None, _run_vector_technical),
                loop.run_in_executor(None, _run_fts),
            )
            # Merge the two semantic vectors first, then merge with FTS
            vector_raw = _rrf_merge(vec_original_raw, vec_technical_raw, fetch_k, fetch_k, storage)
            results = _rrf_merge(vector_raw, fts_raw, fetch_k, top_k, storage)
        else:
            def _run_vector():
                vec = embed_provider.embed_query(query)
                return storage.search_vector(vec, top_k=fetch_k, filter_ext=filter_ext, filter_path=filter_path)

            vector_raw, fts_raw = await asyncio.gather(
                loop.run_in_executor(None, _run_vector),
                loop.run_in_executor(None, _run_fts),
            )
            results = _rrf_merge(vector_raw, fts_raw, fetch_k, top_k, storage)

        # Graph expansion: follow calls_json one hop (results include calls_json from _rrf_merge)
        expanded = _expand_via_calls(results, results, top_k, storage)
        results = results + expanded

        # Reverse expansion: find callers of each result (1 hop)
        callers_expanded = _expand_via_callers(results, top_k, storage)
        results = results + callers_expanded

        # Module clustering: add co-located chunks from directories with 2+ results
        siblings = _expand_module_siblings(results, top_k, storage, filter_ext=filter_ext)
        results = results + siblings

        # Cross-layer boost: find matching symbols in other languages
        cross_layer = _expand_cross_layer(results, storage)
        results = results + cross_layer

        # Test file association: find test files for source files in results
        test_files = _expand_test_files(results, storage)

        # Re-ranking heuristic: diversity + quality adjustments
        results = _rerank_results(results)

        # Test files appended at the END, after reranking
        results = results + test_files

        tokens_saved = calculate_savings(results, repo_path, index_path=storage.index_path)
        coverage = _build_coverage(results, storage, query)
        query_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info("search_hybrid: query=%r, results=%d, time=%dms", query, len(results), query_time_ms)

        return _format_search_response(results, tokens_saved, query_time_ms, index_status, _idx, coverage)

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


def _build_coverage(results: list[dict], storage: Storage, query: str) -> dict:
    all_meta = storage.get_all_chunks_metadata()
    languages_indexed = sorted({m["language"] for m in all_meta if m.get("language")})
    languages_with_results = sorted({r["language"] for r in results if r.get("language")})

    if not languages_indexed:
        return {
            "languages_indexed": [],
            "languages_with_results": languages_with_results,
            "confidence": "low",
            "suggestion": "",
        }

    scores = [r.get("score", 0.0) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    total = len(languages_indexed)
    covered = len(languages_with_results)
    coverage_ratio = covered / total if total > 0 else 0.0

    if coverage_ratio >= 0.8 and avg_score > 0.7:
        confidence = "high"
    elif coverage_ratio >= 0.5 or 0.5 <= avg_score <= 0.7:
        confidence = "partial"
    else:
        confidence = "low"

    suggestion = ""
    if confidence != "high" and len(query.split()) > 1:
        missing = sorted(set(languages_indexed) - set(languages_with_results))
        if missing:
            covered_str = ", ".join(languages_with_results) if languages_with_results else "nenhuma linguagem"
            suggestion = (
                f"Resultados cobrem apenas {covered_str}. "
                "Para cobertura cross-language, considere Explorer."
            )
        else:
            suggestion = "Para cobertura completa, considere Explorer."

    return {
        "languages_indexed": languages_indexed,
        "languages_with_results": languages_with_results,
        "confidence": confidence,
        "suggestion": suggestion,
    }


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
