"""Benchmark runner for claude-token-saver search quality evaluation.

Executes golden set queries against search tools directly via the plugin's
Python API (not via MCP). Produces JSON reports and stdout summaries.

Usage:
    cd products/plugin
    uv run python -m tests.benchmarks.runner --golden-set golden_sets/cts_internal.yaml
    uv run python -m tests.benchmarks.runner --golden-set golden_sets/cts_internal.yaml --output reports/run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import Config
from src.embeddings import create_embedding_provider
from src.indexer import Indexer
from src.server import search_exact, search_hybrid, search_semantic
from src.storage import Storage

from .scoring import compute_all_metrics, coverage_score

logger = logging.getLogger("cts.benchmark.runner")

# Directory of this file
_HERE = Path(__file__).parent

# Estimated tokens for naive grep/cat exploration (baseline for TER)
_NAIVE_EXPLORER_TOKENS = 5000


def _load_golden_set(golden_set_path: Path) -> dict:
    """Load and return the golden set YAML as a dict."""
    with open(golden_set_path) as f:
        return yaml.safe_load(f)


def _parse_result(raw_json: str) -> list[dict]:
    """Parse search tool JSON response into a list of result dicts."""
    try:
        data = json.loads(raw_json)
        return data.get("results", [])
    except (json.JSONDecodeError, TypeError):
        return []


def _estimate_tokens(results: list[dict]) -> int:
    """Estimate tokens used by the plugin for this response."""
    response_json = json.dumps(results, ensure_ascii=False)
    return max(1, len(response_json) // 4)


async def _run_single_query(
    query_entry: dict,
    method: str,
    config: Config,
    storage: Storage,
    indexer: Indexer,
    embed_provider: Any,
    repo_path: Path,
    k: int = 10,
) -> dict:
    """Run one (query, method) pair and return scored result dict."""
    import src.server as srv

    query_text = query_entry["query"]
    expected = query_entry.get("expected_results", [])
    required_dirs = query_entry.get("coverage_dirs", [])

    # Temporarily inject our test components into server globals
    old_state = (
        srv._config,
        srv._storage,
        srv._indexer,
        srv._embed_provider,
        srv._repo_path,
    )
    srv._config = config
    srv._storage = storage
    srv._indexer = indexer
    srv._embed_provider = embed_provider
    srv._repo_path = repo_path

    try:
        t0 = time.perf_counter()

        if method == "semantic":
            raw_json = await search_semantic(query_text, top_k=k)
        elif method == "exact":
            raw_json = await search_exact(query_text, top_k=k)
        elif method == "hybrid":
            raw_json = await search_hybrid(query_text, top_k=k)
        else:
            raise ValueError(f"Unknown search method: {method}")

        latency_ms = int((time.perf_counter() - t0) * 1000)

    finally:
        (
            srv._config,
            srv._storage,
            srv._indexer,
            srv._embed_provider,
            srv._repo_path,
        ) = old_state

    returned = _parse_result(raw_json)
    tokens_plugin = _estimate_tokens(returned)

    metrics = compute_all_metrics(
        returned=returned,
        expected=expected,
        k=k,
        tokens_plugin=tokens_plugin,
        tokens_explorer=_NAIVE_EXPLORER_TOKENS,
    )

    # Add coverage score (uses coverage_dirs from golden set entry)
    metrics["coverage"] = round(
        coverage_score(returned, required_dirs), 4
    )
    metrics["latency_ms"] = latency_ms

    return {
        "query_id": query_entry["id"],
        "query": query_text,
        "category": query_entry.get("category", ""),
        "difficulty": query_entry.get("difficulty", ""),
        "method": method,
        "metrics": metrics,
        "n_returned": len(returned),
        "n_expected_relevant": sum(
            1 for e in expected if e.get("relevance", 0) >= 2
        ),
    }


async def run_benchmark(
    golden_set_path: Path,
    repo_path: Path | None = None,
    k: int = 10,
) -> dict:
    """Run the full benchmark suite against a golden set YAML.

    Args:
        golden_set_path: Path to the golden set YAML file.
        repo_path: Repository to search. Defaults to project root detected from cwd.
        k: Rank cutoff for metrics.

    Returns:
        Full report dict with per-query results and aggregated stats.
    """
    golden = _load_golden_set(golden_set_path)
    queries = golden.get("queries", [])
    methods = ["semantic", "exact", "hybrid"]

    # Determine repo path
    if repo_path is None:
        # Walk up from runner location to find the plugin root, then the repo root
        plugin_root = _HERE.parent.parent  # products/plugin/
        repo_root = plugin_root.parent.parent  # monorepo root
        repo_path = repo_root

    # Initialize search components
    config = Config(embedding_mode="lite")
    storage = Storage(config, repo_path, base_path=None)
    embed_provider = create_embedding_provider(config)
    indexer = Indexer(config, storage, embed_provider, repo_path=repo_path)

    # Ensure index is up to date
    hashes = storage.get_file_hashes()
    if not hashes:
        logger.info("Building index for benchmark...")
        await indexer.reindex(force=False)

    results: list[dict] = []

    for query_entry in queries:
        for method in methods:
            logger.info(
                "Running query=%s method=%s", query_entry["id"], method
            )
            result = await _run_single_query(
                query_entry=query_entry,
                method=method,
                config=config,
                storage=storage,
                indexer=indexer,
                embed_provider=embed_provider,
                repo_path=repo_path,
                k=k,
            )
            results.append(result)

    report = _build_report(golden, results, k)
    return report


def _build_report(golden: dict, results: list[dict], k: int) -> dict:
    """Aggregate per-query results into a benchmark report."""
    methods = ["semantic", "exact", "hybrid"]
    metric_keys = [
        f"precision@{k}",
        f"recall@{k}",
        f"f1@{k}",
        "mrr",
        f"ndcg@{k}",
        "coverage",
        "token_efficiency_ratio",
        "latency_ms",
    ]

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    # Per-method aggregation
    per_method: dict[str, dict] = {}
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        per_method[method] = {}
        for mk in metric_keys:
            vals = [r["metrics"].get(mk, 0.0) for r in method_results]
            per_method[method][mk] = _avg(vals)

    # Per-category aggregation
    categories = sorted({r["category"] for r in results})
    per_category: dict[str, dict] = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        per_category[cat] = {}
        for method in methods:
            method_cat = [r for r in cat_results if r["method"] == method]
            per_category[cat][method] = {}
            for mk in metric_keys:
                vals = [r["metrics"].get(mk, 0.0) for r in method_cat]
                per_category[cat][method][mk] = _avg(vals)

    # Per-difficulty aggregation
    difficulties = sorted({r["difficulty"] for r in results})
    per_difficulty: dict[str, dict] = {}
    for diff in difficulties:
        diff_results = [r for r in results if r["difficulty"] == diff]
        per_difficulty[diff] = {}
        for method in methods:
            method_diff = [r for r in diff_results if r["method"] == method]
            per_difficulty[diff][method] = {}
            for mk in metric_keys:
                vals = [r["metrics"].get(mk, 0.0) for r in method_diff]
                per_difficulty[diff][method][mk] = _avg(vals)

    return {
        "meta": {
            "golden_set": golden.get("repo", "unknown"),
            "n_queries": len({r["query_id"] for r in results}),
            "n_methods": len(methods),
            "k": k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "per_method": per_method,
        "per_category": per_category,
        "per_difficulty": per_difficulty,
        "raw_results": results,
    }


def _print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    meta = report["meta"]
    k = meta["k"]
    print(
        f"\n=== Benchmark Report: {meta['golden_set']} "
        f"({meta['n_queries']} queries, k={k}) ==="
    )
    print(f"Timestamp: {meta['timestamp']}\n")

    print("Per-method averages:")
    header = f"{'Method':<12} {'P@k':>7} {'R@k':>7} {'F1@k':>7} {'MRR':>7} {'NDCG@k':>8} {'Coverage':>10} {'Lat(ms)':>9}"
    print(header)
    print("-" * len(header))
    for method, metrics in report["per_method"].items():
        print(
            f"{method:<12} "
            f"{metrics.get(f'precision@{k}', 0):.3f}   "
            f"{metrics.get(f'recall@{k}', 0):.3f}   "
            f"{metrics.get(f'f1@{k}', 0):.3f}   "
            f"{metrics.get('mrr', 0):.3f}   "
            f"{metrics.get(f'ndcg@{k}', 0):.4f}   "
            f"{metrics.get('coverage', 0):.4f}   "
            f"{metrics.get('latency_ms', 0):>7.0f}"
        )

    print("\nPer-category F1@k (hybrid):")
    for cat, methods in report["per_category"].items():
        f1 = methods.get("hybrid", {}).get(f"f1@{k}", 0)
        print(f"  {cat:<28} F1={f1:.3f}")


def _save_report(report: dict, output_path: Path) -> None:
    """Write report to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {output_path}")


async def _main(args: argparse.Namespace) -> None:
    golden_set_path = _HERE / args.golden_set
    repo_path = Path(args.repo_path).resolve() if args.repo_path else None

    report = await run_benchmark(
        golden_set_path=golden_set_path,
        repo_path=repo_path,
        k=args.k,
    )

    _print_summary(report)

    output_path = (
        Path(args.output)
        if args.output
        else _HERE
        / "reports"
        / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    _save_report(report, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="CTS benchmark runner")
    parser.add_argument(
        "--golden-set",
        default="golden_sets/cts_internal.yaml",
        help="Path to golden set YAML (relative to tests/benchmarks/)",
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to repository to search (default: auto-detect)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Rank cutoff for metrics (default: 10)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for JSON report (default: reports/run_<timestamp>.json)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))
