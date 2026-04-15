"""Scoring metrics for search quality benchmarks.

Pure functions — no side effects, no imports from src/.
All metrics operate on ranked lists of result dicts and ground-truth sets.

Relevance threshold for binary precision/recall: >= 2.
Graded relevance scale for NDCG: 0-3.
"""

from __future__ import annotations

import math


def precision_at_k(
    returned: list[dict],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of top-K returned results that are relevant.

    Args:
        returned: Ranked list of result dicts (must have 'name' key).
        relevant: Set of relevant result names (relevance >= 2).
        k: Cutoff rank.

    Returns:
        Float in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = returned[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r.get("name", "") in relevant)
    return hits / len(top_k)


def recall_at_k(
    returned: list[dict],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of all relevant results found in top-K.

    Args:
        returned: Ranked list of result dicts (must have 'name' key).
        relevant: Set of relevant result names (relevance >= 2).
        k: Cutoff rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = returned[:k]
    hits = sum(1 for r in top_k if r.get("name", "") in relevant)
    return hits / len(relevant)


def f1_at_k(
    returned: list[dict],
    relevant: set[str],
    k: int,
) -> float:
    """Harmonic mean of precision@k and recall@k.

    Args:
        returned: Ranked list of result dicts.
        relevant: Set of relevant result names.
        k: Cutoff rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if both precision and recall are 0.
    """
    p = precision_at_k(returned, relevant, k)
    r = recall_at_k(returned, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(
    returned: list[dict],
    relevant: set[str],
) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result.

    Args:
        returned: Ranked list of result dicts.
        relevant: Set of relevant result names.

    Returns:
        Float in [0, 1]. Returns 0.0 if no relevant result found.
    """
    for rank, r in enumerate(returned, start=1):
        if r.get("name", "") in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    returned: list[dict],
    relevance_map: dict[str, int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain using graded relevance (0-3).

    Args:
        returned: Ranked list of result dicts.
        relevance_map: Dict mapping result name -> relevance grade (0-3).
        k: Cutoff rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if ideal DCG is 0.
    """
    if k <= 0:
        return 0.0

    def _dcg(ranked_grades: list[int]) -> float:
        total = 0.0
        for i, grade in enumerate(ranked_grades[:k], start=1):
            total += (2**grade - 1) / math.log2(i + 1)
        return total

    actual_grades = [
        relevance_map.get(r.get("name", ""), 0) for r in returned[:k]
    ]

    # Ideal: sort all known relevant items by grade descending
    ideal_grades = sorted(relevance_map.values(), reverse=True)

    dcg_val = _dcg(actual_grades)
    idcg_val = _dcg(ideal_grades)

    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def coverage_score(
    returned: list[dict],
    required_dirs: list[str],
) -> float:
    """Fraction of required directories that have at least one result.

    Args:
        returned: Ranked list of result dicts (must have 'file_path' key).
        required_dirs: List of directory path prefixes that must be covered.

    Returns:
        Float in [0, 1]. Returns 1.0 if required_dirs is empty.
    """
    if not required_dirs:
        return 1.0

    covered = 0
    for req_dir in required_dirs:
        for r in returned:
            fp = r.get("file_path", "")
            if req_dir in fp:
                covered += 1
                break

    return covered / len(required_dirs)


def token_efficiency_ratio(
    f1: float,
    tokens_plugin: int,
    tokens_explorer: int,
) -> float:
    """Ratio of search quality (F1) to token cost relative to naive exploration.

    TER = F1 / (tokens_plugin / tokens_explorer)

    Higher is better: high quality at low token cost.
    Returns 0.0 if tokens_explorer is 0 or denominator would be 0.

    Args:
        f1: F1 score for the search method.
        tokens_plugin: Tokens consumed by the plugin search.
        tokens_explorer: Tokens consumed by naive grep/cat exploration.

    Returns:
        Float >= 0.
    """
    if tokens_explorer <= 0 or tokens_plugin <= 0:
        return 0.0
    ratio = tokens_plugin / tokens_explorer
    if ratio == 0:
        return 0.0
    return f1 / ratio


def compute_all_metrics(
    returned: list[dict],
    expected: list[dict],
    k: int = 10,
    tokens_plugin: int = 0,
    tokens_explorer: int = 0,
) -> dict:
    """Compute all 7 metrics for a single (query, method) pair.

    Args:
        returned: Ranked list of result dicts from the search tool.
        expected: List of ground-truth dicts with 'name' and 'relevance' keys.
        k: Cutoff rank (default 10).
        tokens_plugin: Tokens used by plugin for this query.
        tokens_explorer: Estimated tokens for naive exploration.

    Returns:
        Dict with keys: precision_at_k, recall_at_k, f1_at_k, mrr,
        ndcg_at_k, coverage, token_efficiency_ratio.
    """
    # Binary relevant set: relevance >= 2
    relevant = {e["name"] for e in expected if e.get("relevance", 0) >= 2}

    # Graded relevance map for NDCG
    relevance_map = {e["name"]: e.get("relevance", 0) for e in expected}

    p = precision_at_k(returned, relevant, k)
    r = recall_at_k(returned, relevant, k)
    f = f1_at_k(returned, relevant, k)
    m = mrr(returned, relevant)
    n = ndcg_at_k(returned, relevance_map, k)
    ter = token_efficiency_ratio(f, tokens_plugin, tokens_explorer)

    return {
        f"precision@{k}": round(p, 4),
        f"recall@{k}": round(r, 4),
        f"f1@{k}": round(f, 4),
        "mrr": round(m, 4),
        f"ndcg@{k}": round(n, 4),
        "token_efficiency_ratio": round(ter, 4),
    }
