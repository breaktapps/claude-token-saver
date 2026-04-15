"""
Acceptance tests for Story 10.6: Cross-Layer Boost.

Validates that _expand_cross_layer finds matching symbols in other languages
using normalized name matching, with suffix stripping and deduplication.
"""

import random

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.server import _expand_cross_layer, _normalize_name
from src.storage import Storage


DIM = LITE_DIMENSION


def _random_vector(seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(DIM)]


def _make_chunk(
    file_path: str = "src/main.py",
    chunk_type: str = "class",
    name: str = "MyClass",
    line_start: int = 1,
    line_end: int = 20,
    content: str = "class MyClass: pass",
    file_hash: str = "abc123",
    language: str = "python",
    parent_name: str = "",
    calls_json: str = "[]",
    outline_json: str = "[]",
    seed: int | None = None,
) -> dict:
    return {
        "file_path": file_path,
        "chunk_type": chunk_type,
        "name": name,
        "line_start": line_start,
        "line_end": line_end,
        "content": content,
        "embedding": _random_vector(seed=seed),
        "file_hash": file_hash,
        "language": language,
        "parent_name": parent_name,
        "calls_json": calls_json,
        "outline_json": outline_json,
        "imports": [],
    }


@pytest.fixture
def storage(tmp_path):
    config = Config.load(config_path=tmp_path / "nonexistent.yaml")
    repo_path = tmp_path / "fake-repo"
    repo_path.mkdir()
    return Storage(config, repo_path, base_path=tmp_path / "indices")


class TestNormalizeName:
    """Unit tests for _normalize_name helper."""

    def test_service_suffix_stripped(self):
        assert _normalize_name("FeatureGateService") == "feature_gate"

    def test_controller_suffix_stripped(self):
        assert _normalize_name("BillingController") == "billing"

    def test_provider_suffix_stripped(self):
        assert _normalize_name("SubscriptionProvider") == "subscription"

    def test_widget_suffix_stripped(self):
        assert _normalize_name("PaymentWidget") == "payment"

    def test_state_suffix_stripped(self):
        assert _normalize_name("AuthState") == "auth"

    def test_screen_suffix_stripped(self):
        assert _normalize_name("HomeScreen") == "home"

    def test_handler_suffix_stripped(self):
        assert _normalize_name("EventHandler") == "event"

    def test_guard_suffix_stripped(self):
        assert _normalize_name("BillingGuard") == "billing"

    def test_compound_camel_case(self):
        assert _normalize_name("BillingGuardService") == "billing_guard"

    def test_already_snake_case(self):
        assert _normalize_name("feature_gate") == "feature_gate"

    def test_no_suffix(self):
        assert _normalize_name("UserProfile") == "user_profile"

    def test_hyphen_replaced(self):
        assert _normalize_name("feature-gate") == "feature_gate"


class TestExpandCrossLayerBasic:
    """AC1: Cross-layer chunks aparecem com cross_layer=True e score correto."""

    def test_ts_service_finds_dart_equivalent(self, storage):
        """FeatureGateService (TS) deve encontrar feature_gate (Dart)."""
        dart_chunk = _make_chunk(
            file_path="lib/feature_gate.dart",
            name="FeatureGate",
            language="dart",
            seed=1,
        )
        storage.upsert([dart_chunk])

        ts_result = {
            "file_path": "src/feature_gate.service.ts",
            "name": "FeatureGateService",
            "chunk_type": "class",
            "line_start": 1,
            "line_end": 30,
            "content": "class FeatureGateService {}",
            "score": 0.9,
            "language": "typescript",
            "stale": False,
            "parent_name": "",
        }

        cross = _expand_cross_layer([ts_result], storage)

        found_names = [c["name"] for c in cross]
        assert "FeatureGate" in found_names

        match = next(c for c in cross if c["name"] == "FeatureGate")
        assert match["cross_layer"] is True
        assert abs(match["score"] - 0.9 * 0.7) < 1e-5
        assert match["language"] == "dart"

    def test_dart_provider_finds_ts_service(self, storage):
        """SubscriptionProvider (Dart) deve encontrar SubscriptionService (TS)."""
        ts_chunk = _make_chunk(
            file_path="src/subscription.service.ts",
            name="SubscriptionService",
            language="typescript",
            seed=2,
        )
        storage.upsert([ts_chunk])

        dart_result = {
            "file_path": "lib/subscription_provider.dart",
            "name": "SubscriptionProvider",
            "chunk_type": "class",
            "line_start": 1,
            "line_end": 40,
            "content": "class SubscriptionProvider {}",
            "score": 0.85,
            "language": "dart",
            "stale": False,
            "parent_name": "",
        }

        cross = _expand_cross_layer([dart_result], storage)

        found_names = [c["name"] for c in cross]
        assert "SubscriptionService" in found_names

    def test_billing_guard_service_finds_billing_guard(self, storage):
        """BillingGuardService (TS) deve encontrar BillingGuard (Dart)."""
        dart_chunk = _make_chunk(
            file_path="lib/billing_guard.dart",
            name="BillingGuard",
            language="dart",
            seed=3,
        )
        storage.upsert([dart_chunk])

        ts_result = {
            "file_path": "src/billing_guard.service.ts",
            "name": "BillingGuardService",
            "chunk_type": "class",
            "line_start": 1,
            "line_end": 25,
            "content": "class BillingGuardService {}",
            "score": 0.88,
            "language": "typescript",
            "stale": False,
            "parent_name": "",
        }

        cross = _expand_cross_layer([ts_result], storage)

        found_names = [c["name"] for c in cross]
        assert "BillingGuard" in found_names


class TestExpandCrossLayerDeduplication:
    """AC2: Chunks já presentes nos resultados não devem ser duplicados."""

    def test_already_in_results_skipped(self, storage):
        """Chunk já presente (mesmo file_path + name) não deve ser adicionado novamente."""
        chunk = _make_chunk(
            file_path="lib/feature_gate.dart",
            name="FeatureGate",
            language="dart",
            seed=10,
        )
        storage.upsert([chunk])

        ts_result = {
            "file_path": "src/feature_gate.service.ts",
            "name": "FeatureGateService",
            "chunk_type": "class",
            "line_start": 1,
            "line_end": 20,
            "content": "class FeatureGateService {}",
            "score": 0.9,
            "language": "typescript",
            "stale": False,
            "parent_name": "",
        }
        dart_result = {
            "file_path": "lib/feature_gate.dart",
            "name": "FeatureGate",
            "chunk_type": "class",
            "line_start": 1,
            "line_end": 20,
            "content": "class FeatureGate {}",
            "score": 0.8,
            "language": "dart",
            "stale": False,
            "parent_name": "",
        }

        cross = _expand_cross_layer([ts_result, dart_result], storage)

        # FeatureGate already in results — should not appear again
        added_names = [c["name"] for c in cross]
        assert added_names.count("FeatureGate") == 0

    def test_same_language_skipped(self, storage):
        """Chunks na mesma linguagem não devem ser adicionados como cross-layer."""
        py_chunk = _make_chunk(
            file_path="src/feature_gate_service.py",
            name="FeatureGateService",
            language="python",
            seed=11,
        )
        storage.upsert([py_chunk])

        py_result = {
            "file_path": "src/billing.py",
            "name": "BillingService",
            "chunk_type": "class",
            "line_start": 1,
            "line_end": 20,
            "content": "class BillingService: pass",
            "score": 0.8,
            "language": "python",
            "stale": False,
            "parent_name": "",
        }

        # FeatureGateService in index has same language (python) as result — skip
        cross = _expand_cross_layer([py_result], storage)

        cross_langs = {c["language"] for c in cross}
        # No python cross-layer results should appear
        assert "python" not in cross_langs or all(
            c["language"] != "python" for c in cross
        )


class TestExpandCrossLayerEmpty:
    """Edge cases: resultados vazios e nome não encontrado."""

    def test_empty_results_returns_empty(self, storage):
        cross = _expand_cross_layer([], storage)
        assert cross == []

    def test_no_match_in_index_returns_empty(self, storage):
        result = {
            "file_path": "src/totally_unique_service.ts",
            "name": "TotallyUniqueService",
            "chunk_type": "class",
            "line_start": 1,
            "line_end": 10,
            "content": "class TotallyUniqueService {}",
            "score": 0.75,
            "language": "typescript",
            "stale": False,
            "parent_name": "",
        }

        cross = _expand_cross_layer([result], storage)
        assert cross == []
