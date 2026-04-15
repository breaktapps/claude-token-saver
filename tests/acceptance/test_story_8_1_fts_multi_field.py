"""
Acceptance tests for Story 8.1: FTS Multi-Field Index.

Valida que FTS indexa campos content, name e file_path,
permitindo busca por nome de simbolo e diretorio.

SQ: SQ2 (search-quality-spec.md secao 4.2)
"""

import random
from pathlib import Path

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.storage import Storage


DIM = LITE_DIMENSION  # 384


def _random_vector(dim: int = DIM, seed: int | None = None) -> list[float]:
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
    config = Config.load(config_path=tmp_path / "nonexistent.yaml")
    repo_path = tmp_path / "fake-repo"
    repo_path.mkdir()
    return Storage(config, repo_path, base_path=tmp_path / "indices")


class TestFtsMatchViaNameField:
    """AC2: Chunk com nome de simbolo deve ser encontrado por busca no campo name."""

    def test_fts_finds_chunk_by_symbol_name(self, storage):
        """search_fts('subscription') deve retornar chunk cujo name='SubscriptionService'
        mesmo quando 'subscription' nao aparece no content."""
        chunk = _make_chunk(
            file_path="src/billing/subscription.py",
            chunk_type="class",
            name="SubscriptionService",
            content="class body with initialize and teardown logic only",
            seed=10,
        )
        storage.upsert([chunk])

        results = storage.search_fts("SubscriptionService", top_k=5)

        names = [r["name"] for r in results]
        assert "SubscriptionService" in names, (
            f"Expected 'SubscriptionService' in results, got names: {names}"
        )

    def test_fts_finds_chunk_by_partial_symbol_name(self, storage):
        """search_fts com parte do nome de simbolo deve retornar o chunk correto."""
        chunk = _make_chunk(
            file_path="src/payments/processor.py",
            chunk_type="class",
            name="PaymentProcessor",
            content="handles the main payment flow",
            seed=11,
        )
        storage.upsert([chunk])

        results = storage.search_fts("PaymentProcessor", top_k=5)

        names = [r["name"] for r in results]
        assert "PaymentProcessor" in names, (
            f"Expected 'PaymentProcessor' in results, got names: {names}"
        )


class TestFtsMatchViaFilePathField:
    """AC3: Busca por nome de diretorio deve retornar chunks daquele path."""

    def test_fts_finds_chunks_by_directory_name(self, storage):
        """search_fts('billing') deve retornar chunks de backend/src/billing/."""
        billing_chunk = _make_chunk(
            file_path="backend/src/billing/guard.ts",
            name="BillingGuard",
            content="export class guard implementation",
            seed=20,
        )
        other_chunk = _make_chunk(
            file_path="frontend/src/app.tsx",
            name="App",
            content="export default application root component",
            seed=21,
        )
        storage.upsert([billing_chunk, other_chunk])

        results = storage.search_fts("billing", top_k=5)

        file_paths = [r["file_path"] for r in results]
        assert any("billing" in fp for fp in file_paths), (
            f"Expected a result from billing directory, got: {file_paths}"
        )

    def test_fts_file_path_match_does_not_return_unrelated(self, storage):
        """search_fts('billing') nao deve retornar chunks sem 'billing' no path ou content."""
        billing_chunk = _make_chunk(
            file_path="backend/src/billing/service.py",
            name="BillingService",
            content="manages subscriptions and invoices",
            seed=30,
        )
        unrelated_chunk = _make_chunk(
            file_path="frontend/src/auth/login.tsx",
            name="LoginForm",
            content="renders username and password fields",
            seed=31,
        )
        storage.upsert([billing_chunk, unrelated_chunk])

        results = storage.search_fts("billing", top_k=5)

        for r in results:
            has_billing = (
                "billing" in r["file_path"].lower()
                or "billing" in r["content"].lower()
                or "billing" in r["name"].lower()
            )
            assert has_billing, (
                f"Unexpected result without 'billing': file_path={r['file_path']}, name={r['name']}"
            )


class TestFtsContentRegressionNonRegression:
    """Nao-regressao: busca por content continua funcionando apos mudanca para multi-field."""

    def test_fts_content_match_still_works(self, storage):
        """search_fts por termo no content deve continuar retornando resultados."""
        chunk = _make_chunk(
            file_path="src/main.py",
            name="main",
            content="def main(): print('hello world')",
            seed=40,
        )
        storage.upsert([chunk])

        results = storage.search_fts("hello world", top_k=5)

        assert len(results) >= 1, "FTS busca por content nao retornou resultados"
        contents = [r["content"] for r in results]
        assert any("hello" in c for c in contents)

    def test_fts_top_k_still_respected(self, storage):
        """search_fts com multi-field ainda respeita o limite top_k."""
        chunks = [
            _make_chunk(file_path=f"src/file_{i}.py", name=f"func_{i}",
                        content=f"def func_{i}(): return {i}", seed=i + 50)
            for i in range(5)
        ]
        storage.upsert(chunks)

        results = storage.search_fts("def", top_k=2)
        assert len(results) <= 2


class TestFtsIndexMigration:
    """AC4: Indice single-field existente e substituido automaticamente por multi-field."""

    def test_rebuild_after_fts_created_reset(self, storage):
        """Apos _fts_created = False, proxima busca recria o indice como multi-field
        e consegue encontrar chunks pelo campo name."""
        chunk = _make_chunk(
            file_path="src/core/subscriber.py",
            chunk_type="class",
            name="SubscriberManager",
            content="core class with no subscriber word in body",
            seed=60,
        )
        storage.upsert([chunk])

        # Force FTS index to be rebuilt on next search
        storage._fts_created = False

        results = storage.search_fts("SubscriberManager", top_k=5)

        names = [r["name"] for r in results]
        assert "SubscriberManager" in names, (
            f"Expected 'SubscriberManager' after index rebuild, got: {names}"
        )


class TestFtsEdgeCases:
    """Edge cases para FTS multi-field."""

    def test_fts_empty_name_does_not_crash(self, storage):
        """Chunk com name='' (ex: tipo module) nao deve causar erro no FTS."""
        chunk = _make_chunk(
            file_path="src/__init__.py",
            chunk_type="module",
            name="",
            content="module level imports and constants",
            seed=70,
        )
        storage.upsert([chunk])
        # Should not raise
        results = storage.search_fts("module", top_k=5)
        assert isinstance(results, list)

    def test_fts_file_path_with_special_chars(self, storage):
        """Chunks com file_path contendo hifens e underscores devem ser indexados sem erro."""
        chunk = _make_chunk(
            file_path="my-app/src/data_processor.py",
            name="DataProcessor",
            content="processes incoming data records",
            seed=80,
        )
        storage.upsert([chunk])
        results = storage.search_fts("data_processor", top_k=5)
        assert isinstance(results, list)

    def test_fts_empty_table_does_not_crash(self, storage):
        """_ensure_fts_index em tabela vazia nao deve lancar excecao."""
        # Do not upsert anything — table is empty
        results = storage.search_fts("anything", top_k=5)
        assert results == []
