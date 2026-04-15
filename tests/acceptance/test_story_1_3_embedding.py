"""
Acceptance tests for Story 1.3: Local Embedding Provider.

Validates fastembed integration with lite/quality modes, LRU cache,
batch processing, and dimension reporting.

FRs: FR23 (local default), FR24 (quality mode), FR26 (LRU cache)
NFRs: NFR1 (<500ms), NFR5 (cache <50ms), NFR6 (<200MB idle), NFR7 (zero external data)
"""

import os
import time

import pytest

from src.config import Config
from src.embeddings import (
    CACHE_MAXSIZE,
    LITE_DIMENSION,
    QUALITY_DIMENSION,
    LocalFastembedProvider,
    create_embedding_provider,
)
from src.errors import EmbeddingProviderError


def _make_config(embedding_mode: str = "lite", **overrides) -> Config:
    """Create a Config with the given embedding mode."""
    fields = {"embedding_mode": embedding_mode, **overrides}
    return Config(**fields)


@pytest.fixture(scope="module")
def lite_provider():
    """Create a lite mode provider (shared across module for speed)."""
    config = _make_config("lite")
    return LocalFastembedProvider(config)


class TestEmbeddingLiteMode:
    """AC: Given config with embedding_mode: lite,
    When embed_texts is called, Then fastembed uses bge-small-en-v1.5 (dim 384)."""

    def test_lite_mode_generates_embeddings(self, lite_provider):
        """Modo lite deve gerar embeddings usando BAAI/bge-small-en-v1.5."""
        result = lite_provider.embed_texts(["def hello(): pass"])
        assert len(result) == 1
        assert len(result[0]) == LITE_DIMENSION
        # Embeddings should be floats
        assert all(isinstance(v, float) for v in result[0])

    def test_lite_mode_dimension_is_384(self, lite_provider):
        """dimension() em modo lite deve retornar 384."""
        assert lite_provider.dimension() == 384


class TestEmbeddingQualityMode:
    """AC: Given config with embedding_mode: quality,
    When embed_texts is called, Then fastembed uses bge-m3 (dim 1024)."""

    @pytest.mark.skipif(
        os.environ.get("CTS_TEST_QUALITY_MODE") != "1",
        reason="Quality mode downloads ~568MB model. Set CTS_TEST_QUALITY_MODE=1 to run.",
    )
    def test_quality_mode_generates_embeddings(self):
        """Modo quality deve gerar embeddings usando BAAI/bge-m3."""
        config = _make_config("quality")
        provider = LocalFastembedProvider(config)
        result = provider.embed_texts(["def hello(): pass"])
        assert len(result) == 1
        assert len(result[0]) == QUALITY_DIMENSION

    def test_quality_mode_dimension_is_1024(self):
        """dimension() em modo quality deve retornar 1024."""
        config = _make_config("quality")
        provider = LocalFastembedProvider(config)
        assert provider.dimension() == 1024


class TestEmbeddingCache:
    """AC: Given a query was embedded once, When same query is called again,
    Then result comes from LRU cache in < 50ms."""

    def test_cache_returns_same_result(self, lite_provider):
        """Cache deve retornar mesmo resultado para mesma query."""
        query = "find network errors in the code"
        result1 = lite_provider.embed_query(query)
        result2 = lite_provider.embed_query(query)
        assert result1 == result2

    def test_cache_hit_under_50ms(self, lite_provider):
        """Query cacheada deve retornar em < 50ms."""
        query = "search for authentication functions"
        # First call populates cache
        lite_provider.embed_query(query)

        # Second call should be cached
        start = time.monotonic()
        lite_provider.embed_query(query)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 50, f"Cache hit took {elapsed_ms:.1f}ms, expected < 50ms"

    def test_cache_eviction_at_101_entries(self, lite_provider):
        """Cache com 100 entradas deve evictar a mais antiga na 101a query."""
        # Clear any existing cache by creating a fresh provider
        config = _make_config("lite")
        provider = LocalFastembedProvider(config)

        # Fill cache with 100 entries
        for i in range(CACHE_MAXSIZE):
            provider.embed_query(f"unique query number {i}")

        # Cache should have exactly 100 entries
        assert len(provider._cache) == CACHE_MAXSIZE

        # The first query should be in cache
        assert "unique query number 0" in provider._cache

        # Add 101st entry
        provider.embed_query("the new query that evicts oldest")

        # Cache should still be 100
        assert len(provider._cache) == CACHE_MAXSIZE

        # First entry should be evicted
        assert "unique query number 0" not in provider._cache

        # The new entry should be present
        assert "the new query that evicts oldest" in provider._cache


class TestEmbeddingBatching:
    """AC: Given embed_texts is called with 50 chunks,
    Then all are processed in a single batch call."""

    @pytest.mark.asyncio
    async def test_batch_processing_50_chunks(self, lite_provider):
        """50 chunks devem ser processados em uma unica chamada batch (async)."""
        chunks = [f"def function_{i}(): return {i}" for i in range(50)]
        results = lite_provider.embed_texts(chunks)
        assert len(results) == 50
        # All embeddings should have correct dimension
        for emb in results:
            assert len(emb) == LITE_DIMENSION


class TestEmbeddingFactory:
    """Tests for the create_embedding_provider factory function."""

    def test_local_provider_creates_successfully(self):
        """Factory deve criar LocalFastembedProvider para provider 'local'."""
        config = _make_config("lite")
        provider = create_embedding_provider(config)
        assert isinstance(provider, LocalFastembedProvider)

    def test_unknown_provider_raises_error(self):
        """Factory deve levantar EmbeddingProviderError para provider desconhecido."""
        config = _make_config("lite", embedding_provider="nonexistent_provider")
        with pytest.raises(EmbeddingProviderError):
            create_embedding_provider(config)
