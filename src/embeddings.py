"""Embedding provider abstraction for claude-token-saver.

Supports local fastembed provider with LRU cache for queries.
BYOK providers (Voyage, OpenAI, Ollama) configured via env vars:
  - CTS_EMBEDDING_PROVIDER=voyage + VOYAGE_API_KEY
  - CTS_EMBEDDING_PROVIDER=openai + OPENAI_API_KEY
  - CTS_EMBEDDING_PROVIDER=ollama + CTS_OLLAMA_URL (default: http://localhost:11434)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from .config import Config
from .errors import EmbeddingProviderError

# Model constants
LITE_MODEL = "BAAI/bge-small-en-v1.5"
QUALITY_MODEL = "BAAI/bge-m3"

LITE_DIMENSION = 384
QUALITY_DIMENSION = 1024

# Mode to model/dimension mapping
_MODE_MAP: dict[str, dict[str, Any]] = {
    "lite": {"model": LITE_MODEL, "dimension": LITE_DIMENSION},
    "quality": {"model": QUALITY_MODEL, "dimension": QUALITY_DIMENSION},
}

# Cache size
CACHE_MAXSIZE = 100


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (for indexation). No caching."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query (with LRU cache)."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class LocalFastembedProvider(EmbeddingProvider):
    """Local embedding provider using fastembed (ONNX Runtime)."""

    def __init__(self, config: Config) -> None:
        mode_info = _MODE_MAP.get(config.embedding_mode)
        if mode_info is None:
            raise EmbeddingProviderError(
                f"Unknown embedding mode: '{config.embedding_mode}'. "
                "Use 'lite' or 'quality'."
            )

        self._model_name = mode_info["model"]
        self._dimension = mode_info["dimension"]
        self._batch_size = config.batch_size
        self._model = None  # Lazy init on first embed call
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _get_model(self) -> Any:
        """Lazy-load the fastembed model on first use."""
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Processes in batches of config.batch_size."""
        if not texts:
            return []

        model = self._get_model()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = list(model.embed(batch))
            all_embeddings.extend([emb.tolist() for emb in embeddings])

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with LRU cache."""
        if query in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(query)
            return self._cache[query]

        # Compute embedding
        result = self.embed_texts([query])[0]

        # Add to cache, evict oldest if full
        if len(self._cache) >= CACHE_MAXSIZE:
            self._cache.popitem(last=False)  # Remove oldest
        self._cache[query] = result

        return result

    def dimension(self) -> int:
        """Return embedding dimension for current mode."""
        return self._dimension


class VoyageProvider(EmbeddingProvider):
    """BYOK provider using Voyage AI embeddings API.

    Requires: pip install voyageai
    Config: CTS_EMBEDDING_PROVIDER=voyage + VOYAGE_API_KEY env var
    """

    # voyage-code-2 is tuned for code; dimension 1536
    _DEFAULT_MODEL = "voyage-code-2"
    _DEFAULT_DIMENSION = 1536

    def __init__(self, config: Config) -> None:
        api_key = os.environ.get("VOYAGE_API_KEY", "")
        if not api_key:
            raise EmbeddingProviderError(
                "VOYAGE_API_KEY environment variable not set. "
                "Export VOYAGE_API_KEY=<your-key> to use the Voyage provider."
            )
        self._api_key = api_key
        self._model = self._DEFAULT_MODEL
        self._dim = self._DEFAULT_DIMENSION
        self._client = None  # lazy init

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import voyageai  # type: ignore[import]
            except ImportError:
                raise EmbeddingProviderError(
                    "Package 'voyageai' not installed. Run: pip install voyageai"
                )
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            client = self._get_client()
            result = client.embed(texts, model=self._model, input_type="document")
            return result.embeddings
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Voyage AI embedding failed: {exc}. "
                "Check VOYAGE_API_KEY and network connectivity."
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        try:
            client = self._get_client()
            result = client.embed([query], model=self._model, input_type="query")
            return result.embeddings[0]
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Voyage AI query embedding failed: {exc}. "
                "Check VOYAGE_API_KEY and network connectivity."
            ) from exc

    def dimension(self) -> int:
        return self._dim


class OpenAIProvider(EmbeddingProvider):
    """BYOK provider using OpenAI embeddings API.

    Requires: pip install openai
    Config: CTS_EMBEDDING_PROVIDER=openai + OPENAI_API_KEY env var
    """

    _DEFAULT_MODEL = "text-embedding-3-small"
    _DEFAULT_DIMENSION = 1536

    def __init__(self, config: Config) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EmbeddingProviderError(
                "OPENAI_API_KEY environment variable not set. "
                "Export OPENAI_API_KEY=<your-key> to use the OpenAI provider."
            )
        self._api_key = api_key
        self._model = self._DEFAULT_MODEL
        self._dim = self._DEFAULT_DIMENSION
        self._client = None  # lazy init

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError:
                raise EmbeddingProviderError(
                    "Package 'openai' not installed. Run: pip install openai"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            client = self._get_client()
            response = client.embeddings.create(input=texts, model=self._model)
            return [item.embedding for item in response.data]
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                f"OpenAI embedding failed: {exc}. "
                "Check OPENAI_API_KEY and network connectivity."
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        result = self.embed_texts([query])
        return result[0]

    def dimension(self) -> int:
        return self._dim


class OllamaProvider(EmbeddingProvider):
    """BYOK provider using a local Ollama instance.

    Requires: running Ollama server
    Config: CTS_EMBEDDING_PROVIDER=ollama + CTS_OLLAMA_URL (default: http://localhost:11434)
            CTS_OLLAMA_MODEL (default: nomic-embed-text)
    """

    _DEFAULT_URL = "http://localhost:11434"
    _DEFAULT_MODEL = "nomic-embed-text"
    _DEFAULT_DIMENSION = 768  # nomic-embed-text default

    def __init__(self, config: Config) -> None:
        self._base_url = os.environ.get("CTS_OLLAMA_URL", self._DEFAULT_URL).rstrip("/")
        self._model = os.environ.get("CTS_OLLAMA_MODEL", self._DEFAULT_MODEL)
        self._dim = self._DEFAULT_DIMENSION

    def _embed_single(self, text: str) -> list[float]:
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            raise EmbeddingProviderError(
                "Package 'httpx' not installed. Run: pip install httpx"
            )
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["embedding"]
                # Update dimension from actual response
                self._dim = len(embedding)
                return embedding
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Ollama embedding failed at {self._base_url}: {exc}. "
                "Ensure Ollama is running and CTS_OLLAMA_URL is correct."
            ) from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_single(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_single(query)

    def dimension(self) -> int:
        return self._dim


def create_embedding_provider(config: Config) -> EmbeddingProvider:
    """Factory function to create the appropriate embedding provider.

    Supported providers:
    - 'local': fastembed (default, no API key needed)
    - 'voyage': Voyage AI (requires VOYAGE_API_KEY env var)
    - 'openai': OpenAI embeddings (requires OPENAI_API_KEY env var)
    - 'ollama': Local Ollama (requires running Ollama, CTS_OLLAMA_URL optional)
    """
    provider = config.embedding_provider

    if provider == "local":
        return LocalFastembedProvider(config)
    elif provider == "voyage":
        return VoyageProvider(config)
    elif provider == "openai":
        return OpenAIProvider(config)
    elif provider == "ollama":
        return OllamaProvider(config)
    else:
        raise EmbeddingProviderError(
            f"Unknown provider '{provider}'. "
            "Supported: 'local', 'voyage', 'openai', 'ollama'."
        )
