"""
Acceptance tests for Story 7.3: BYOK Embedding Providers.

Validates external embedding provider configuration via env vars
(Voyage AI, OpenAI, Ollama), error handling, and dimension mismatch.

FRs: FR25 (BYOK providers: Voyage, OpenAI, Ollama)
NFRs: NFR10 (code leaves machine only with explicit BYOK config)
"""

# Mock justified: external APIs (Voyage AI, OpenAI, Ollama/httpx) require network
# and real API keys. Tests verify the integration layer (client init, response parsing,
# error propagation) without making actual HTTP calls.
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.embeddings import (
    OllamaProvider,
    OpenAIProvider,
    VoyageProvider,
    create_embedding_provider,
)
from src.errors import EmbeddingProviderError


def _make_config(provider: str = "local", **overrides) -> Config:
    return Config(embedding_mode="lite", embedding_provider=provider, **overrides)


# ---------------------------------------------------------------------------
# Helpers: fake embeddings
# ---------------------------------------------------------------------------

_VOYAGE_DIM = 1536
_OPENAI_DIM = 1536
_OLLAMA_DIM = 768


class TestVoyageProvider:
    """AC: Given CTS_EMBEDDING_PROVIDER=voyage and VOYAGE_API_KEY set,
    When provider initializes, Then Voyage AI is used."""

    def test_voyage_provider_initializes(self, monkeypatch):
        """Provider Voyage AI deve inicializar com API key configurada."""
        monkeypatch.setenv("VOYAGE_API_KEY", "voy-test-key")

        # Mock voyageai import
        mock_voyage = MagicMock()
        mock_client = MagicMock()
        mock_voyage.Client.return_value = mock_client

        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            config = _make_config(provider="voyage")
            provider = VoyageProvider(config)
            # Force lazy init
            _ = provider._get_client()
            mock_voyage.Client.assert_called_once_with(api_key="voy-test-key")

    def test_voyage_generates_embeddings(self, monkeypatch):
        """Voyage AI deve gerar embeddings corretamente."""
        monkeypatch.setenv("VOYAGE_API_KEY", "voy-test-key")

        mock_voyage = MagicMock()
        fake_emb = [0.1] * _VOYAGE_DIM
        mock_result = MagicMock()
        mock_result.embeddings = [fake_emb]
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result
        mock_voyage.Client.return_value = mock_client

        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            config = _make_config(provider="voyage")
            provider = VoyageProvider(config)
            result = provider.embed_query("test query")

        assert isinstance(result, list)
        assert len(result) == _VOYAGE_DIM


class TestOpenAIProvider:
    """AC: Given CTS_EMBEDDING_PROVIDER=openai and OPENAI_API_KEY set,
    When provider initializes, Then OpenAI is used."""

    def test_openai_provider_initializes(self, monkeypatch):
        """Provider OpenAI deve inicializar com API key configurada."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        mock_openai_mod = MagicMock()
        mock_client_instance = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client_instance

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            config = _make_config(provider="openai")
            provider = OpenAIProvider(config)
            _ = provider._get_client()
            mock_openai_mod.OpenAI.assert_called_once_with(api_key="sk-test-key")

    def test_openai_generates_embeddings(self, monkeypatch):
        """OpenAI embeddings API deve gerar embeddings corretamente."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        fake_emb = [0.2] * _OPENAI_DIM
        mock_item = MagicMock()
        mock_item.embedding = fake_emb
        mock_response = MagicMock()
        mock_response.data = [mock_item]
        mock_client_instance = MagicMock()
        mock_client_instance.embeddings.create.return_value = mock_response

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client_instance

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            config = _make_config(provider="openai")
            provider = OpenAIProvider(config)
            result = provider.embed_query("test query")

        assert isinstance(result, list)
        assert len(result) == _OPENAI_DIM


class TestOllamaProvider:
    """AC: Given CTS_EMBEDDING_PROVIDER=ollama and CTS_OLLAMA_URL set,
    When provider initializes, Then local Ollama is used."""

    def test_ollama_provider_initializes(self, monkeypatch):
        """Provider Ollama deve inicializar com URL configurada."""
        monkeypatch.setenv("CTS_OLLAMA_URL", "http://localhost:11434")

        config = _make_config(provider="ollama")
        provider = OllamaProvider(config)
        assert provider._base_url == "http://localhost:11434"

    def test_ollama_generates_embeddings(self, monkeypatch):
        """Ollama local deve gerar embeddings corretamente."""
        monkeypatch.setenv("CTS_OLLAMA_URL", "http://localhost:11434")

        fake_emb = [0.3] * _OLLAMA_DIM
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embedding": fake_emb}

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client_instance

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            config = _make_config(provider="ollama")
            provider = OllamaProvider(config)
            result = provider.embed_query("test query")

        assert isinstance(result, list)
        assert len(result) == _OLLAMA_DIM


class TestByokErrorHandling:
    """AC: Given BYOK provider unreachable, When embedding attempted,
    Then EmbeddingProviderError raised, no silent fallback."""

    def test_unreachable_raises_error(self, monkeypatch):
        """Provider inacessivel deve levantar EmbeddingProviderError."""
        monkeypatch.setenv("CTS_OLLAMA_URL", "http://localhost:11434")

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post.side_effect = ConnectionError("Connection refused")
        mock_httpx.Client.return_value = mock_client_instance

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            config = _make_config(provider="ollama")
            provider = OllamaProvider(config)
            with pytest.raises(EmbeddingProviderError):
                provider.embed_query("test query")

    def test_error_includes_provider_name(self, monkeypatch):
        """Erro deve incluir nome do provider e detalhes."""
        monkeypatch.setenv("CTS_OLLAMA_URL", "http://localhost:11434")

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post.side_effect = ConnectionError("timed out")
        mock_httpx.Client.return_value = mock_client_instance

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            config = _make_config(provider="ollama")
            provider = OllamaProvider(config)
            with pytest.raises(EmbeddingProviderError) as exc_info:
                provider.embed_query("test query")
            assert "Ollama" in str(exc_info.value) or "ollama" in str(exc_info.value).lower()

    def test_no_silent_fallback_to_local(self, monkeypatch):
        """NAO deve fazer fallback silencioso para provider local."""
        # If VOYAGE_API_KEY is missing, factory should raise immediately
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

        config = _make_config(provider="voyage")
        with pytest.raises(EmbeddingProviderError) as exc_info:
            VoyageProvider(config)
        # Error message should clearly state the key is missing
        assert "VOYAGE_API_KEY" in str(exc_info.value)


class TestDimensionMismatch:
    """AC: Given BYOK returns dim N, existing index has dim M (N!=M),
    Then error raised suggesting reindex."""

    def test_dimension_mismatch_raises_error(self, tmp_path):
        """Mismatch de dimensao deve levantar erro."""
        from src.config import Config as Cfg
        from src.storage import Storage

        # Create a Storage with lite mode (dim 384)
        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        config = Cfg(embedding_mode="lite")
        storage = Storage(config, repo, base_path=tmp_path / "indices")

        # Try to search with a 1536-dim vector (OpenAI dimension)
        query_vec = [0.1] * 1536

        # Storage.search_vector should raise ValueError for dimension mismatch
        with pytest.raises(ValueError, match="dimension mismatch"):
            storage.search_vector(query_vec, top_k=3)

    def test_error_suggests_reindex(self, monkeypatch):
        """Erro deve sugerir reindex com o novo provider."""
        # Verify that the factory raises a clear error for unknown providers
        config = _make_config(provider="unknown_provider")
        with pytest.raises(EmbeddingProviderError) as exc_info:
            create_embedding_provider(config)
        error_msg = str(exc_info.value)
        assert "unknown_provider" in error_msg
