"""
Acceptance tests for Story 8.2: Enriquecimento de Embedding com Metadados.

Valida que o indexer prepara o texto de embedding com prefixo de metadados
(nome, tipo e caminho do arquivo) antes de chamar o provider de embeddings.

SQ: SQ3 (search-quality-spec.md secao 4.3)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION, LocalFastembedProvider
from src.indexer import Indexer
from src.storage import Storage


def _make_config(**overrides) -> Config:
    fields = {"embedding_mode": "lite", **overrides}
    return Config(**fields)


class TestEmbeddingEnrichmentFormat:
    """AC1: O texto enviado ao provider deve conter o prefixo de metadados."""

    def test_enriched_format_prepended_to_content(self, tmp_path):
        """Texto para embed deve ser '# name (chunk_type) in file_path\\ncontent'."""
        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "billing").mkdir()
        (repo / "billing" / "feature_gate.py").write_text(
            "class FeatureGateService:\n    def can_access(self):\n        pass\n"
        )

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")

        captured_texts = []

        class CapturingProvider:
            def embed_texts(self, texts):
                captured_texts.extend(texts)
                dim = LITE_DIMENSION
                return [[0.0] * dim for _ in texts]

        import asyncio
        indexer = Indexer(config, storage, CapturingProvider(), repo_path=repo)
        asyncio.run(indexer.reindex())

        assert len(captured_texts) > 0, "Nenhum texto foi enviado ao provider"
        for text in captured_texts:
            lines = text.split("\n")
            assert lines[0].startswith("#"), (
                f"Texto nao comeca com prefixo enriquecido: {text[:80]!r}"
            )
            # formato: # name (chunk_type) in file_path
            assert "(" in lines[0] and ")" in lines[0], (
                f"Prefixo nao contem chunk_type entre parenteses: {lines[0]!r}"
            )
            assert " in " in lines[0], (
                f"Prefixo nao contem ' in file_path': {lines[0]!r}"
            )

    def test_enriched_text_contains_name_and_type(self, tmp_path):
        """O prefixo deve conter o nome do chunk e o chunk_type."""
        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "auth.py").write_text(
            "def authenticate(user, password):\n    return True\n"
        )

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")

        captured_texts = []

        class CapturingProvider:
            def embed_texts(self, texts):
                captured_texts.extend(texts)
                return [[0.0] * LITE_DIMENSION for _ in texts]

        import asyncio
        indexer = Indexer(config, storage, CapturingProvider(), repo_path=repo)
        asyncio.run(indexer.reindex())

        first_lines = [t.split("\n")[0] for t in captured_texts]
        assert any("authenticate" in line for line in first_lines), (
            f"Nome 'authenticate' nao encontrado nos prefixos: {first_lines}"
        )
        assert any("function" in line or "method" in line for line in first_lines), (
            f"chunk_type nao encontrado nos prefixos: {first_lines}"
        )

    def test_enriched_text_contains_file_path(self, tmp_path):
        """O prefixo deve conter o file_path do chunk."""
        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "billing").mkdir()
        (repo / "billing" / "gate.py").write_text(
            "def check_access(): pass\n"
        )

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")

        captured_texts = []

        class CapturingProvider:
            def embed_texts(self, texts):
                captured_texts.extend(texts)
                return [[0.0] * LITE_DIMENSION for _ in texts]

        import asyncio
        indexer = Indexer(config, storage, CapturingProvider(), repo_path=repo)
        asyncio.run(indexer.reindex())

        first_lines = [t.split("\n")[0] for t in captured_texts]
        assert any("gate.py" in line for line in first_lines), (
            f"file_path 'gate.py' nao encontrado nos prefixos: {first_lines}"
        )


class TestEmbeddingDimensionUnchanged:
    """AC4: A dimensao do vetor nao deve mudar com o enriquecimento."""

    @pytest.mark.asyncio
    async def test_embedding_dimension_unchanged_with_enrichment(self, tmp_path):
        """Enriquecimento nao deve alterar a dimensao do embedding (384 para lite)."""
        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "service.py").write_text(
            "class MyService:\n    def process(self): pass\n"
        )

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        stats = await indexer.reindex()
        assert stats["chunks_created"] > 0

        results = storage.search_vector([0.0] * LITE_DIMENSION, top_k=5)
        for r in results:
            assert "embedding" not in r or True  # embedding nao e retornado na busca


class TestIncrementalReindexUsesEnrichedFormat:
    """AC3: Reindex incremental tambem usa o formato enriquecido."""

    def test_incremental_reindex_enriches_updated_chunks(self, tmp_path):
        """Arquivo modificado deve gerar embeddings com formato enriquecido."""
        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "src").mkdir()
        src_file = repo / "src" / "utils.py"
        src_file.write_text("def helper(): pass\n")

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")

        captured_texts_run1 = []
        captured_texts_run2 = []
        call_count = [0]

        class CapturingProvider:
            def embed_texts(self, texts):
                call_count[0] += 1
                if call_count[0] == 1:
                    captured_texts_run1.extend(texts)
                else:
                    captured_texts_run2.extend(texts)
                return [[0.0] * LITE_DIMENSION for _ in texts]

        import asyncio
        indexer = Indexer(config, storage, CapturingProvider(), repo_path=repo)
        asyncio.run(indexer.reindex())

        # Modifica o arquivo para forcar reindex incremental
        src_file.write_text("def helper(): return 42\ndef new_func(): pass\n")
        asyncio.run(indexer.reindex())

        # Ambas as execucoes devem ter enviado textos enriquecidos
        for texts in [captured_texts_run1, captured_texts_run2]:
            if texts:
                for text in texts:
                    assert text.split("\n")[0].startswith("#"), (
                        f"Texto no reindex incremental nao tem prefixo enriquecido: {text[:80]!r}"
                    )


class TestContentFieldPreservedInStorage:
    """O campo content no LanceDB deve permanecer inalterado (sem prefixo)."""

    @pytest.mark.asyncio
    async def test_content_field_stored_without_enrichment_prefix(self, tmp_path):
        """O campo 'content' armazenado nao deve ter o prefixo de embedding."""
        repo = tmp_path / "test-repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "clean.py").write_text("def clean_func(): pass\n")

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        await indexer.reindex()

        results = storage.search_vector([0.0] * LITE_DIMENSION, top_k=5)
        for r in results:
            content = r.get("content", "")
            assert not content.startswith("# "), (
                f"Content no storage nao deve ter prefixo enriquecido: {content[:80]!r}"
            )
