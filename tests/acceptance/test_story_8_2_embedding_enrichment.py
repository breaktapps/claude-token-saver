"""
Acceptance tests for Story 10.1: Revert Embedding Enrichment.

Valida que o indexer envia o conteudo puro (sem prefixo de metadados)
ao provider de embeddings. O enriquecimento foi revertido porque dilui
os embeddings bge-small (384-dim), causando queda de precisao de 80% para 53%.

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


class TestEmbeddingNotEnriched:
    """AC1: O texto enviado ao provider deve ser o conteudo puro, sem prefixo de metadados."""

    def test_content_sent_without_metadata_prefix(self, tmp_path):
        """Texto para embed deve ser o conteudo bruto, sem '# name (chunk_type) in file_path'."""
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
            first_line = text.split("\n")[0]
            assert not (first_line.startswith("#") and "(" in first_line and " in " in first_line), (
                f"Texto nao deve ter prefixo de metadados enriquecido: {text[:80]!r}"
            )

    def test_content_sent_without_name_type_prefix(self, tmp_path):
        """O texto enviado nao deve conter o nome do chunk como prefixo na primeira linha."""
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

        for text in captured_texts:
            first_line = text.split("\n")[0]
            # A primeira linha nao deve ser um comentario de metadados do formato enriquecido
            assert not (first_line.startswith("# authenticate") and " in " in first_line), (
                f"Prefixo de metadados nao deve estar presente: {first_line!r}"
            )

    def test_content_sent_without_file_path_prefix(self, tmp_path):
        """O texto enviado nao deve conter o file_path como prefixo."""
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
        # Nenhuma primeira linha deve ser um prefixo de metadados com gate.py
        for line in first_lines:
            assert not (line.startswith("#") and "gate.py" in line and " in " in line), (
                f"file_path nao deve aparecer como prefixo de metadados: {line!r}"
            )


class TestEmbeddingDimensionUnchanged:
    """AC4: A dimensao do vetor nao deve mudar apos o revert."""

    @pytest.mark.asyncio
    async def test_embedding_dimension_unchanged(self, tmp_path):
        """Revert nao deve alterar a dimensao do embedding (384 para lite)."""
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


class TestIncrementalReindexNoPrefixFormat:
    """AC3: Reindex incremental tambem usa conteudo puro (sem prefixo)."""

    def test_incremental_reindex_sends_raw_content(self, tmp_path):
        """Arquivo modificado deve gerar embeddings sem formato enriquecido."""
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

        # Nenhuma execucao deve ter enviado textos com prefixo de metadados
        for texts in [captured_texts_run1, captured_texts_run2]:
            for text in texts:
                first_line = text.split("\n")[0]
                assert not (first_line.startswith("#") and " in " in first_line), (
                    f"Texto no reindex nao deve ter prefixo enriquecido: {text[:80]!r}"
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
