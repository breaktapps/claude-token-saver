"""
Acceptance tests for Story 10.5: Module Clustering.

Validates that _expand_module_siblings adds co-located chunks when 2+ results
share the same directory, respects the max-5-files cap, prefers class chunks,
and that search_exact is not affected.
"""

import random
from pathlib import Path

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.server import _expand_module_siblings
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
        "imports": [],
    }


@pytest.fixture
def storage(tmp_path):
    config = Config.load(config_path=tmp_path / "nonexistent.yaml")
    repo_path = tmp_path / "fake-repo"
    repo_path.mkdir()
    return Storage(config, repo_path, base_path=tmp_path / "indices")


class TestExpandModuleSiblingsSignal:
    """AC1: 2+ resultados no mesmo diretório → siblings incluídos."""

    def test_two_results_same_dir_adds_sibling(self, storage):
        """Com 2 resultados no mesmo dir, chunks adicionais desse dir são incluídos."""
        # Chunk sibling já indexado mas não nos resultados
        sibling = _make_chunk(
            file_path="src/models/sibling.py",
            name="SiblingClass",
            chunk_type="class",
            content="class SiblingClass: pass",
            seed=10,
        )
        storage.upsert([sibling])

        results = [
            {
                "file_path": "src/models/alpha.py",
                "name": "AlphaFunc",
                "chunk_type": "function",
                "score": 0.9,
                "line_start": 1,
                "line_end": 5,
                "content": "def alpha(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
            {
                "file_path": "src/models/beta.py",
                "name": "BetaFunc",
                "chunk_type": "function",
                "score": 0.85,
                "line_start": 1,
                "line_end": 5,
                "content": "def beta(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
        ]

        siblings = _expand_module_siblings(results, top_k=5, storage=storage)

        sibling_paths = {s["file_path"] for s in siblings}
        assert "src/models/sibling.py" in sibling_paths

    def test_siblings_have_module_match_flag(self, storage):
        """Chunks adicionados como siblings devem ter module_match=True."""
        sibling = _make_chunk(
            file_path="src/models/sibling.py",
            name="SiblingClass",
            chunk_type="class",
            content="class SiblingClass: pass",
            seed=11,
        )
        storage.upsert([sibling])

        results = [
            {
                "file_path": "src/models/alpha.py",
                "name": "Alpha",
                "chunk_type": "function",
                "score": 0.9,
                "line_start": 1,
                "line_end": 5,
                "content": "def alpha(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
            {
                "file_path": "src/models/beta.py",
                "name": "Beta",
                "chunk_type": "function",
                "score": 0.8,
                "line_start": 1,
                "line_end": 5,
                "content": "def beta(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
        ]

        siblings = _expand_module_siblings(results, top_k=5, storage=storage)

        for s in siblings:
            assert s.get("module_match") is True

    def test_siblings_score_is_max_times_0_7(self, storage):
        """Score dos siblings = max score dos resultados * 0.7."""
        sibling = _make_chunk(
            file_path="src/models/sibling.py",
            name="SiblingClass",
            chunk_type="class",
            content="class SiblingClass: pass",
            seed=12,
        )
        storage.upsert([sibling])

        results = [
            {
                "file_path": "src/models/alpha.py",
                "name": "Alpha",
                "chunk_type": "function",
                "score": 0.8,
                "line_start": 1,
                "line_end": 5,
                "content": "def alpha(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
            {
                "file_path": "src/models/beta.py",
                "name": "Beta",
                "chunk_type": "function",
                "score": 0.6,
                "line_start": 1,
                "line_end": 5,
                "content": "def beta(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
        ]

        siblings = _expand_module_siblings(results, top_k=5, storage=storage)

        # max score = 0.8; sibling score = 0.8 * 0.7 = 0.56
        for s in siblings:
            assert s["score"] == pytest.approx(0.56, abs=1e-5)


class TestExpandModuleSiblingsNoSignal:
    """AC2: 1 resultado num diretório → sem siblings."""

    def test_single_result_in_dir_no_siblings(self, storage):
        """Com apenas 1 resultado num diretório, nenhum sibling deve ser adicionado."""
        sibling = _make_chunk(
            file_path="src/models/sibling.py",
            name="SiblingClass",
            chunk_type="class",
            content="class SiblingClass: pass",
            seed=20,
        )
        storage.upsert([sibling])

        # Apenas 1 resultado em src/models/
        results = [
            {
                "file_path": "src/models/alpha.py",
                "name": "Alpha",
                "chunk_type": "function",
                "score": 0.9,
                "line_start": 1,
                "line_end": 5,
                "content": "def alpha(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
            # segundo resultado em diretório diferente
            {
                "file_path": "src/utils/helper.py",
                "name": "Helper",
                "chunk_type": "function",
                "score": 0.8,
                "line_start": 1,
                "line_end": 5,
                "content": "def helper(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
        ]

        siblings = _expand_module_siblings(results, top_k=5, storage=storage)

        sibling_paths = {s["file_path"] for s in siblings}
        # src/models/sibling.py NÃO deve aparecer (apenas 1 resultado em src/models/)
        assert "src/models/sibling.py" not in sibling_paths

    def test_empty_results_returns_empty(self, storage):
        """Com lista de resultados vazia, retorna vazia."""
        siblings = _expand_module_siblings([], top_k=5, storage=storage)
        assert siblings == []


class TestExpandModuleSiblingsCap:
    """AC3: Máximo de 5 arquivos co-localizados adicionados."""

    def test_max_five_files_cap(self, storage):
        """Nunca mais de 5 arquivos devem ser adicionados como siblings."""
        # Indexa 8 arquivos no mesmo diretório como siblings
        sibling_chunks = [
            _make_chunk(
                file_path=f"src/models/sibling_{i}.py",
                name=f"Sibling{i}",
                chunk_type="function",
                content=f"def sibling_{i}(): pass",
                seed=30 + i,
            )
            for i in range(8)
        ]
        storage.upsert(sibling_chunks)

        # 2 resultados no mesmo diretório para ativar o sinal
        results = [
            {
                "file_path": "src/models/alpha.py",
                "name": "Alpha",
                "chunk_type": "function",
                "score": 0.9,
                "line_start": 1,
                "line_end": 5,
                "content": "def alpha(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
            {
                "file_path": "src/models/beta.py",
                "name": "Beta",
                "chunk_type": "function",
                "score": 0.8,
                "line_start": 1,
                "line_end": 5,
                "content": "def beta(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
        ]

        siblings = _expand_module_siblings(results, top_k=5, storage=storage)

        assert len(siblings) <= 5


class TestExpandModuleSiblingsClassPreference:
    """AC4: Chunks de tipo 'class' são preferidos sobre métodos."""

    def test_class_chunks_preferred_over_method(self, storage):
        """Quando há class e method disponíveis, class deve ser adicionado."""
        class_chunk = _make_chunk(
            file_path="src/models/widget.py",
            name="Widget",
            chunk_type="class",
            content="class Widget: pass",
            seed=40,
        )
        method_chunk = _make_chunk(
            file_path="src/models/widget.py",
            name="Widget.render",
            chunk_type="method",
            content="def render(self): pass",
            seed=41,
        )
        storage.upsert([class_chunk, method_chunk])

        results = [
            {
                "file_path": "src/models/alpha.py",
                "name": "Alpha",
                "chunk_type": "function",
                "score": 0.9,
                "line_start": 1,
                "line_end": 5,
                "content": "def alpha(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
            {
                "file_path": "src/models/beta.py",
                "name": "Beta",
                "chunk_type": "function",
                "score": 0.8,
                "line_start": 1,
                "line_end": 5,
                "content": "def beta(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
        ]

        siblings = _expand_module_siblings(results, top_k=5, storage=storage)

        widget_siblings = [s for s in siblings if s["file_path"] == "src/models/widget.py"]
        # Apenas 1 chunk por arquivo (o representante)
        assert len(widget_siblings) == 1
        assert widget_siblings[0]["chunk_type"] == "class"
        assert widget_siblings[0]["name"] == "Widget"


class TestExpandModuleSiblingsNotDuplicated:
    """AC: Arquivos já presentes nos resultados não são duplicados."""

    def test_existing_files_not_added_as_siblings(self, storage):
        """Chunks de arquivos já nos resultados não devem aparecer como siblings."""
        # Indexa chunk de arquivo que já está nos resultados
        chunk_alpha = _make_chunk(
            file_path="src/models/alpha.py",
            name="AlphaExtra",
            chunk_type="function",
            content="def alpha_extra(): pass",
            seed=50,
        )
        storage.upsert([chunk_alpha])

        results = [
            {
                "file_path": "src/models/alpha.py",
                "name": "Alpha",
                "chunk_type": "function",
                "score": 0.9,
                "line_start": 1,
                "line_end": 5,
                "content": "def alpha(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
            {
                "file_path": "src/models/beta.py",
                "name": "Beta",
                "chunk_type": "function",
                "score": 0.8,
                "line_start": 1,
                "line_end": 5,
                "content": "def beta(): pass",
                "language": "python",
                "stale": False,
                "parent_name": "",
            },
        ]

        siblings = _expand_module_siblings(results, top_k=5, storage=storage)

        sibling_paths = {s["file_path"] for s in siblings}
        assert "src/models/alpha.py" not in sibling_paths


class TestSearchExactNotAffected:
    """AC5: search_exact não deve aplicar module clustering."""

    def test_search_exact_has_no_module_match_results(self, storage):
        """_expand_module_siblings não é chamado em search_exact — importando a função
        e verificando que ela não é acoplada ao fluxo de search_exact."""
        # Este teste valida a assinatura: search_exact não chama _expand_module_siblings.
        # Verificamos indiretamente: importando de server, garantimos que a função existe
        # mas que ela é independente de search_exact.
        from src.server import _expand_module_siblings as expand_fn
        import inspect
        import src.server as server_module

        # Lê o código-fonte de search_exact para confirmar que não chama _expand_module_siblings
        source = inspect.getsource(server_module.search_exact)
        assert "_expand_module_siblings" not in source, (
            "search_exact não deve chamar _expand_module_siblings"
        )

    def test_search_semantic_calls_expand_module_siblings(self):
        """search_semantic deve chamar _expand_module_siblings."""
        import inspect
        import src.server as server_module

        source = inspect.getsource(server_module.search_semantic)
        assert "_expand_module_siblings" in source

    def test_search_hybrid_calls_expand_module_siblings(self):
        """search_hybrid deve chamar _expand_module_siblings."""
        import inspect
        import src.server as server_module

        source = inspect.getsource(server_module.search_hybrid)
        assert "_expand_module_siblings" in source
