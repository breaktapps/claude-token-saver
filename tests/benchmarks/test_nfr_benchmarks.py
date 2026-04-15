"""
Benchmark tests for NFR gaps identified in trace matrix.

NFR1  — search_semantic < 500ms end-to-end
NFR3  — indexação 500 arquivos < 60s modo lite
NFR6  — memória idle < 200MB

Run with:
    uv run pytest tests/benchmarks/ -v -m benchmark
    uv run pytest tests/benchmarks/ -v -m "benchmark and slow"  # inclui NFR3
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import pytest

from src.config import Config
from src.embeddings import LocalFastembedProvider
from src.indexer import Indexer
from src.server import search_semantic
from src.storage import Storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    return Config(embedding_mode="lite", **overrides)


def _create_small_repo(tmp_path: Path, n_files: int = 15) -> Path:
    """Gera um repositório Python sintético com n_files arquivos."""
    repo = tmp_path / "bench-repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    src = repo / "src"
    src.mkdir()

    for i in range(n_files):
        content = _generate_python_file(i)
        (src / f"module_{i:03d}.py").write_text(content)

    return repo


def _generate_python_file(index: int) -> str:
    """Gera um arquivo Python válido com funções e classes."""
    return f'''\
"""Module {index}: synthetic file for benchmark tests."""

from typing import List, Optional


class Processor{index}:
    """Processes items of type {index}."""

    def __init__(self, name: str, limit: int = 100) -> None:
        self.name = name
        self.limit = limit
        self._cache: dict = {{}}

    def process(self, items: List[str]) -> List[str]:
        """Apply transformation to each item."""
        results = []
        for item in items[: self.limit]:
            transformed = self._transform(item)
            results.append(transformed)
        return results

    def _transform(self, item: str) -> str:
        if item in self._cache:
            return self._cache[item]
        result = item.strip().lower().replace(" ", "_")
        self._cache[item] = result
        return result

    def reset(self) -> None:
        """Clear internal cache."""
        self._cache.clear()


def load_data_{index}(path: str, encoding: str = "utf-8") -> List[str]:
    """Load lines from a file at *path*."""
    with open(path, encoding=encoding) as fh:
        return [line.rstrip("\\n") for line in fh]


def save_results_{index}(data: List[str], output_path: str) -> int:
    """Write *data* to *output_path*, return number of lines written."""
    with open(output_path, "w", encoding="utf-8") as fh:
        for line in data:
            fh.write(line + "\\n")
    return len(data)


def validate_input_{index}(value: Optional[str]) -> bool:
    """Return True if *value* is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0
'''


def _create_large_repo(tmp_path: Path, n_files: int = 500) -> Path:
    """Gera um repositório Python sintético com 500 arquivos."""
    repo = tmp_path / "bench-large-repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    # Distribui em sub-diretórios para parecer um projeto real
    subdirs = ["src/core", "src/utils", "src/services", "src/models", "src/handlers"]
    for sd in subdirs:
        (repo / sd).mkdir(parents=True, exist_ok=True)

    for i in range(n_files):
        subdir = subdirs[i % len(subdirs)]
        content = _generate_python_file(i)
        (repo / subdir / f"module_{i:04d}.py").write_text(content)

    return repo


# ---------------------------------------------------------------------------
# NFR1 — search_semantic < 500ms end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestNFR1SearchLatency:
    """NFR1: search_semantic deve retornar em < 500ms (P95 de 3 execuções)."""

    @pytest.fixture(scope="class")
    def indexed_env(self, tmp_path_factory):
        """Cria e indexa um repo pequeno uma única vez para a classe."""
        tmp_path = tmp_path_factory.mktemp("nfr1")
        repo = _create_small_repo(tmp_path, n_files=15)
        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)
        return config, storage, indexer, provider, repo

    @pytest.mark.asyncio
    async def test_search_semantic_under_500ms(self, indexed_env):
        """P95 de 3 execuções de search_semantic deve ser < 500ms."""
        import src.server as srv

        config, storage, indexer, provider, repo = indexed_env

        # Garante que o índice existe
        await indexer.reindex()

        old_state = (
            srv._config, srv._storage, srv._indexer,
            srv._embed_provider, srv._repo_path,
        )
        try:
            srv._config = config
            srv._storage = storage
            srv._indexer = indexer
            srv._embed_provider = provider
            srv._repo_path = repo

            latencies: list[float] = []
            for _ in range(3):
                t0 = time.perf_counter()
                result_json = await search_semantic("process data from file")
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)

                response = json.loads(result_json)
                assert "results" in response, "Resposta inválida do search_semantic"

            latencies.sort()
            p95 = latencies[-1]  # com apenas 3 amostras, P95 ≈ máximo

            assert p95 < 0.5, (
                f"NFR1 FALHOU: P95 de latência = {p95 * 1000:.1f}ms "
                f"(limite: 500ms). Latências: {[f'{l*1000:.1f}ms' for l in latencies]}"
            )
        finally:
            srv._config, srv._storage, srv._indexer, srv._embed_provider, srv._repo_path = old_state


# ---------------------------------------------------------------------------
# NFR3 — indexação 500 arquivos < 60s modo lite
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.slow
class TestNFR3IndexThroughput:
    """NFR3: indexação completa de 500 arquivos Python deve terminar em < 60s (modo lite)."""

    @pytest.mark.asyncio
    async def test_index_500_files_under_60s(self, tmp_path):
        """Reindex de 500 arquivos sintéticos deve completar em < 60s."""
        repo = _create_large_repo(tmp_path, n_files=500)
        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        indexer = Indexer(config, storage, provider, repo_path=repo)

        t0 = time.perf_counter()
        stats = await indexer.reindex(force=True)
        elapsed = time.perf_counter() - t0

        assert stats["files_indexed"] > 0, "Nenhum arquivo foi indexado"
        assert stats["chunks_created"] > 0, "Nenhum chunk foi criado"

        assert elapsed < 60, (
            f"NFR3 FALHOU: indexação de {stats['files_indexed']} arquivos "
            f"({stats['chunks_created']} chunks) levou {elapsed:.1f}s "
            f"(limite: 60s)"
        )


# ---------------------------------------------------------------------------
# NFR6 — memória idle < 200MB
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestNFR6MemoryIdle:
    """NFR6: uso de memória em idle (após importar módulos) deve ser < 200MB."""

    def test_memory_idle_under_200mb(self, tmp_path):
        """Instanciar componentes em idle deve adicionar < 200MB ao RSS do processo.

        Mede o delta de RSS antes e depois da instanciação, não o RSS absoluto,
        pois o processo pytest já carrega várias dependências antes do teste.
        ru_maxrss no macOS registra o pico histórico, então usamos tracemalloc
        para medir alocações Python da instanciação dos componentes.
        """
        import tracemalloc

        repo = tmp_path / "idle-repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        # Garante que os módulos já foram importados (aquece o cache de imports)
        import src.chunker  # noqa: F401
        import src.embeddings  # noqa: F401
        import src.errors  # noqa: F401
        import src.indexer  # noqa: F401
        import src.metrics  # noqa: F401
        import src.storage  # noqa: F401

        tracemalloc.start()

        config = _make_config()
        storage = Storage(config, repo, base_path=tmp_path / "indices")
        provider = LocalFastembedProvider(config)
        _ = Indexer(config, storage, provider, repo_path=repo)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)

        assert peak_mb < 200, (
            f"NFR6 FALHOU: alocação de memória em idle = {peak_mb:.1f}MB "
            f"(limite: 200MB). Verifique instanciação de componentes."
        )
