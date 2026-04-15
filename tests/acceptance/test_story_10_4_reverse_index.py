"""
Acceptance tests for Story 10.4: Reverse Index / called_by.

Validates:
- callers.json is built during reindex
- get_callers returns correct callers
- _expand_via_callers adds caller chunks with caller_of field
- Generic names (50+ callers) are filtered out
- search_exact is NOT affected by caller expansion
"""

import json
import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.indexer import Indexer
from src.server import _expand_via_callers
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


# ---------------------------------------------------------------------------
# AC1: callers.json é gerado corretamente pelo reindex
# ---------------------------------------------------------------------------

class TestBuildCallersIndex:
    """callers.json é criado no diretório do índice após reindex."""

    def test_callers_json_created_after_reindex(self, tmp_path):
        """Após reindex, callers.json deve existir no diretório do índice."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        repo_path = tmp_path / "fake-repo"
        repo_path.mkdir()

        storage = Storage(config, repo_path, base_path=tmp_path / "indices")

        # Insere chunks diretamente (simulando o que o indexer faria)
        caller_chunk = _make_chunk(
            name="process_order",
            calls_json='["validate_payment", "charge_card"]',
            seed=1,
        )
        callee_a = _make_chunk(name="validate_payment", seed=2)
        callee_b = _make_chunk(name="charge_card", seed=3)
        storage.upsert([caller_chunk, callee_a, callee_b])

        # Cria indexer e chama _build_callers_index diretamente
        embed_provider = MagicMock()
        embed_provider.embed_texts.return_value = [_random_vector() for _ in range(3)]
        indexer = Indexer(config, storage, embed_provider, repo_path=repo_path)
        indexer._build_callers_index()

        callers_path = storage.index_path / "callers.json"
        assert callers_path.exists(), "callers.json deve ser criado"

    def test_callers_json_content_correct(self, tmp_path):
        """callers.json deve mapear called_name -> [caller_names] corretamente."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        repo_path = tmp_path / "fake-repo"
        repo_path.mkdir()

        storage = Storage(config, repo_path, base_path=tmp_path / "indices")

        chunk_a = _make_chunk(name="router", calls_json='["handler_a", "handler_b"]', seed=1)
        chunk_b = _make_chunk(name="dispatcher", calls_json='["handler_a"]', seed=2)
        chunk_c = _make_chunk(name="handler_a", calls_json="[]", seed=3)
        chunk_d = _make_chunk(name="handler_b", calls_json="[]", seed=4)
        storage.upsert([chunk_a, chunk_b, chunk_c, chunk_d])

        embed_provider = MagicMock()
        indexer = Indexer(config, storage, embed_provider, repo_path=repo_path)
        indexer._build_callers_index()

        callers_path = storage.index_path / "callers.json"
        callers = json.loads(callers_path.read_text(encoding="utf-8"))

        assert "handler_a" in callers
        assert set(callers["handler_a"]) == {"router", "dispatcher"}
        assert "handler_b" in callers
        assert callers["handler_b"] == ["router"]

    def test_callers_json_empty_when_no_calls(self, tmp_path):
        """Sem calls_json preenchido, callers.json deve ser dict vazio."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        repo_path = tmp_path / "fake-repo"
        repo_path.mkdir()

        storage = Storage(config, repo_path, base_path=tmp_path / "indices")
        storage.upsert([_make_chunk(name="lonely", calls_json="[]", seed=1)])

        embed_provider = MagicMock()
        indexer = Indexer(config, storage, embed_provider, repo_path=repo_path)
        indexer._build_callers_index()

        callers_path = storage.index_path / "callers.json"
        callers = json.loads(callers_path.read_text(encoding="utf-8"))
        assert callers == {}


# ---------------------------------------------------------------------------
# AC2: get_callers retorna os callers corretos
# ---------------------------------------------------------------------------

class TestGetCallers:
    """Storage.get_callers() carrega e retorna corretamente do cache."""

    def test_get_callers_returns_correct_list(self, storage):
        """get_callers deve retornar a lista de callers para um nome indexado."""
        callers_data = {"pay": ["checkout", "retry_payment"]}
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps(callers_data), encoding="utf-8")

        result = storage.get_callers("pay")
        assert set(result) == {"checkout", "retry_payment"}

    def test_get_callers_returns_empty_for_unknown(self, storage):
        """get_callers deve retornar lista vazia para nome desconhecido."""
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text("{}", encoding="utf-8")

        result = storage.get_callers("nonexistent")
        assert result == []

    def test_get_callers_returns_empty_when_no_file(self, storage):
        """get_callers deve retornar lista vazia se callers.json não existe."""
        result = storage.get_callers("anything")
        assert result == []

    def test_get_callers_uses_cache(self, storage):
        """get_callers não deve reler o arquivo em chamadas subsequentes."""
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps({"foo": ["bar"]}), encoding="utf-8")

        result1 = storage.get_callers("foo")
        assert result1 == ["bar"]

        # Modifica o arquivo — cache deve ser usado, resultado não muda
        callers_path.write_text(json.dumps({"foo": ["baz"]}), encoding="utf-8")
        result2 = storage.get_callers("foo")
        assert result2 == ["bar"]  # still cached

    def test_invalidate_callers_cache_forces_reload(self, storage):
        """invalidate_callers_cache deve forçar recarga do arquivo."""
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps({"foo": ["bar"]}), encoding="utf-8")
        storage.get_callers("foo")  # warm cache

        callers_path.write_text(json.dumps({"foo": ["baz"]}), encoding="utf-8")
        storage.invalidate_callers_cache()

        result = storage.get_callers("foo")
        assert result == ["baz"]


# ---------------------------------------------------------------------------
# AC3: _expand_via_callers adiciona chunks de callers
# ---------------------------------------------------------------------------

class TestExpandViaCallers:
    """_expand_via_callers retorna chunks que chamam os resultados diretos."""

    def test_caller_added_with_caller_of_field(self, storage):
        """Caller deve aparecer com caller_of apontando para o nome do resultado."""
        caller_chunk = _make_chunk(
            name="checkout",
            content="def checkout(): pay()",
            seed=10,
        )
        storage.upsert([caller_chunk])

        # Simula callers.json
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps({"pay": ["checkout"]}), encoding="utf-8")

        result = {
            "file_path": "src/payment.py",
            "name": "pay",
            "chunk_type": "function",
            "line_start": 1,
            "line_end": 5,
            "content": "def pay(): ...",
            "score": 0.9,
            "language": "python",
            "stale": False,
            "parent_name": "",
        }

        expanded = _expand_via_callers([result], top_k=10, storage=storage)

        assert len(expanded) == 1
        exp = expanded[0]
        assert exp["name"] == "checkout"
        assert exp["caller_of"] == "pay"
        assert exp["expanded"] is True
        assert abs(exp["score"] - 0.9 * 0.75) < 1e-5

    def test_caller_not_duplicated_if_already_in_results(self, storage):
        """Caller já presente nos resultados não deve ser duplicado."""
        caller_chunk = _make_chunk(name="checkout", seed=10)
        storage.upsert([caller_chunk])

        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps({"pay": ["checkout"]}), encoding="utf-8")

        results = [
            {"name": "pay", "score": 0.9, "file_path": "a.py", "chunk_type": "function",
             "line_start": 1, "line_end": 5, "content": "x", "language": "python",
             "stale": False, "parent_name": ""},
            {"name": "checkout", "score": 0.8, "file_path": "b.py", "chunk_type": "function",
             "line_start": 1, "line_end": 5, "content": "x", "language": "python",
             "stale": False, "parent_name": ""},
        ]

        expanded = _expand_via_callers(results, top_k=10, storage=storage)
        assert all(e["name"] != "checkout" for e in expanded)

    def test_multiple_callers_added(self, storage):
        """Múltiplos callers devem ser incluídos até o limite."""
        chunk_a = _make_chunk(name="caller_a", seed=1)
        chunk_b = _make_chunk(name="caller_b", seed=2)
        storage.upsert([chunk_a, chunk_b])

        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(
            json.dumps({"target_func": ["caller_a", "caller_b"]}), encoding="utf-8"
        )

        result = {
            "name": "target_func", "score": 0.8, "file_path": "src/x.py",
            "chunk_type": "function", "line_start": 1, "line_end": 10,
            "content": "x", "language": "python", "stale": False, "parent_name": "",
        }

        expanded = _expand_via_callers([result], top_k=10, storage=storage)
        names = {e["name"] for e in expanded}
        assert "caller_a" in names
        assert "caller_b" in names

    def test_returns_empty_when_no_callers(self, storage):
        """Deve retornar lista vazia se não há callers para os resultados."""
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text("{}", encoding="utf-8")

        result = {
            "name": "orphan_func", "score": 0.7, "file_path": "src/x.py",
            "chunk_type": "function", "line_start": 1, "line_end": 5,
            "content": "x", "language": "python", "stale": False, "parent_name": "",
        }

        expanded = _expand_via_callers([result], top_k=10, storage=storage)
        assert expanded == []

    def test_max_10_callers_total(self, storage):
        """Expansão de callers deve ser limitada a 10 no total."""
        chunks = [_make_chunk(name=f"caller_{i}", seed=i) for i in range(20)]
        storage.upsert(chunks)

        callers_data = {"popular_func": [f"caller_{i}" for i in range(20)]}
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps(callers_data), encoding="utf-8")

        result = {
            "name": "popular_func", "score": 0.8, "file_path": "src/x.py",
            "chunk_type": "function", "line_start": 1, "line_end": 5,
            "content": "x", "language": "python", "stale": False, "parent_name": "",
        }

        expanded = _expand_via_callers([result], top_k=10, storage=storage)
        assert len(expanded) <= 10


# ---------------------------------------------------------------------------
# AC4: Nomes genéricos (50+ callers) são filtrados
# ---------------------------------------------------------------------------

class TestGenericNameFilter:
    """Nomes com 50+ callers não devem disparar expansão (muito genéricos)."""

    def test_generic_name_skipped(self, storage):
        """Se um resultado tem nome que aparece em 50+ callers, não expande."""
        # Cria 50 callers para "build"
        callers_list = [f"module_{i}" for i in range(50)]
        callers_data = {"build": callers_list}
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps(callers_data), encoding="utf-8")

        # Insere alguns dos callers no índice
        chunks = [_make_chunk(name=f"module_{i}", seed=i) for i in range(5)]
        storage.upsert(chunks)

        result = {
            "name": "build", "score": 0.9, "file_path": "src/builder.py",
            "chunk_type": "function", "line_start": 1, "line_end": 5,
            "content": "def build(): ...", "language": "python",
            "stale": False, "parent_name": "",
        }

        expanded = _expand_via_callers([result], top_k=10, storage=storage)
        assert expanded == [], "Nome genérico não deve gerar expansão"

    def test_non_generic_name_not_skipped(self, storage):
        """Nome com menos de 50 callers deve gerar expansão normalmente."""
        callers_data = {"specific_func": [f"caller_{i}" for i in range(3)]}
        callers_path = storage.index_path / "callers.json"
        callers_path.write_text(json.dumps(callers_data), encoding="utf-8")

        chunks = [_make_chunk(name=f"caller_{i}", seed=i + 100) for i in range(3)]
        storage.upsert(chunks)

        result = {
            "name": "specific_func", "score": 0.8, "file_path": "src/x.py",
            "chunk_type": "function", "line_start": 1, "line_end": 5,
            "content": "x", "language": "python", "stale": False, "parent_name": "",
        }

        expanded = _expand_via_callers([result], top_k=10, storage=storage)
        assert len(expanded) > 0


# ---------------------------------------------------------------------------
# AC5: search_exact não é afetado pela expansão de callers
# ---------------------------------------------------------------------------

class TestSearchExactNotAffected:
    """search_exact não deve chamar _expand_via_callers."""

    def test_expand_via_callers_not_called_in_search_exact(self):
        """_expand_via_callers não deve ser invocado durante search_exact."""
        import asyncio
        import src.server as server_module

        with patch.object(server_module, "_expand_via_callers") as mock_expand:
            with patch.object(server_module, "_init_components") as mock_init:
                config = MagicMock()
                storage = MagicMock()
                storage.get_file_hashes.return_value = {"src/x.py": "abc"}
                storage.is_stale.return_value = False
                storage.search_fts.return_value = []
                storage.index_path = Path("/tmp/fake_index")
                indexer = MagicMock()
                embed_provider = MagicMock()
                repo_path = Path("/tmp/fake_repo")
                mock_init.return_value = (config, storage, indexer, embed_provider, repo_path)

                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(
                        server_module.search_exact(query="some_function", top_k=5)
                    )
                finally:
                    loop.close()

            mock_expand.assert_not_called()
