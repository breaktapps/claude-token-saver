"""
Acceptance tests for Story 10.7: Test File Association.

Validates that _expand_test_files finds test files matching source files
in search results, using _test suffix, test_ prefix, .test, and .spec patterns.
"""

import random

import pytest

from src.config import Config
from src.embeddings import LITE_DIMENSION
from src.server import _expand_test_files
from src.storage import Storage


DIM = LITE_DIMENSION


def _random_vector(seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(DIM)]


def _make_chunk(
    file_path: str = "src/billing.py",
    chunk_type: str = "function",
    name: str = "process_payment",
    line_start: int = 1,
    line_end: int = 15,
    content: str = "def process_payment(): pass",
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


def _make_result(
    file_path: str,
    name: str = "my_func",
    language: str = "python",
    score: float = 0.9,
) -> dict:
    return {
        "file_path": file_path,
        "name": name,
        "chunk_type": "function",
        "line_start": 1,
        "line_end": 10,
        "content": f"def {name}(): pass",
        "score": score,
        "language": language,
        "stale": False,
        "parent_name": "",
    }


class TestTestSuffixPattern:
    """AC1: Padrão {name}_test deve ser encontrado."""

    def test_dart_test_suffix_found(self, storage):
        """feature_gate.dart → feature_gate_test.dart deve ser encontrado."""
        test_chunk = _make_chunk(
            file_path="test/feature_gate_test.dart",
            name="featureGateTest",
            language="dart",
            content="void main() { test('feature gate', () {}); }",
            seed=1,
        )
        storage.upsert([test_chunk])

        result = _make_result(
            file_path="lib/feature_gate.dart",
            name="FeatureGate",
            language="dart",
            score=0.9,
        )

        test_files = _expand_test_files([result], storage)

        found_fps = [t["file_path"] for t in test_files]
        assert "test/feature_gate_test.dart" in found_fps

    def test_python_test_suffix_found(self, storage):
        """billing.py → billing_test.py deve ser encontrado."""
        test_chunk = _make_chunk(
            file_path="tests/billing_test.py",
            name="test_process_payment",
            language="python",
            content="def test_process_payment(): assert True",
            seed=2,
        )
        storage.upsert([test_chunk])

        result = _make_result(
            file_path="src/billing.py",
            name="process_payment",
            language="python",
            score=0.85,
        )

        test_files = _expand_test_files([result], storage)

        found_fps = [t["file_path"] for t in test_files]
        assert "tests/billing_test.py" in found_fps


class TestTestPrefixPattern:
    """AC2: Padrão test_{name} deve ser encontrado."""

    def test_test_prefix_found(self, storage):
        """auth.py → test_auth.py deve ser encontrado."""
        test_chunk = _make_chunk(
            file_path="tests/test_auth.py",
            name="test_login",
            language="python",
            content="def test_login(): assert True",
            seed=3,
        )
        storage.upsert([test_chunk])

        result = _make_result(
            file_path="src/auth.py",
            name="login",
            language="python",
            score=0.88,
        )

        test_files = _expand_test_files([result], storage)

        found_fps = [t["file_path"] for t in test_files]
        assert "tests/test_auth.py" in found_fps


class TestDotPatterns:
    """AC3: Padrões .test e .spec devem ser encontrados."""

    def test_dot_test_pattern_found(self, storage):
        """payment.ts → payment.test.ts deve ser encontrado."""
        test_chunk = _make_chunk(
            file_path="src/payment.test.ts",
            name="paymentTest",
            language="typescript",
            content="describe('payment', () => { it('works', () => {}); });",
            seed=4,
        )
        storage.upsert([test_chunk])

        result = _make_result(
            file_path="src/payment.ts",
            name="processPayment",
            language="typescript",
            score=0.9,
        )

        test_files = _expand_test_files([result], storage)

        found_fps = [t["file_path"] for t in test_files]
        assert "src/payment.test.ts" in found_fps

    def test_dot_spec_pattern_found(self, storage):
        """auth.ts → auth.spec.ts deve ser encontrado."""
        test_chunk = _make_chunk(
            file_path="src/auth.spec.ts",
            name="authSpec",
            language="typescript",
            content="describe('auth', () => {});",
            seed=5,
        )
        storage.upsert([test_chunk])

        result = _make_result(
            file_path="src/auth.ts",
            name="authenticate",
            language="typescript",
            score=0.87,
        )

        test_files = _expand_test_files([result], storage)

        found_fps = [t["file_path"] for t in test_files]
        assert "src/auth.spec.ts" in found_fps


class TestNoTestFileFound:
    """AC4: Sem arquivo de teste → nada adicionado."""

    def test_no_test_file_returns_empty(self, storage):
        """Quando não existe arquivo de teste no índice, retornar lista vazia."""
        result = _make_result(
            file_path="src/orphan_module.py",
            name="orphan_func",
            language="python",
            score=0.8,
        )

        test_files = _expand_test_files([result], storage)

        assert test_files == []

    def test_empty_results_returns_empty(self, storage):
        test_files = _expand_test_files([], storage)
        assert test_files == []


class TestTestFilesAtEnd:
    """AC5: Arquivos de teste devem aparecer no FINAL da lista de resultados."""

    def test_test_files_appended_after_source_results(self, storage):
        """Test files retornados por _expand_test_files têm test_for e score correto."""
        test_chunk = _make_chunk(
            file_path="tests/billing_test.py",
            name="test_billing",
            language="python",
            content="def test_billing(): pass",
            seed=6,
        )
        storage.upsert([test_chunk])

        source_result = _make_result(
            file_path="src/billing.py",
            name="process_billing",
            language="python",
            score=0.9,
        )

        test_files = _expand_test_files([source_result], storage)

        assert len(test_files) >= 1
        tf = next(t for t in test_files if t["file_path"] == "tests/billing_test.py")
        assert tf["test_for"] == "src/billing.py"
        assert abs(tf["score"] - 0.9 * 0.6) < 1e-5

    def test_test_files_have_test_for_flag(self, storage):
        """Cada test file deve ter campo test_for apontando para o arquivo fonte."""
        test_chunk = _make_chunk(
            file_path="tests/auth_test.py",
            name="test_auth_flow",
            language="python",
            content="def test_auth_flow(): pass",
            seed=7,
        )
        storage.upsert([test_chunk])

        source_result = _make_result(
            file_path="src/auth.py",
            name="authenticate",
            language="python",
            score=0.85,
        )

        test_files = _expand_test_files([source_result], storage)

        found = [t for t in test_files if t["file_path"] == "tests/auth_test.py"]
        assert len(found) >= 1
        assert found[0]["test_for"] == "src/auth.py"


class TestDeduplication:
    """Test files já presentes nos resultados não devem ser duplicados."""

    def test_already_in_results_skipped(self, storage):
        """Se o arquivo de teste já está nos resultados, não adicionar novamente."""
        test_chunk = _make_chunk(
            file_path="tests/billing_test.py",
            name="test_billing",
            language="python",
            content="def test_billing(): pass",
            seed=8,
        )
        storage.upsert([test_chunk])

        source_result = _make_result(
            file_path="src/billing.py",
            name="process_billing",
            language="python",
            score=0.9,
        )
        # Test file already present in results
        test_already_in = _make_result(
            file_path="tests/billing_test.py",
            name="test_billing",
            language="python",
            score=0.5,
        )

        test_files = _expand_test_files([source_result, test_already_in], storage)

        found_fps = [t["file_path"] for t in test_files]
        assert found_fps.count("tests/billing_test.py") == 0
