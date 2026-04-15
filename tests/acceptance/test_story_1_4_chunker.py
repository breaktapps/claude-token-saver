"""
Acceptance tests for Story 1.4: AST Code Chunking (3 Linguagens MVP).

Validates tree-sitter based parsing for Dart, Python, and TypeScript
with correct chunk generation, metadata extraction, and file filtering.

FRs: FR12 (language detection), FR31 (.gitignore + extra filters)
NFRs: NFR14 (tree-sitter grammars as pip packages)
"""

import json
from pathlib import Path

import pytest

from src.chunker import detect_language, parse, scan_files

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestDartParsing:
    """AC: Given a Dart file with a class containing 3 methods,
    When parse is called, Then 1 class outline + method chunks are produced."""

    def test_dart_class_produces_outline_chunk(self):
        """Classe Dart deve gerar 1 chunk outline com assinaturas dos membros."""
        chunks = parse(FIXTURES / "dart" / "audio_service.dart", "dart")
        class_chunks = [c for c in chunks if c["chunk_type"] == "class"]
        assert len(class_chunks) == 1
        outline = class_chunks[0]
        assert outline["name"] == "AudioService"
        # Outline should have member signatures in outline_json
        # tree-sitter parser also captures the constructor
        members = json.loads(outline["outline_json"])
        assert len(members) >= 3  # initialize, processAudio, dispose (+ constructor)
        names_in_outline = " ".join(members)
        assert "initialize" in names_in_outline
        assert "processAudio" in names_in_outline
        assert "dispose" in names_in_outline

    def test_dart_class_produces_method_chunks(self):
        """Classe Dart com metodos deve gerar chunks de metodo."""
        chunks = parse(FIXTURES / "dart" / "audio_service.dart", "dart")
        method_chunks = [c for c in chunks if c["chunk_type"] == "method"]
        assert len(method_chunks) >= 3
        method_names = {c["name"] for c in method_chunks}
        assert "initialize" in method_names
        assert "processAudio" in method_names
        assert "dispose" in method_names
        # All methods should have parent_name set
        for mc in method_chunks:
            assert mc["parent_name"] == "AudioService"

    def test_dart_method_chunks_have_calls_list(self):
        """Chunks de metodo Dart devem incluir lista de calls extraida do AST."""
        chunks = parse(FIXTURES / "dart" / "audio_service.dart", "dart")
        process_chunk = next(c for c in chunks if c["name"] == "processAudio")
        calls = json.loads(process_chunk["calls_json"])
        assert isinstance(calls, list)
        assert len(calls) > 0
        # Should contain decodeAudio and applyFilter
        assert "decodeAudio" in calls
        assert "applyFilter" in calls


class TestPythonParsing:
    """AC: Given a Python file with 2 top-level functions and 1 class,
    When parse is called, Then functions and class are correctly chunked."""

    def test_python_top_level_functions_as_separate_chunks(self):
        """Cada funcao top-level Python deve ser um chunk separado."""
        chunks = parse(FIXTURES / "python" / "data_processor.py", "python")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(func_chunks) == 2
        names = {c["name"] for c in func_chunks}
        assert "load_data" in names
        assert "transform_records" in names

    def test_python_function_chunks_have_signature_and_calls(self):
        """Chunks de funcao Python devem ter code, signature e calls."""
        chunks = parse(FIXTURES / "python" / "data_processor.py", "python")
        transform = next(c for c in chunks if c["name"] == "transform_records")
        # Has content with code
        assert "def transform_records" in transform["content"]
        # Has calls extracted
        calls = json.loads(transform["calls_json"])
        assert "normalize_fields" in calls
        assert "validate_record" in calls

    def test_python_class_produces_outline_plus_methods(self):
        """Classe Python deve produzir outline chunk + method chunks."""
        chunks = parse(FIXTURES / "python" / "data_processor.py", "python")
        class_chunks = [c for c in chunks if c["chunk_type"] == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0]["name"] == "DataProcessor"

        method_chunks = [c for c in chunks if c["chunk_type"] == "method"]
        method_names = {c["name"] for c in method_chunks}
        assert "__init__" in method_names
        assert "process" in method_names
        assert "extract_records" in method_names
        assert "validate_batch" in method_names


class TestTypeScriptParsing:
    """AC: Given a TypeScript file with exported functions and interfaces,
    When parse is called, Then chunks are correctly extracted."""

    def test_typescript_exported_functions_chunked(self):
        """Funcoes exportadas TypeScript devem ser extraidas como chunks."""
        chunks = parse(FIXTURES / "typescript" / "api_handler.ts", "typescript")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        func_names = {c["name"] for c in func_chunks}
        assert "createClient" in func_names
        assert "handleError" in func_names

    def test_typescript_preserves_type_information(self):
        """Informacao de tipos TypeScript deve ser preservada nos chunks."""
        chunks = parse(FIXTURES / "typescript" / "api_handler.ts", "typescript")
        create = next(c for c in chunks if c["name"] == "createClient")
        # Content should preserve type annotations
        assert "RequestConfig" in create["content"]
        assert "ApiClient" in create["content"]

        # Interfaces should also be chunked
        class_chunks = [c for c in chunks if c["chunk_type"] == "class"]
        class_names = {c["name"] for c in class_chunks}
        assert "RequestConfig" in class_names
        assert "ApiResponse" in class_names


class TestLargeMethodSplitting:
    """AC: Given a method with >150 lines,
    When chunker parses it, Then it splits at first-level AST nodes."""

    def test_large_method_split_at_ast_nodes(self):
        """Metodo >150 linhas deve ser dividido em nos AST de primeiro nivel."""
        chunks = parse(
            FIXTURES / "python" / "large_method.py",
            "python",
            max_chunk_lines=50,  # Use smaller max to force split
        )
        # process_all should be split into multiple parts
        process_chunks = [c for c in chunks if c["name"].startswith("process_all")]
        assert len(process_chunks) > 1
        # Parts should be named with __partN suffix
        for i, pc in enumerate(process_chunks, 1):
            assert f"__part{i}" in pc["name"]

    def test_split_never_mid_block(self):
        """Divisao nunca deve ocorrer no meio de um bloco."""
        chunks = parse(
            FIXTURES / "python" / "large_method.py",
            "python",
            max_chunk_lines=50,
        )
        process_chunks = [c for c in chunks if c["name"].startswith("process_all")]
        for pc in process_chunks:
            content = pc["content"]
            lines = content.split("\n")
            # Each chunk should start at a statement boundary (indentation level
            # should not be mid-block: no orphan else/elif/except/finally)
            first_line = lines[0].strip()
            assert not first_line.startswith(("else:", "elif ", "except", "finally:"))


class TestDartTopLevelFunctions:
    """C5: Dart top-level functions devem ser extraidas como chunks de funcao."""

    def test_top_level_functions_extracted(self):
        """Funcoes top-level Dart devem virar chunks com chunk_type='function'."""
        chunks = parse(FIXTURES / "dart" / "utils_with_functions.dart", "dart")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        func_names = {c["name"] for c in func_chunks}
        assert "formatDate" in func_names
        assert "clamp" in func_names

    def test_top_level_functions_not_inside_class(self):
        """Funcoes top-level nao devem ter parent_name."""
        chunks = parse(FIXTURES / "dart" / "utils_with_functions.dart", "dart")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        for fc in func_chunks:
            assert fc["parent_name"] == ""

    def test_class_methods_not_duplicated_as_functions(self):
        """Metodos de classe nao devem aparecer como funcoes top-level."""
        chunks = parse(FIXTURES / "dart" / "utils_with_functions.dart", "dart")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        func_names = {c["name"] for c in func_chunks}
        # 'format' is a method inside DateHelper, not a top-level function
        assert "format" not in func_names


class TestImportsExtraction:
    """C2: Imports devem ser extraidos e incluidos nos chunks como imports_json."""

    def test_dart_imports_extracted(self):
        """Imports Dart devem estar em imports_json de todos os chunks do arquivo."""
        chunks = parse(FIXTURES / "dart" / "utils_with_functions.dart", "dart")
        assert len(chunks) > 0
        for chunk in chunks:
            assert "imports_json" in chunk
            imports = json.loads(chunk["imports_json"])
            assert isinstance(imports, list)
            assert any("dart:convert" in imp for imp in imports)
            assert any("flutter/material" in imp for imp in imports)

    def test_python_imports_extracted(self):
        """Imports Python devem estar em imports_json dos chunks."""
        chunks = parse(FIXTURES / "python" / "data_processor.py", "python")
        assert len(chunks) > 0
        for chunk in chunks:
            assert "imports_json" in chunk
            imports = json.loads(chunk["imports_json"])
            assert isinstance(imports, list)
            assert any("import json" in imp for imp in imports)
            assert any("pathlib" in imp for imp in imports)

    def test_typescript_imports_extracted(self):
        """Imports TypeScript devem estar em imports_json dos chunks."""
        # api_handler.ts nao tem imports, mas verificamos que o campo existe
        chunks = parse(FIXTURES / "typescript" / "api_handler.ts", "typescript")
        assert len(chunks) > 0
        for chunk in chunks:
            assert "imports_json" in chunk
            imports = json.loads(chunk["imports_json"])
            assert isinstance(imports, list)


class TestLanguageDetection:
    """AC: Given a file with extension .dart,
    When detect_language is called, Then returns 'dart'."""

    def test_detect_dart(self):
        """Extensao .dart deve retornar linguagem 'dart'."""
        assert detect_language("lib/main.dart") == "dart"

    def test_detect_python(self):
        """Extensao .py deve retornar linguagem 'python'."""
        assert detect_language("src/config.py") == "python"

    def test_detect_typescript(self):
        """Extensao .ts deve retornar linguagem 'typescript'."""
        assert detect_language("src/api.ts") == "typescript"
        assert detect_language("src/App.tsx") == "typescript"


class TestFileFiltering:
    """AC: Given .gitignore and extra filters exist,
    When scanner runs, Then matching files are excluded."""

    def test_gitignore_patterns_excluded(self, tmp_path):
        """Arquivos matching .gitignore (ex: build/) devem ser excluidos."""
        # Create repo structure
        (tmp_path / ".gitignore").write_text("build/\n*.log\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# code")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "output.py").write_text("# built")
        (tmp_path / "debug.log").write_text("log")

        files = scan_files(tmp_path)
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "output.py" not in file_names

    def test_extra_filters_excluded(self, tmp_path):
        """Filtros extras (*.g.dart, __pycache__) devem ser excluidos."""
        # Create repo structure
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "main.dart").write_text("class Foo {}")
        (tmp_path / "lib" / "generated.g.dart").write_text("// generated")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cache.py").write_text("# cache")

        files = scan_files(tmp_path, extra_ignore=["*.g.dart"])
        file_names = [f.name for f in files]
        assert "main.dart" in file_names
        assert "generated.g.dart" not in file_names
        assert "cache.py" not in file_names  # __pycache__ is in DEFAULT_IGNORE
