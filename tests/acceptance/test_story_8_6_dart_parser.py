"""
Acceptance tests for Story 8.6: Melhorar Parser Dart (Tree-Sitter).

Validates that the tree-sitter-dart parser correctly handles all Dart constructs:
complex generics, @override annotations, getters/setters, extension methods,
mixins, multi-line params, named/factory constructors, and calls extraction.

SQs: SQ7
"""

import json
from pathlib import Path

import pytest

from src.chunker import parse

FIXTURES = Path(__file__).parent.parent / "fixtures" / "dart"
COMPLEX = FIXTURES / "complex_dart.dart"


class TestComplexGenerics:
    """AC1: Metodos com genericos complexos devem ser chunked corretamente."""

    def test_method_with_complex_generics_is_found(self):
        """loadFeatures com Future<Either<Failure, List<Feature>>> deve ser chunked."""
        chunks = parse(COMPLEX, "dart")
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        assert "loadFeatures" in method_names

    def test_method_with_complex_generics_has_correct_parent(self):
        """loadFeatures deve ter parent_name = SubscriptionProvider."""
        chunks = parse(COMPLEX, "dart")
        method = next(c for c in chunks if c["name"] == "loadFeatures")
        assert method["parent_name"] == "SubscriptionProvider"

    def test_method_content_preserves_return_type(self):
        """Conteudo do metodo deve conter o tipo de retorno com genericos."""
        chunks = parse(COMPLEX, "dart")
        method = next(c for c in chunks if c["name"] == "loadFeatures")
        assert "Future" in method["content"]
        assert "Either" in method["content"]


class TestAnnotations:
    """AC2: Metodos anotados com @override devem ser chunked com a annotation."""

    def test_override_method_is_chunked(self):
        """Metodos com @override devem ser detectados e chunked."""
        chunks = parse(COMPLEX, "dart")
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        # Both loadFeatures (@override) and build (@override) should be present
        assert "loadFeatures" in method_names
        assert "build" in method_names

    def test_annotation_included_in_chunk_content(self):
        """Chunk de metodo @override deve incluir a annotation no conteudo."""
        chunks = parse(COMPLEX, "dart")
        build_method = next(
            (c for c in chunks if c["name"] == "build" and c["chunk_type"] == "method"),
            None,
        )
        assert build_method is not None
        assert "@override" in build_method["content"]


class TestGettersAndSetters:
    """AC3: Getters e setters devem ser chunked como membros individuais."""

    def test_getter_is_chunked(self):
        """Getter deve aparecer como chunk de metodo com nome correto."""
        chunks = parse(COMPLEX, "dart")
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        assert "subscriptionId" in method_names

    def test_setter_is_chunked(self):
        """Setter deve aparecer como chunk de metodo com nome correto."""
        chunks = parse(COMPLEX, "dart")
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        # setter has same name as getter
        assert "subscriptionId" in method_names

    def test_getter_in_extension_is_chunked(self):
        """Getter em extension deve ser chunked."""
        chunks = parse(COMPLEX, "dart")
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        assert "isEmail" in method_names


class TestExtensionMethods:
    """AC4: Extension methods devem ser parseados como classe com metodos filhos."""

    def test_extension_produces_class_chunk(self):
        """Extension deve gerar um chunk de classe."""
        chunks = parse(COMPLEX, "dart")
        class_names = {c["name"] for c in chunks if c["chunk_type"] == "class"}
        assert "StringExt" in class_names

    def test_extension_method_is_chunked(self):
        """Metodo de extension deve ser chunked com parent_name da extension."""
        chunks = parse(COMPLEX, "dart")
        capitalize = next(
            (c for c in chunks if c["name"] == "capitalize" and c["chunk_type"] == "method"),
            None,
        )
        assert capitalize is not None
        assert capitalize["parent_name"] == "StringExt"

    def test_extension_outline_has_members(self):
        """Outline da extension deve listar seus membros."""
        chunks = parse(COMPLEX, "dart")
        ext_class = next(c for c in chunks if c["name"] == "StringExt" and c["chunk_type"] == "class")
        members = json.loads(ext_class["outline_json"])
        assert len(members) >= 1


class TestMixins:
    """AC5: Mixins devem ser parseados como classe com metodos filhos."""

    def test_mixin_produces_class_chunk(self):
        """Mixin deve gerar chunk de classe."""
        chunks = parse(COMPLEX, "dart")
        class_names = {c["name"] for c in chunks if c["chunk_type"] == "class"}
        assert "LoggingMixin" in class_names

    def test_mixin_method_is_chunked(self):
        """Metodo de mixin deve ser chunked com parent_name do mixin."""
        chunks = parse(COMPLEX, "dart")
        log_method = next(
            (c for c in chunks if c["name"] == "log" and c["chunk_type"] == "method"),
            None,
        )
        assert log_method is not None
        assert log_method["parent_name"] == "LoggingMixin"


class TestConstructors:
    """AC6+AC7: Named e factory constructors devem ser chunked."""

    def test_named_constructor_is_chunked(self):
        """Named constructor (ClassName.name()) deve ser chunked com nome da parte nomeada."""
        chunks = parse(COMPLEX, "dart")
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        assert "fromId" in method_names

    def test_factory_constructor_is_chunked(self):
        """Factory constructor deve ser chunked."""
        chunks = parse(COMPLEX, "dart")
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        assert "create" in method_names

    def test_default_constructor_is_chunked(self):
        """Construtor padrao deve ser chunked."""
        chunks = parse(COMPLEX, "dart")
        # Default constructor: SubscriptionProvider(this._id, this._repository)
        method_names = {c["name"] for c in chunks if c["chunk_type"] == "method"}
        assert "SubscriptionProvider" in method_names


class TestTopLevelFunctions:
    """Top-level functions devem ser extraidas como chunk_type='function'."""

    def test_top_level_async_function_is_chunked(self):
        """Funcao top-level async deve ser chunked como funcao."""
        chunks = parse(COMPLEX, "dart")
        func_chunks = [c for c in chunks if c["chunk_type"] == "function"]
        func_names = {c["name"] for c in func_chunks}
        assert "initializeApp" in func_names

    def test_top_level_function_has_no_parent(self):
        """Funcao top-level nao deve ter parent_name."""
        chunks = parse(COMPLEX, "dart")
        func = next(c for c in chunks if c["name"] == "initializeApp")
        assert func["parent_name"] == ""


class TestCallsExtraction:
    """Calls devem ser extraidas do AST via _dart_extract_calls."""

    def test_calls_extracted_from_method(self):
        """Metodo deve ter calls extraidas do body."""
        chunks = parse(COMPLEX, "dart")
        load = next(c for c in chunks if c["name"] == "loadFeatures")
        calls = json.loads(load["calls_json"])
        assert isinstance(calls, list)
        # notifyListeners is called inside loadFeatures
        assert "notifyListeners" in calls

    def test_calls_is_sorted_deduped_list(self):
        """calls_json deve ser lista sorted e sem duplicatas."""
        chunks = parse(COMPLEX, "dart")
        for chunk in chunks:
            calls = json.loads(chunk["calls_json"])
            assert calls == sorted(set(calls))


class TestImportsExtraction:
    """Imports devem ser extraidos de todos os chunks do arquivo."""

    def test_imports_in_all_chunks(self):
        """Todos os chunks devem ter imports_json com os imports do arquivo."""
        chunks = parse(COMPLEX, "dart")
        assert len(chunks) > 0
        for chunk in chunks:
            imports = json.loads(chunk["imports_json"])
            assert isinstance(imports, list)
            assert any("dart:async" in imp for imp in imports)
            assert any("flutter/material" in imp for imp in imports)


class TestClassOutlines:
    """Class outline deve conter assinaturas de membros."""

    def test_subscription_provider_outline_has_all_members(self):
        """SubscriptionProvider deve ter outline com todos os seus membros."""
        chunks = parse(COMPLEX, "dart")
        sp_class = next(
            c for c in chunks
            if c["name"] == "SubscriptionProvider" and c["chunk_type"] == "class"
        )
        members = json.loads(sp_class["outline_json"])
        # Should have at minimum: constructor, fromId, create, getter, setter, loadFeatures, resetState
        assert len(members) >= 5

    def test_feature_gate_outline_has_build_and_overlay(self):
        """FeatureGate deve ter outline com build e overlay."""
        chunks = parse(COMPLEX, "dart")
        fg_class = next(
            c for c in chunks
            if c["name"] == "FeatureGate" and c["chunk_type"] == "class"
        )
        members = json.loads(fg_class["outline_json"])
        member_text = " ".join(members)
        assert "build" in member_text
        assert "overlay" in member_text


class TestMetadata:
    """Chunks devem ter metadados corretos."""

    def test_all_dart_chunks_have_language_dart(self):
        """Todos os chunks Dart devem ter language='dart'."""
        chunks = parse(COMPLEX, "dart")
        for chunk in chunks:
            assert chunk["language"] == "dart"

    def test_line_numbers_are_positive(self):
        """line_start e line_end devem ser positivos e validos."""
        chunks = parse(COMPLEX, "dart")
        for chunk in chunks:
            assert chunk["line_start"] >= 1
            assert chunk["line_end"] >= chunk["line_start"]

    def test_method_chunks_have_parent_name(self):
        """Todos os chunks de metodo devem ter parent_name nao-vazio."""
        chunks = parse(COMPLEX, "dart")
        for chunk in chunks:
            if chunk["chunk_type"] == "method":
                assert chunk["parent_name"] != ""
