"""
Acceptance tests for Story 6.1: MCP Instructions & Tool Descriptions.

Validates MCP handshake instructions, tool descriptions with competitive
keywords, and agent steering toward plugin tools.

FRs: FR40 (MCP instructions), FR41 (optimized descriptions)
"""

import asyncio

import pytest

from src.server import mcp

# Helpers
def _get_instructions() -> str | None:
    return mcp.instructions


def _get_tool_descriptions() -> dict[str, str]:
    """Return {tool_name: description} for all registered tools."""
    tools = asyncio.get_event_loop().run_until_complete(mcp.list_tools())
    return {t.name: (t.description or "") for t in tools}


class TestMcpHandshakeInstructions:
    """AC: Given MCP server starts and handshake occurs,
    When InitializeResult is returned, Then instructions field is present."""

    def test_instructions_field_present(self):
        """InitializeResult deve conter campo instructions."""
        instructions = _get_instructions()
        assert instructions is not None
        assert isinstance(instructions, str)
        assert len(instructions.strip()) > 0

    def test_instructions_around_150_words(self):
        """Instructions deve ter ~150 palavras orientando o agente."""
        instructions = _get_instructions()
        word_count = len(instructions.split())
        # Allow range 100-200 words
        assert 100 <= word_count <= 200, (
            f"Instructions tem {word_count} palavras; esperado entre 100 e 200"
        )

    def test_instructions_fit_2kb_limit(self):
        """Instructions deve caber no limite de 2KB do MCP."""
        instructions = _get_instructions()
        size_bytes = len(instructions.encode("utf-8"))
        assert size_bytes <= 2048, (
            f"Instructions tem {size_bytes} bytes; limite é 2048"
        )

    def test_instructions_mention_prefer_over_grep_read(self):
        """Instructions deve mencionar 'prefer over Grep/Read/Glob'."""
        instructions = _get_instructions().lower()
        assert "prefer" in instructions
        assert "grep" in instructions
        assert "read" in instructions

    def test_instructions_specify_entry_point_tool(self):
        """Instructions deve especificar a tool de entrada."""
        instructions = _get_instructions().lower()
        # search_semantic é o entry point definido no story file
        assert "search_semantic" in instructions


class TestToolDescriptions:
    """AC: Given each MCP tool is registered, When descriptions fetched,
    Then each has 'USE INSTEAD OF' and competitive keywords."""

    def test_search_semantic_description(self):
        """search_semantic deve ter 'USE INSTEAD OF Grep'."""
        descs = _get_tool_descriptions()
        desc = descs.get("search_semantic", "")
        assert "USE INSTEAD OF" in desc
        assert "Grep" in desc

    def test_search_exact_description(self):
        """search_exact deve ter 'USE INSTEAD OF Grep'."""
        descs = _get_tool_descriptions()
        desc = descs.get("search_exact", "")
        assert "USE INSTEAD OF" in desc
        assert "Grep" in desc

    def test_get_file_description(self):
        """get_file deve ter 'USE INSTEAD OF Read'."""
        descs = _get_tool_descriptions()
        desc = descs.get("get_file", "")
        assert "USE INSTEAD OF" in desc
        assert "Read" in desc

    def test_descriptions_have_competitive_keywords(self):
        """Cada descricao deve incluir keywords para discoverability."""
        descs = _get_tool_descriptions()
        # search_semantic deve mencionar "semantic" ou "natural language"
        sem = descs.get("search_semantic", "").lower()
        assert "semantic" in sem or "natural language" in sem
        # search_exact deve mencionar "exact" ou "function" ou "class"
        exact = descs.get("search_exact", "").lower()
        assert "exact" in exact or "function" in exact or "class" in exact

    def test_each_description_fits_2kb(self):
        """Cada descricao deve caber no limite de 2KB."""
        descs = _get_tool_descriptions()
        for name, desc in descs.items():
            size_bytes = len(desc.encode("utf-8"))
            assert size_bytes <= 2048, (
                f"Tool '{name}' description tem {size_bytes} bytes; limite é 2048"
            )


class TestToolSteering:
    """AC: Given agent needs to search/read code,
    Then descriptions guide it to prefer plugin tools."""

    def test_search_description_steers_away_from_grep(self):
        """Descricao deve guiar agente para preferir search_semantic/exact sobre Grep."""
        descs = _get_tool_descriptions()
        # Pelo menos uma das duas search tools deve dizer explicitamente para usar
        # em vez de Grep
        semantic = descs.get("search_semantic", "")
        exact = descs.get("search_exact", "")
        combined = semantic + exact
        assert "Grep" in combined
        assert "USE INSTEAD OF" in combined

    def test_get_file_description_steers_away_from_read(self):
        """Descricao deve guiar agente para preferir get_file sobre Read."""
        descs = _get_tool_descriptions()
        get_file = descs.get("get_file", "")
        assert "Read" in get_file
        assert "USE INSTEAD OF" in get_file
