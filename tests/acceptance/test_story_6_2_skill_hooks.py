"""
Acceptance tests for Story 6.2: Skill Auto-Invocavel & Hooks de Preferencia.

Validates skill injection, PreToolUse nudge hooks, SessionStart compact
hook, and fallback behavior when no index exists.

FRs: FR42 (skill auto-invocable), FR43 (PreToolUse nudge), FR44 (SessionStart compact)
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

# Paths
PLUGIN_DIR = Path(__file__).parent.parent.parent
HOOKS_DIR = PLUGIN_DIR / "hooks"
SKILLS_DIR = PLUGIN_DIR / "skills"
SKILL_FILE = SKILLS_DIR / "token-saver-mode" / "SKILL.md"
HOOKS_JSON = HOOKS_DIR / "hooks.json"
NUDGE_SCRIPT = HOOKS_DIR / "pre-tool-nudge.sh"
REINJECT_SCRIPT = HOOKS_DIR / "reinject-instructions.sh"


def _read_skill() -> str:
    return SKILL_FILE.read_text()


def _read_hooks() -> dict:
    return json.loads(HOOKS_JSON.read_text())


class TestSkillInjection:
    """AC: Given SKILL.md with user-invocable: false,
    When Claude Code loads skill, Then tool selection rules are injected."""

    def test_skill_is_not_user_invocable(self):
        """SKILL.md deve ter user-invocable: false."""
        content = _read_skill()
        assert "user-invocable: false" in content

    def test_skill_injects_tool_selection_rules(self):
        """Skill deve injetar regras de quando usar cada tool do plugin."""
        content = _read_skill().lower()
        # Deve mencionar as tools principais
        assert "search_semantic" in content
        assert "search_exact" in content
        assert "get_file" in content

    def test_skill_includes_anti_patterns(self):
        """Skill deve incluir anti-patterns (nao usar grep quando indice existe)."""
        content = _read_skill().lower()
        # Deve mencionar grep como anti-pattern
        assert "grep" in content
        assert "instead" in content or "prefer" in content or "avoid" in content

    def test_skill_includes_fallback_rules(self):
        """Skill deve incluir quando usar tools nativas (arquivos fora do indice, configs)."""
        content = _read_skill().lower()
        # Deve mencionar quando usar fallback para tools nativas
        assert "fallback" in content or "native" in content or "fall back" in content

    def test_skill_first_5000_tokens_preserved(self):
        """Primeiros 5000 tokens devem ser preservados durante auto-compaction.

        Proxy: o SKILL.md deve ser suficientemente conciso (<5000 tokens ~ 20KB).
        """
        content = _read_skill()
        size_bytes = len(content.encode("utf-8"))
        # 5000 tokens ~= 20KB (4 bytes/token average)
        assert size_bytes <= 20_000, (
            f"SKILL.md tem {size_bytes} bytes; deve caber em 5000 tokens (~20KB)"
        )


class TestPreToolUseGrepNudge:
    """AC: Given agent attempts Grep on project with active index,
    When PreToolUse fires, Then additionalContext suggests plugin tools."""

    def test_grep_nudge_suggests_search_semantic(self):
        """Nudge deve sugerir search_semantic ou search_exact ao inves de Grep."""
        assert NUDGE_SCRIPT.exists(), f"Script {NUDGE_SCRIPT} nao encontrado"
        content = NUDGE_SCRIPT.read_text().lower()
        assert "search_semantic" in content or "search_exact" in content

    def test_grep_nudge_does_not_block(self):
        """Hook NAO deve deny/block o Grep — apenas nudge."""
        assert NUDGE_SCRIPT.exists(), f"Script {NUDGE_SCRIPT} nao encontrado"
        content = NUDGE_SCRIPT.read_text()
        # Script não deve retornar exit 1 que bloquearia a execução
        # (exit 0 é o comportamento correto — só nudge, nunca bloqueia)
        assert "exit 1" not in content
        # Deve terminar com exit 0
        assert "exit 0" in content


class TestPreToolUseReadNudge:
    """AC: Given agent attempts Read on indexed file,
    When PreToolUse fires, Then suggests get_file."""

    def test_read_nudge_suggests_get_file(self):
        """Nudge deve sugerir get_file para leitura estruturada."""
        assert NUDGE_SCRIPT.exists(), f"Script {NUDGE_SCRIPT} nao encontrado"
        content = NUDGE_SCRIPT.read_text().lower()
        assert "get_file" in content

    def test_read_nudge_does_not_block(self):
        """Hook NAO deve bloquear Read — apenas nudge."""
        assert NUDGE_SCRIPT.exists(), f"Script {NUDGE_SCRIPT} nao encontrado"
        content = NUDGE_SCRIPT.read_text()
        # Script não deve retornar exit 1 que bloquearia a execução
        assert "exit 1" not in content
        assert "exit 0" in content


class TestNoIndexNoNudge:
    """AC: Given agent attempts Grep but NO index exists,
    When PreToolUse checks, Then no nudge is injected."""

    def test_no_nudge_without_index(self):
        """Sem indice, nenhum nudge deve ser injetado (tools nativas sao corretas)."""
        assert NUDGE_SCRIPT.exists(), f"Script {NUDGE_SCRIPT} nao encontrado"
        content = NUDGE_SCRIPT.read_text()
        # Script deve verificar existência do índice antes de emitir nudge
        # Procura por condicional que verifique index antes de dar nudge
        assert (
            "index" in content.lower()
            or "lance" in content.lower()
            or ".cts" in content.lower()
        ), "Script deve verificar se o indice existe antes de nudge"


class TestSessionStartCompact:
    """AC: Given context compaction occurs,
    When SessionStart compact matcher fires,
    Then tool preference instructions are re-injected."""

    def test_compact_reinjects_tool_preferences(self):
        """Apos compaction, instrucoes de preferencia de tools devem ser re-injetadas."""
        assert REINJECT_SCRIPT.exists(), (
            f"Script {REINJECT_SCRIPT} nao encontrado"
        )
        content = REINJECT_SCRIPT.read_text().lower()
        # Deve injetar/mencionar as tools preferidas
        assert "search_semantic" in content or "search_exact" in content or "get_file" in content
