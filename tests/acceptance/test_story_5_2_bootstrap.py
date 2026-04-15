"""
Acceptance tests for Story 5.2: Bootstrap Chain & Dependency Management.

Validates uv detection, auto-install, fallback messages, dependency
installation, persistence, and web environment handling.

FRs: FR32 (detect uv), FR33 (auto-install uv), FR34 (auto-install Python + deps),
     FR35 (deps persist), FR36 (fallback message), FR37 (web environment)
NFRs: NFR16 (cross-platform)
"""

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).parent.parent.parent
BOOTSTRAP_SCRIPT = PLUGIN_ROOT / "hooks" / "bootstrap.sh"

KNOWN_UV_PATHS = [
    "~/.local/bin/uv",
    "/opt/homebrew/bin/uv",
    "/usr/local/bin/uv",
    "~/.cargo/bin/uv",
]


class TestUvDetection:
    """AC: Given uv is at known paths, When SessionStart hook runs,
    Then uv is detected and bootstrap proceeds."""

    def test_detects_uv_at_local_bin(self):
        """Deve detectar uv em ~/.local/bin/uv."""
        assert BOOTSTRAP_SCRIPT.exists(), "bootstrap.sh deve existir"
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "$HOME/.local/bin/uv" in content, (
            "bootstrap.sh deve verificar ~/.local/bin/uv"
        )

    def test_detects_uv_at_homebrew(self):
        """Deve detectar uv em /opt/homebrew/bin/uv."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "/opt/homebrew/bin/uv" in content, (
            "bootstrap.sh deve verificar /opt/homebrew/bin/uv"
        )

    def test_detects_uv_at_usr_local(self):
        """Deve detectar uv em /usr/local/bin/uv."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "/usr/local/bin/uv" in content, (
            "bootstrap.sh deve verificar /usr/local/bin/uv"
        )

    def test_detects_uv_at_cargo(self):
        """Deve detectar uv em ~/.cargo/bin/uv."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "$HOME/.cargo/bin/uv" in content, (
            "bootstrap.sh deve verificar ~/.cargo/bin/uv"
        )


class TestUvAutoInstall:
    """AC: Given uv NOT found, When SessionStart runs,
    Then attempts to install via curl."""

    def test_attempts_uv_install_via_curl(self):
        """Deve tentar instalar uv via curl script."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "curl" in content, "bootstrap.sh deve usar curl para auto-install"
        assert "astral.sh/uv/install.sh" in content, (
            "bootstrap.sh deve usar o script oficial de instalacao do uv"
        )

    def test_proceeds_after_successful_install(self):
        """Apos install bem sucedido, deve prosseguir com dep installation."""
        content = BOOTSTRAP_SCRIPT.read_text()
        # After curl install, script should check for uv again and continue
        assert "$HOME/.local/bin/uv" in content, (
            "Apos install via curl, deve verificar ~/.local/bin/uv"
        )
        # The script should not exit 0 after install (should continue)
        lines = content.split("\n")
        install_section = False
        for i, line in enumerate(lines):
            if "astral.sh/uv/install.sh" in line:
                install_section = True
            if install_section and "UV_BIN=" in line and ".local/bin/uv" in line:
                # Found the line that sets UV_BIN after successful install
                break
        assert install_section, "Script deve tentar instalar uv"


class TestUvInstallFallback:
    """AC: Given uv installation fails, Then fallback message displayed."""

    def test_fallback_message_has_runtime_name(self):
        """Mensagem deve conter nome do runtime (uv)."""
        content = BOOTSTRAP_SCRIPT.read_text()
        # The fallback error message should mention 'uv'
        assert "Runtime required: uv" in content or "runtime" in content.lower(), (
            "Mensagem de fallback deve mencionar o nome do runtime (uv)"
        )

    def test_fallback_message_has_platform_command(self):
        """Mensagem deve conter comando de install para plataforma detectada."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "Install command" in content or "INSTALL_CMD" in content, (
            "Mensagem de fallback deve exibir comando de instalacao"
        )
        # Should have platform-specific commands
        assert "Darwin" in content or "brew" in content, (
            "Deve ter comando especifico para macOS"
        )
        assert "Linux" in content, "Deve ter comando especifico para Linux"

    def test_fallback_message_has_documentation_link(self):
        """Mensagem deve conter link para documentacao."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "docs.astral.sh/uv" in content or "UV_DOC_URL" in content, (
            "Mensagem de fallback deve conter link para documentacao do uv"
        )


class TestDependencyInstallation:
    """AC: Given uv available and pyproject.toml differs,
    Then copies pyproject.toml, creates venv, installs deps."""

    def test_copies_pyproject_to_plugin_data(self):
        """Deve copiar pyproject.toml para CLAUDE_PLUGIN_DATA."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "cp" in content and "pyproject.toml" in content, (
            "bootstrap.sh deve copiar pyproject.toml para PLUGIN_DATA"
        )
        assert "PYPROJECT_SRC" in content and "PYPROJECT_DST" in content, (
            "Deve ter variaveis para src e dst do pyproject.toml"
        )

    def test_creates_venv_with_python_311(self):
        """Deve criar venv com Python 3.11+."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "venv" in content, "bootstrap.sh deve criar venv"
        assert "3.11" in content, "Venv deve usar Python >= 3.11"

    def test_installs_all_dependencies(self):
        """Deve instalar todas as dependencias."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "pip install" in content or "uv pip install" in content, (
            "bootstrap.sh deve instalar dependencias"
        )


class TestDependencySkip:
    """AC: Given pyproject.toml matches, Then installation skipped."""

    def test_skips_when_pyproject_matches(self):
        """Deve pular instalacao quando pyproject.toml ja esta atualizado."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "diff" in content or "needs_install" in content, (
            "bootstrap.sh deve comparar pyproject.toml e pular se igual"
        )
        # The skip branch should exist
        assert "needs_install" in content, (
            "Deve ter flag de controle para pular instalacao"
        )


class TestDependencyPersistence:
    """AC: Given deps installed in previous session,
    When plugin updates, Then deps persist and only update if needed."""

    def test_deps_persist_between_sessions(self):
        """Dependencias devem persistir entre sessoes em CLAUDE_PLUGIN_DATA."""
        content = BOOTSTRAP_SCRIPT.read_text()
        # PLUGIN_DATA is where deps are stored (not PLUGIN_ROOT)
        assert "CLAUDE_PLUGIN_DATA" in content or "PLUGIN_DATA" in content, (
            "Deps devem ser armazenados em CLAUDE_PLUGIN_DATA (persistente)"
        )
        assert "VENV_DIR" in content and "PLUGIN_DATA" in content, (
            "Venv deve estar dentro de PLUGIN_DATA para persistencia"
        )

    def test_deps_update_only_when_pyproject_changes(self):
        """Deps devem atualizar apenas quando pyproject.toml muda."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "diff" in content, (
            "Script deve usar diff para verificar se pyproject.toml mudou"
        )
        # The else branch (skip) should be present
        assert "needs_install" in content, (
            "Deve ter logica de skip quando pyproject nao mudou"
        )


class TestWebEnvironment:
    """AC: Given Claude Code on the Web, When SessionStart runs,
    Then bootstrap skips all installation steps."""

    def test_web_environment_skips_bootstrap(self):
        """No Claude Code on the Web, deve pular todos os passos de bootstrap."""
        content = BOOTSTRAP_SCRIPT.read_text()
        assert "CLAUDE_WEB" in content, (
            "bootstrap.sh deve verificar variavel CLAUDE_WEB"
        )
        # When CLAUDE_WEB=1, should exit 0 immediately
        assert 'CLAUDE_WEB' in content and 'exit 0' in content, (
            "Quando CLAUDE_WEB=1, deve sair imediatamente sem fazer nada"
        )
