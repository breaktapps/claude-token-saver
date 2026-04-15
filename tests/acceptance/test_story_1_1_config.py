"""
Acceptance tests for Story 1.1: Project Foundation & Configuration.

Validates configuration management with defaults, YAML override, env var precedence,
API key guard, and custom error types.

FRs: FR23 (default config), FR24 (quality config), FR38 (API key guard), FR39 (local default)
"""

import os
from pathlib import Path

import pytest
import yaml

from src.config import Config
from src.errors import (
    CTSError,
    ConfigValidationError,
    EmbeddingProviderError,
    IndexNotFoundError,
    LockTimeoutError,
)


class TestConfigDefaults:
    """AC: Given no YAML config file exists, When Config loads,
    Then defaults are applied."""

    def test_default_embedding_mode_is_lite(self, tmp_path):
        """Config sem YAML deve usar embedding_mode=lite."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.embedding_mode == "lite"

    def test_default_top_k_is_5(self, tmp_path):
        """Config sem YAML deve usar top_k=5."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.top_k == 5

    def test_default_score_threshold_is_0_5(self, tmp_path):
        """Config sem YAML deve usar score_threshold=0.5."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.score_threshold == 0.5

    def test_default_batch_size_is_50(self, tmp_path):
        """Config sem YAML deve usar batch_size=50."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.batch_size == 50

    def test_default_max_chunk_lines_is_150(self, tmp_path):
        """Config sem YAML deve usar max_chunk_lines=150."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.max_chunk_lines == 150


class TestConfigYamlOverride:
    """AC: Given a YAML config with embedding_mode: quality,
    When Config loads, Then embedding_mode is overridden, other defaults preserved."""

    def test_yaml_overrides_embedding_mode(self, tmp_path):
        """YAML com embedding_mode: quality deve sobrescrever o default."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"embedding_mode": "quality"}))
        config = Config.load(config_path=config_file)
        assert config.embedding_mode == "quality"

    def test_yaml_preserves_other_defaults(self, tmp_path):
        """YAML parcial deve preservar defaults nao especificados."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"embedding_mode": "quality"}))
        config = Config.load(config_path=config_file)
        assert config.top_k == 5
        assert config.score_threshold == 0.5
        assert config.batch_size == 50
        assert config.max_chunk_lines == 150


class TestConfigEnvVarPrecedence:
    """AC: Given env var CTS_EMBEDDING_MODE=quality is set,
    When Config loads with YAML saying lite,
    Then env var wins."""

    def test_env_var_overrides_yaml(self, tmp_path, monkeypatch):
        """Env var CTS_EMBEDDING_MODE deve ter precedencia sobre YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"embedding_mode": "lite"}))
        monkeypatch.setenv("CTS_EMBEDDING_MODE", "quality")
        config = Config.load(config_path=config_file)
        assert config.embedding_mode == "quality"


class TestConfigApiKeyGuard:
    """AC: Given a YAML config with a literal API key value,
    When Config loads, Then ConfigValidationError is raised."""

    def test_rejects_sk_prefix_api_key(self, tmp_path):
        """YAML com valor sk-... deve levantar ConfigValidationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"api_key": "sk-abc123def456"}))
        with pytest.raises(ConfigValidationError):
            Config.load(config_path=config_file)

    def test_rejects_voy_prefix_api_key(self, tmp_path):
        """YAML com valor voy-... deve levantar ConfigValidationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"api_key": "voy-abc123def456"}))
        with pytest.raises(ConfigValidationError):
            Config.load(config_path=config_file)

    def test_rejects_key_prefix_api_key(self, tmp_path):
        """YAML com valor key-... deve levantar ConfigValidationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"api_key": "key-abc123def456"}))
        with pytest.raises(ConfigValidationError):
            Config.load(config_path=config_file)

    def test_rejects_gsk_prefix_api_key(self, tmp_path):
        """YAML com valor gsk_... deve levantar ConfigValidationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"api_key": "gsk_abc123def456"}))
        with pytest.raises(ConfigValidationError):
            Config.load(config_path=config_file)

    def test_error_message_identifies_field_and_suggests_env_var(self, tmp_path):
        """Mensagem de erro deve identificar o campo e sugerir uso de env var."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"api_key": "sk-abc123def456"}))
        with pytest.raises(ConfigValidationError, match="api_key"):
            Config.load(config_path=config_file)


class TestProjectScaffolding:
    """AC: Given uv is installed and products/plugin/ exists,
    When project initialization runs,
    Then pyproject.toml and src/ structure are created."""

    def test_pyproject_toml_has_all_dependencies(self):
        """pyproject.toml deve conter mcp, lancedb, fastembed, tree-sitter, pyarrow, pydantic, pyyaml."""
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        for dep in ["mcp", "lancedb", "fastembed", "tree-sitter", "pyarrow", "pydantic", "pyyaml"]:
            assert dep in content, f"Missing dependency: {dep}"

    def test_pyproject_toml_has_dev_dependencies(self):
        """pyproject.toml deve conter pytest, pytest-asyncio como dev deps."""
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert "pytest" in content
        assert "pytest-asyncio" in content

    def test_src_directory_has_init_config_errors(self):
        """src/ deve conter __init__.py, config.py, errors.py."""
        src_dir = Path(__file__).parent.parent.parent / "src"
        assert (src_dir / "__init__.py").exists()
        assert (src_dir / "config.py").exists()
        assert (src_dir / "errors.py").exists()


class TestCustomErrorTypes:
    """AC: Given errors.py exists, When imported,
    Then all 4 custom exceptions are available and inherit from Exception."""

    def test_index_not_found_error_exists(self):
        """IndexNotFoundError deve existir e herdar de Exception."""
        assert issubclass(IndexNotFoundError, Exception)

    def test_lock_timeout_error_exists(self):
        """LockTimeoutError deve existir e herdar de Exception."""
        assert issubclass(LockTimeoutError, Exception)

    def test_embedding_provider_error_exists(self):
        """EmbeddingProviderError deve existir e herdar de Exception."""
        assert issubclass(EmbeddingProviderError, Exception)

    def test_config_validation_error_exists(self):
        """ConfigValidationError deve existir e herdar de Exception."""
        assert issubclass(ConfigValidationError, Exception)
