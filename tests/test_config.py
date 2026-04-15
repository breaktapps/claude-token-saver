"""Unit tests for config module -- edge cases."""

import pytest
import yaml

from src.config import Config
from src.errors import ConfigValidationError


class TestConfigEdgeCases:
    """Edge cases for Config loading."""

    def test_malformed_yaml_returns_defaults(self, tmp_path):
        """YAML malformado (nao-dict) deve retornar defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("just a string")
        config = Config.load(config_path=config_file)
        assert config.embedding_mode == "lite"

    def test_empty_yaml_returns_defaults(self, tmp_path):
        """YAML vazio deve retornar defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        config = Config.load(config_path=config_file)
        assert config.embedding_mode == "lite"

    def test_config_is_frozen(self, tmp_path):
        """Config deve ser imutavel apos load."""
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        with pytest.raises(Exception):
            config.embedding_mode = "quality"

    def test_nested_yaml_embedding_section(self, tmp_path):
        """YAML com secao embedding.mode aninhada deve funcionar."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"embedding": {"mode": "quality", "provider": "voyage"}}))
        config = Config.load(config_path=config_file)
        assert config.embedding_mode == "quality"
        assert config.embedding_provider == "voyage"

    def test_api_key_in_nested_dict(self, tmp_path):
        """API key em dict aninhado deve ser detectada."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"embedding": {"api_key": "sk-secret123"}}))
        with pytest.raises(ConfigValidationError):
            Config.load(config_path=config_file)

    def test_env_var_int_conversion(self, tmp_path, monkeypatch):
        """Env var CTS_TOP_K deve converter para int."""
        monkeypatch.setenv("CTS_TOP_K", "10")
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.top_k == 10

    def test_env_var_float_conversion(self, tmp_path, monkeypatch):
        """Env var CTS_SCORE_THRESHOLD deve converter para float."""
        monkeypatch.setenv("CTS_SCORE_THRESHOLD", "0.8")
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.score_threshold == 0.8


class TestConfigLoadNoEnvVarLeak:
    """Ensure env vars from other tests dont leak."""

    def test_defaults_without_env_vars(self, tmp_path, monkeypatch):
        """Garantir que sem env vars CTS_, defaults sao usados."""
        for var in ["CTS_EMBEDDING_MODE", "CTS_TOP_K", "CTS_SCORE_THRESHOLD",
                     "CTS_BATCH_SIZE", "CTS_MAX_CHUNK_LINES", "CTS_EMBEDDING_PROVIDER"]:
            monkeypatch.delenv(var, raising=False)
        config = Config.load(config_path=tmp_path / "nonexistent.yaml")
        assert config.embedding_mode == "lite"
        assert config.top_k == 5
