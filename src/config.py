"""Configuration management for claude-token-saver.

Load chain: defaults (hardcoded) -> YAML (~/.claude-token-saver/config.yaml) -> env vars (CTS_*)
Config is immutable after load.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .errors import ConfigValidationError

logger = logging.getLogger("cts.config")

# Defaults
DEFAULT_EMBEDDING_MODE = "lite"
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_CHUNK_LINES = 150

# API key patterns that must never appear as literal values in YAML
# Covers: OpenAI (sk-), Anthropic (sk-ant-), Voyage (voy-), generic (key-),
#         Google/Gemini (gsk_), Cohere (co-)
_API_KEY_PATTERNS = re.compile(r"^(sk-|sk-ant-|voy-|key-|gsk_|co-)")

# Default config path
CONFIG_PATH = Path.home() / ".claude-token-saver" / "config.yaml"

# Mapping of Config fields to CTS_ env var names
_ENV_VAR_MAP = {
    "embedding_mode": "CTS_EMBEDDING_MODE",
    "embedding_provider": "CTS_EMBEDDING_PROVIDER",
    "top_k": "CTS_TOP_K",
    "score_threshold": "CTS_SCORE_THRESHOLD",
    "batch_size": "CTS_BATCH_SIZE",
    "max_chunk_lines": "CTS_MAX_CHUNK_LINES",
}


def _check_api_key_literals(data: dict) -> None:
    """Raise ConfigValidationError if any YAML value looks like a literal API key."""
    for key, value in data.items():
        if isinstance(value, str) and _API_KEY_PATTERNS.match(value):
            raise ConfigValidationError(
                f"API key detected in config field '{key}'. "
                f"Use environment variable instead (e.g., export {key.upper()}=your-key)"
            )
        if isinstance(value, dict):
            _check_api_key_literals(value)


def _load_yaml(config_path: Path) -> dict:
    """Load YAML config file, returning empty dict if not found."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _flatten_yaml(data: dict) -> dict:
    """Extract known config fields from potentially nested YAML structure."""
    result = {}
    # Check top-level keys directly
    for field in _ENV_VAR_MAP:
        if field in data:
            result[field] = data[field]
    # Check nested sections (embedding.mode -> embedding_mode, etc.)
    if "embedding" in data and isinstance(data["embedding"], dict):
        emb = data["embedding"]
        if "mode" in emb:
            result["embedding_mode"] = emb["mode"]
        if "provider" in emb:
            result["embedding_provider"] = emb["provider"]
    if "search" in data and isinstance(data["search"], dict):
        search = data["search"]
        if "default_top_k" in search:
            result["top_k"] = search["default_top_k"]
        if "score_threshold" in search:
            result["score_threshold"] = search["score_threshold"]
        if "query_cache_size" in search:
            pass  # Not a Config field in v1
    if "indexing" in data and isinstance(data["indexing"], dict):
        idx = data["indexing"]
        if "batch_size" in idx:
            result["batch_size"] = idx["batch_size"]
        if "extra_ignore" in idx:
            result["extra_ignore"] = idx["extra_ignore"]
    if "chunking" in data and isinstance(data["chunking"], dict):
        chunk = data["chunking"]
        if "max_chunk_lines" in chunk:
            result["max_chunk_lines"] = chunk["max_chunk_lines"]
    return result


def _apply_env_vars(overrides: dict) -> dict:
    """Apply CTS_ environment variable overrides."""
    result = dict(overrides)
    for field, env_var in _ENV_VAR_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Convert to appropriate type
            if field in ("top_k", "batch_size", "max_chunk_lines"):
                result[field] = int(value)
            elif field == "score_threshold":
                result[field] = float(value)
            else:
                result[field] = value
    return result


class Config(BaseModel):
    """Immutable configuration for claude-token-saver."""

    model_config = ConfigDict(frozen=True)

    embedding_mode: str = DEFAULT_EMBEDDING_MODE
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    top_k: int = DEFAULT_TOP_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES
    extra_ignore: list[str] = []

    @classmethod
    def load(cls, config_path: Path | None = None) -> Config:
        """Load config with chain: defaults -> YAML -> env vars.

        Each layer overrides the previous one.
        """
        path = config_path or CONFIG_PATH

        # Load and validate YAML
        yaml_data = _load_yaml(path)
        _check_api_key_literals(yaml_data)

        # Flatten nested YAML into flat config fields
        yaml_overrides = _flatten_yaml(yaml_data)

        # Apply env var overrides on top
        final_overrides = _apply_env_vars(yaml_overrides)

        cfg = cls(**final_overrides)
        logger.info(
            "Config loaded: embedding_mode=%s, embedding_provider=%s, top_k=%d",
            cfg.embedding_mode,
            cfg.embedding_provider,
            cfg.top_k,
        )
        return cfg
