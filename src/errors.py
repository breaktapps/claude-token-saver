"""Custom exceptions for claude-token-saver."""


class CTSError(Exception):
    """Base exception for all claude-token-saver errors."""


class IndexNotFoundError(CTSError):
    """Raised when the index does not exist for a repository."""


class LockTimeoutError(CTSError):
    """Raised when the file lock cannot be acquired within the timeout."""


class EmbeddingProviderError(CTSError):
    """Raised when the embedding provider fails (offline, model not found, etc.)."""


class ConfigValidationError(CTSError):
    """Raised when configuration is invalid (bad YAML, literal API key detected, etc.)."""
