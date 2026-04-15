"""Unit tests for errors module."""

from src.errors import (
    CTSError,
    ConfigValidationError,
    EmbeddingProviderError,
    IndexNotFoundError,
    LockTimeoutError,
)


class TestErrorHierarchy:
    """Verify error class hierarchy and messages."""

    def test_all_inherit_from_cts_error(self):
        """All custom exceptions should inherit from CTSError."""
        for exc_class in [IndexNotFoundError, LockTimeoutError,
                          EmbeddingProviderError, ConfigValidationError]:
            assert issubclass(exc_class, CTSError)

    def test_cts_error_inherits_from_exception(self):
        """CTSError should inherit from Exception."""
        assert issubclass(CTSError, Exception)

    def test_exceptions_carry_message(self):
        """Exceptions should carry custom messages."""
        msg = "Test error message"
        for exc_class in [IndexNotFoundError, LockTimeoutError,
                          EmbeddingProviderError, ConfigValidationError]:
            exc = exc_class(msg)
            assert str(exc) == msg

    def test_exceptions_are_catchable_as_exception(self):
        """All custom exceptions should be catchable as Exception."""
        for exc_class in [IndexNotFoundError, LockTimeoutError,
                          EmbeddingProviderError, ConfigValidationError]:
            try:
                raise exc_class("test")
            except Exception as e:
                assert isinstance(e, exc_class)
