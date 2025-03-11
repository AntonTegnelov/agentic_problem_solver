"""Tests for common error types."""

from src.common_types.error_types import (
    AgentAuthenticationError,
    AgentAuthorizationError,
    AgentConfigError,
    AgentError,
    AgentNotFoundError,
    AgentNotReadyError,
    AgentProcessingError,
    AgentStateError,
    AgentTimeoutError,
    AgentValidationError,
    ConfigError,
    ProviderError,
    RetryError,
)


def test_agent_error_hierarchy() -> None:
    """Test that all agent errors inherit from AgentError."""
    assert issubclass(AgentNotFoundError, AgentError)
    assert issubclass(AgentNotReadyError, AgentError)
    assert issubclass(AgentTimeoutError, AgentError)
    assert issubclass(AgentConfigError, AgentError)
    assert issubclass(AgentStateError, AgentError)
    assert issubclass(AgentProcessingError, AgentError)
    assert issubclass(AgentValidationError, AgentError)
    assert issubclass(AgentAuthenticationError, AgentError)
    assert issubclass(AgentAuthorizationError, AgentError)


def test_error_instantiation() -> None:
    """Test that all errors can be instantiated with a message."""
    # Test AgentError
    error = AgentError("Test error message")
    assert str(error) == "Test error message"

    # Test specific agent errors
    error = AgentNotFoundError("Agent not found")
    assert str(error) == "Agent not found"
    assert isinstance(error, AgentError)

    error = AgentNotReadyError("Agent not ready")
    assert str(error) == "Agent not ready"
    assert isinstance(error, AgentError)

    error = AgentTimeoutError("Agent timed out")
    assert str(error) == "Agent timed out"
    assert isinstance(error, AgentError)

    # Test other error types
    error = ConfigError("Configuration error")
    assert str(error) == "Configuration error"
    assert isinstance(error, Exception)

    error = RetryError("Maximum retries exceeded")
    assert str(error) == "Maximum retries exceeded"
    assert isinstance(error, Exception)

    error = ProviderError("Provider error")
    assert str(error) == "Provider error"
    assert isinstance(error, Exception)


def test_error_chaining() -> None:
    """Test error chaining using from_exception."""
    # Create a cause exception
    cause = ValueError("Original error")

    # Test with AgentError
    error = AgentError("Agent error with cause")
    error.__cause__ = cause

    # Check that the cause is properly set
    assert isinstance(error.__cause__, ValueError)
    assert str(error.__cause__) == "Original error"

    # Same for ConfigError
    error = ConfigError("Config error with cause")
    error.__cause__ = cause

    # Check that the cause is properly set
    assert isinstance(error.__cause__, ValueError)
    assert str(error.__cause__) == "Original error"
