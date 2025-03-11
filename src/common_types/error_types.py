"""Common error types."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-related errors."""


class AgentNotFoundError(AgentError):
    """Error raised when an agent is not found."""


class AgentNotReadyError(AgentError):
    """Error raised when an agent is not ready."""


class AgentTimeoutError(AgentError):
    """Error raised when an agent operation times out."""


class AgentConfigError(AgentError):
    """Error raised when there is an agent configuration error."""


class AgentStateError(AgentError):
    """Error raised when there is an agent state error."""


class AgentProcessingError(AgentError):
    """Error raised when there is an error processing a message."""


class AgentValidationError(AgentError):
    """Error raised when there is a validation error."""


class AgentAuthenticationError(AgentError):
    """Error raised when there is an authentication error."""


class AgentAuthorizationError(AgentError):
    """Error raised when there is an authorization error."""


class ConfigError(Exception):
    """Raised when there is a configuration error."""


class RetryError(Exception):
    """Raised when maximum retries are exceeded."""


class ProviderError(Exception):
    """Raised when there is a provider error."""
