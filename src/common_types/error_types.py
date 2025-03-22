"""Common error types."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-related errors."""

    def __contains__(self, item: str) -> bool:
        """Implement the 'in' operator for string containment tests.

        Args:
            item: The string to check for containment.

        Returns:
            True if the string is contained in the error message, False otherwise.

        """
        return item in str(self)


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


class AgentCreationError(AgentError):
    """Error raised when agent creation fails."""


class AgentCommunicationError(AgentError):
    """Error raised when communication with an agent fails."""


class AgentExecutionError(AgentError):
    """Error raised when agent execution fails."""


class ConfigError(Exception):
    """Raised when there is a configuration error."""


class RetryError(Exception):
    """Raised when maximum retries are exceeded."""


class ProviderError(Exception):
    """Raised when there is a provider error."""


# Additional exceptions
class APIKeyError(ValueError):
    """Raised when API key is missing or invalid."""


class EmptyResponseError(RuntimeError):
    """Raised when response is empty."""


class InvalidModelError(ValueError):
    """Raised when model name is invalid."""


class TemperatureError(ValueError):
    """Raised when temperature is out of range."""


class ValidationError(Exception):
    """Raised when validation fails."""


class ProcessingError(Exception):
    """Raised when message processing fails."""


class RoutingError(Exception):
    """Raised when message routing fails."""
