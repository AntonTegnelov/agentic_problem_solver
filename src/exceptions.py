"""Custom exceptions for the application."""


class APIKeyError(ValueError):
    """Raised when API key is missing or invalid."""


class ConfigError(Exception):
    """Raised when configuration is invalid."""


class EmptyResponseError(RuntimeError):
    """Raised when response is empty."""


class InvalidModelError(ValueError):
    """Raised when model name is invalid."""


class RetryError(Exception):
    """Raised when retry attempts are exhausted."""


class TemperatureError(ValueError):
    """Raised when temperature is out of range."""


class AgentError(Exception):
    """Base class for agent-related errors."""


class AgentNotFoundError(AgentError):
    """Raised when agent is not found."""


class ValidationError(Exception):
    """Raised when validation fails."""


class ProcessingError(Exception):
    """Raised when message processing fails."""
