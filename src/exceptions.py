"""Custom exceptions for the application."""


class APIKeyError(ValueError):
    """Raised when API key is missing or invalid."""


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


class EmptyResponseError(RuntimeError):
    """Raised when response is empty."""


class InvalidModelError(ValueError):
    """Raised when model name is invalid."""


class RetryError(RuntimeError):
    """Raised when max retries exceeded."""


class TemperatureError(ValueError):
    """Raised when temperature is out of range."""
