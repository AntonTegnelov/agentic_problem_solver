"""LLM provider configuration error classes."""

from src.exceptions import ConfigError


class ProviderConfigError(ConfigError):
    """Raised when provider configuration is invalid."""


class ProviderNotFoundError(ConfigError):
    """Raised when provider is not found."""


class ProviderInitializationError(ConfigError):
    """Raised when provider initialization fails."""


class ModelConfigError(ConfigError):
    """Raised when model configuration is invalid."""


class ModelNotFoundError(ConfigError):
    """Raised when model is not found."""
