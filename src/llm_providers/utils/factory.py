"""LLM provider factory utilities."""

import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.providers.gemini import GeminiProvider

# Error messages
API_KEY_ERROR = "GEMINI_API_KEY not found in .env file"
PROVIDER_NOT_SET_ERROR = "No provider set"
UNSUPPORTED_PROVIDER_ERROR = "Unsupported provider: {}"
PROVIDER_NOT_FOUND = "Provider {name} not found"
API_KEY_REQUIRED = "API key is required"


class ProviderNotFoundError(ValueError):
    """Raised when provider is not found."""

    def __init__(self, name: str) -> None:
        """Initialize error.

        Args:
            name: Provider name.

        """
        super().__init__(PROVIDER_NOT_FOUND.format(name=name))


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _instance: ClassVar[type["LLMProviderFactory"] | None] = None
    _initialized: ClassVar[bool] = False
    _providers: ClassVar[dict[str, type[BaseLLMProvider]]] = {}

    def __new__(cls) -> "LLMProviderFactory":
        """Create or return singleton instance.

        Returns:
            The singleton instance.

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize factory."""
        if not self._initialized:
            # Load environment variables from .env file
            env_path = Path(".env")
            if not env_path.exists():
                raise ValueError(API_KEY_ERROR)
            load_dotenv(env_path)

            # Get API key from environment variables
            self._api_key = os.getenv("GEMINI_API_KEY")
            if not self._api_key:
                raise ValueError(API_KEY_ERROR)

            self._current_provider: BaseLLMProvider | None = None
            self._provider_name: str | None = None

            # Set default provider
            self.set_provider("gemini")
            self._initialized = True

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[BaseLLMProvider]) -> None:
        """Register a provider.

        Args:
            name: Provider name.
            provider_cls: Provider class.

        """
        cls._providers[name] = provider_cls

    @classmethod
    def create_provider(cls, name: str, api_key: str) -> BaseLLMProvider:
        """Create provider instance."""
        if not api_key:
            raise ValueError(API_KEY_REQUIRED)

        provider_class = cls._providers.get(name)
        if not provider_class:
            raise ValueError(PROVIDER_NOT_FOUND.format(name=name))

        return provider_class(api_key)

    def set_provider(self, name: str) -> None:
        """Set the active provider."""
        if name not in self._providers:
            error_msg = UNSUPPORTED_PROVIDER_ERROR.format(name)
            raise ValueError(error_msg)
        self._provider_name = name
        self._current_provider = self._providers[name](self._api_key)

    def get_provider(self) -> BaseLLMProvider:
        """Get the current provider instance."""
        if not self._current_provider:
            raise ValueError(PROVIDER_NOT_SET_ERROR)
        return self._current_provider

    def get_provider_name(self) -> str | None:
        """Get the current provider name.

        Returns:
            Provider name or None if no provider is set.

        """
        return self._provider_name

    @classmethod
    def get_provider_class(cls, name: str) -> type[BaseLLMProvider]:
        """Get a provider class.

        Args:
            name: Provider name.

        Returns:
            Provider class.

        Raises:
            ProviderNotFoundError: If provider not found.

        """
        if name not in cls._providers:
            raise ProviderNotFoundError(name)
        return cls._providers[name]


# Register default providers
LLMProviderFactory.register_provider("gemini", GeminiProvider)
