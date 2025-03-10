"""LLM provider factory."""

from typing import ClassVar

from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.providers.gemini import GeminiProvider

# Error messages
PROVIDER_NOT_FOUND = "Provider {name} not found"
API_KEY_REQUIRED = "API key is required"
NO_PROVIDER_SET = "No provider set"
UNSUPPORTED_PROVIDER = "Unsupported provider: {}"


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
    _current_provider: ClassVar[BaseLLMProvider | None] = None
    _provider_name: ClassVar[str | None] = None

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
    def get_provider(cls, name: str) -> type[BaseLLMProvider]:
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

    @classmethod
    def get_current_provider(cls) -> BaseLLMProvider | None:
        """Get current provider instance.

        Returns:
            Current provider instance or None.

        """
        return cls._current_provider

    @classmethod
    def get_current_provider_name(cls) -> str | None:
        """Get current provider name.

        Returns:
            Current provider name or None.

        """
        return cls._provider_name

    def set_provider(self, name: str) -> None:
        """Set the active provider."""
        if name not in self._providers:
            raise ValueError(UNSUPPORTED_PROVIDER.format(name))
        self._provider_name = name
        self._current_provider = self._providers[name](self._api_key)

    def get_provider_name(self) -> str | None:
        """Get the current provider name."""
        return self._provider_name

    @classmethod
    def create_provider(cls, name: str, api_key: str) -> BaseLLMProvider:
        """Create provider instance."""
        if not api_key:
            raise ValueError(API_KEY_REQUIRED)

        provider_class = cls._providers.get(name)
        if not provider_class:
            raise ValueError(PROVIDER_NOT_FOUND.format(name=name))

        return provider_class(api_key)


# Register default providers
LLMProviderFactory.register_provider("gemini", GeminiProvider)
