"""Factory for creating LLM providers."""

import os

from src.llm_providers.base import BaseLLMProvider
from src.llm_providers.gemini import GeminiProvider


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _instance = None
    _initialized = False

    def __new__(cls, api_key: str | None = None) -> "LLMProviderFactory":
        """Create a new factory instance or return existing one."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the factory."""
        if not self._initialized:
            self._api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self._providers: dict[str, type[BaseLLMProvider]] = {}
            self._current_provider: BaseLLMProvider | None = None
            self._provider_name: str | None = None

            # Register and set Gemini as the default provider
            self.register_provider("gemini", GeminiProvider)
            self.set_active_provider("gemini")

            self._initialized = True

    def register_provider(
        self, name: str, provider_class: type[BaseLLMProvider]
    ) -> None:
        """Register a new provider."""
        self._providers[name] = provider_class

    def set_active_provider(self, name: str) -> None:
        """Set the active provider."""
        if name not in self._providers:
            raise ValueError(f"Unsupported provider: {name}")
        self._provider_name = name
        self._current_provider = self._providers[name](self._api_key)

    def get_provider(self) -> BaseLLMProvider:
        """Get the current provider instance."""
        if not self._current_provider:
            raise ValueError("No provider set")
        return self._current_provider

    def set_api_key(self, api_key: str) -> None:
        """Set the API key."""
        self._api_key = api_key
        if self._provider_name:
            self._current_provider = self._providers[self._provider_name](api_key)
