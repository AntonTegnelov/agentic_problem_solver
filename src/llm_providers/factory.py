from typing import Dict, Type, Optional
import os
from .base import LLMProvider
from .gemini import GeminiProvider
from src.config import DEFAULT_LLM_CONFIG


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers: Dict[str, Type[LLMProvider]] = {
        "gemini": GeminiProvider,
    }

    _instance = None
    _current_provider = None
    _provider = None
    _api_key = None
    _model = None
    _current_provider_name = "gemini"  # Track current provider name

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the factory."""
        if not hasattr(self, "initialized"):
            self.initialized = True
            self._load_default_provider()

    def _load_default_provider(self) -> None:
        """Load the default provider from environment variables."""
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()

        if provider_name not in self._providers:
            raise ValueError(f"Unsupported provider: {provider_name}")

        provider_class = self._providers[provider_name]
        self._current_provider_name = provider_name

        if provider_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self._current_provider = provider_class(api_key=api_key)

    def get_provider(self, model: Optional[str] = None) -> LLMProvider:
        """Get or create an LLM provider instance.

        Args:
            model: Optional model name to use. If not provided, uses the default.

        Returns:
            An LLM provider instance

        Raises:
            ValueError: If no API key is found
        """
        api_key = self._get_api_key()

        # Use default model if none specified
        model = model or DEFAULT_LLM_CONFIG["model"]

        # Create new provider if model changed
        if self._provider is None or model != self._model:
            self._provider = self._create_provider(api_key, model)
            self._model = model

        return self._provider

    def _create_provider(
        self, api_key: str, model: Optional[str] = None
    ) -> LLMProvider:
        """Create a new LLM provider instance.

        Args:
            api_key: API key to use
            model: Optional model name to use. If not provided, uses the default.

        Returns:
            A new LLM provider instance
        """
        # Use default model if none specified
        model = model or DEFAULT_LLM_CONFIG["model"]

        # Use current provider class
        provider_class = self._providers[self._current_provider_name]
        return provider_class(api_key, model)

    def set_provider(self, provider_name: str, **kwargs) -> None:
        """Set a new LLM provider.

        Args:
            provider_name: Name of the provider to use
            **kwargs: Provider-specific configuration

        Raises:
            ValueError: If provider_name is not supported
        """
        provider_name = provider_name.lower()
        if provider_name not in self._providers:
            raise ValueError(f"Unsupported provider: {provider_name}")

        provider_class = self._providers[provider_name]

        # Update API key if provided
        if "api_key" in kwargs:
            self._api_key = kwargs["api_key"]
        # If no API key in kwargs, use the current one
        elif self._api_key:
            kwargs["api_key"] = self._api_key

        # Reset provider state
        self._provider = provider_class(**kwargs)
        self._model = None
        self._current_provider = self._provider
        self._current_provider_name = provider_name

    def register_provider(self, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a new provider type.

        Args:
            name: Name to register the provider under
            provider_class: Provider class to register
        """
        self._providers[name.lower()] = provider_class

    @property
    def available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return list(self._providers.keys())

    def _get_api_key(self) -> str:
        """Get the API key from environment variables.

        Returns:
            API key

        Raises:
            ValueError: If no API key is found
        """
        if self._api_key is None:
            self._api_key = os.environ.get("GEMINI_API_KEY")
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
        return self._api_key

    def _set_api_key(self, api_key: str) -> None:
        """Set the API key.

        Args:
            api_key: API key to use
        """
        self._api_key = api_key
