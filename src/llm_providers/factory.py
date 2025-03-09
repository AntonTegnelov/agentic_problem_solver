from typing import Dict, Type
import os
from .base import LLMProvider
from .gemini import GeminiProvider

class LLMProviderFactory:
    """Factory class for creating and managing LLM providers."""
    
    _providers: Dict[str, Type[LLMProvider]] = {
        "gemini": GeminiProvider,
    }
    
    _instance = None
    _current_provider = None
    
    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the factory."""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._load_default_provider()
    
    def _load_default_provider(self) -> None:
        """Load the default provider from environment variables."""
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if provider_name not in self._providers:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        provider_class = self._providers[provider_name]
        
        if provider_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self._current_provider = provider_class(api_key=api_key)
    
    def get_provider(self) -> LLMProvider:
        """Get the current LLM provider.
        
        Returns:
            Current LLM provider instance
        """
        if self._current_provider is None:
            self._load_default_provider()
        return self._current_provider
    
    def set_provider(self, provider_name: str, **kwargs) -> None:
        """Set a new LLM provider.
        
        Args:
            provider_name: Name of the provider to use
            **kwargs: Provider-specific configuration
        """
        if provider_name not in self._providers:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        provider_class = self._providers[provider_name]
        self._current_provider = provider_class(**kwargs)
    
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