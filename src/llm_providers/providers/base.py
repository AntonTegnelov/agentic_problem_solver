"""Base provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

from src.llm_providers.config.provider_config import ProviderConfig


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.

        """
        ...

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate text from prompt as a stream.

        Args:
            prompt: Input prompt.

        Yields:
            Generated text chunks.

        """
        ...

    def update_config(self, config: dict[str, str]) -> None:
        """Update provider configuration.

        Args:
            config: Configuration dictionary.

        """
        ...

    def get_config(self) -> dict[str, str]:
        """Get current configuration.

        Returns:
            Configuration dictionary.

        """
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize provider.

        Args:
            api_key: Optional API key.

        """
        self.config = self._create_config(api_key)
        self._validate_config()

    @abstractmethod
    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration.

        Args:
            api_key: Optional API key.

        Returns:
            Provider configuration.

        """

    def _validate_config(self) -> None:
        """Validate provider configuration.

        Raises:
            ValueError: If configuration is invalid.

        """
        self.config.validate()

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.

        """

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate text from prompt as a stream.

        Args:
            prompt: Input prompt.

        Yields:
            Generated text chunks.

        """

    def update_config(self, config: dict[str, str]) -> None:
        """Update provider configuration.

        Args:
            config: Configuration dictionary.

        """
        self.config.update(config)
        self._validate_config()

    def get_config(self) -> dict[str, str]:
        """Get current configuration.

        Returns:
            Configuration dictionary.

        """
        return self.config.to_dict()
