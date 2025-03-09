"""Base LLM provider implementation."""

from collections.abc import AsyncGenerator
from typing import Any, Protocol

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(Protocol):
    """Base LLM provider protocol."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        """Initialize provider.

        Args:
            api_key: Provider API key.
            model: Model name to use.
        """
        ...

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            ValueError: If prompt is empty or parameters are invalid.
            RuntimeError: If generation fails.
        """
        ...

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text from prompt with streaming.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Yields:
            Generated text chunks.

        Raises:
            ValueError: If prompt is empty or parameters are invalid.
            RuntimeError: If generation fails.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Token count.
        """
        ...

    def get_config(self) -> dict[str, Any]:
        """Get current configuration.

        Returns:
            Current configuration.
        """
        ...

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.
        """
        ...
