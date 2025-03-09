from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncGenerator


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Controls randomness in generation (0.0 to 1.0)
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate text from prompt with streaming.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Controls randomness in generation (0.0 to 1.0)
            **kwargs: Additional arguments

        Yields:
            Generated text chunks
        """
        raise NotImplementedError

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get token count for text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.

        Returns:
            Current configuration
        """
        raise NotImplementedError

    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration.

        Args:
            config: New configuration values
        """
        raise NotImplementedError
