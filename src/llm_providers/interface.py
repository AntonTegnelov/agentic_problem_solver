"""LLM provider interface definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Message
    from src.llm_providers.type_defs import GenerationConfig


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    async def generate(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages.

        Args:
            messages: Messages to generate from.
            config: Optional generation configuration.

        Returns:
            Generated response.

        """
        ...  # pragma: no cover

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages.

        Args:
            messages: Messages to generate from.
            config: Optional generation configuration.

        Yields:
            Generated response chunks.

        """
        if not messages:
            return
        if config is not None:
            pass
        yield ""

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Token count.

        """
        ...  # pragma: no cover

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If configuration is invalid.

        """
        ...  # pragma: no cover

    def get_config(self) -> GenerationConfig:
        """Get current configuration.

        Returns:
            Current configuration.

        """
        ...  # pragma: no cover

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.

        """
        ...  # pragma: no cover
