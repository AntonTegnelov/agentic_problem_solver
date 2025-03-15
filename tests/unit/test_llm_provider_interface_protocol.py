"""Unit tests for LLM provider protocol compliance and edge cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig
from src.messages.creation import create_human_message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Message


class CustomLLMProvider:
    """Custom LLM provider implementation for testing protocol compliance."""

    def __init__(self) -> None:
        """Initialize provider."""
        self.config = GenerationConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            top_k=40,
        )
        self.error_mode = ""

    def generate(
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

        Raises:
            ValueError: If error_mode is "generate".
            RuntimeError: If error_mode is "runtime".

        """
        if self.error_mode == "generate":
            msg = "Simulated generate error"
            raise ValueError(msg)
        if self.error_mode == "runtime":
            msg = "Simulated runtime error"
            raise RuntimeError(msg)

        if not messages:
            return ""

        response = f"Response to: {messages[0].content}"
        if config:
            if config.temperature > 0.7:
                response += " (high temperature)"
            if config.max_tokens < 100:
                response = response[:100]

        return response

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

        Raises:
            ValueError: If error_mode is "stream".
            RuntimeError: If error_mode is "runtime".

        """
        if self.error_mode == "stream":
            msg = "Simulated stream error"
            raise ValueError(msg)
        if self.error_mode == "runtime":
            msg = "Simulated runtime error"
            raise RuntimeError(msg)

        if not messages:
            return

        yield "Chunk 1"
        if config and config.temperature > 0.7:
            yield "High temperature chunk"
        yield "Chunk 2"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Token count.

        Raises:
            ValueError: If error_mode is "count".
            RuntimeError: If error_mode is "runtime".

        """
        if self.error_mode == "count":
            msg = "Simulated count error"
            raise ValueError(msg)
        if self.error_mode == "runtime":
            msg = "Simulated runtime error"
            raise RuntimeError(msg)

        if not text:
            return 0

        # Handle different text formats
        if text.startswith(("{", "[")):
            # JSON-like text
            return len(text) // 2
        if text.startswith(("<", "</")):
            # XML-like text
            # Count each tag as a token and content between tags as tokens
            import re

            tags = re.findall(r"<[^>]+>", text)  # Find all tags
            content = re.sub(r"<[^>]+>", " ", text).strip()  # Remove tags to get content
            return len(tags) + (len(content.split()) if content else 0)
        return len(text.split())

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If configuration is invalid or error_mode is "validate".
            RuntimeError: If error_mode is "runtime".

        """
        if self.error_mode == "validate":
            msg = "Simulated validate error"
            raise ValueError(msg)
        if self.error_mode == "runtime":
            msg = "Simulated runtime error"
            raise RuntimeError(msg)

        if not config.model:
            msg = "Model name cannot be empty"
            raise ValueError(msg)
        if config.temperature < 0 or config.temperature > 1:
            msg = "Temperature must be between 0 and 1"
            raise ValueError(msg)
        if config.max_tokens < 1:
            msg = "Max tokens must be positive"
            raise ValueError(msg)
        if config.top_p < 0 or config.top_p > 1:
            msg = "Top P must be between 0 and 1"
            raise ValueError(msg)
        if config.top_k < 1:
            msg = "Top K must be positive"
            raise ValueError(msg)

    def get_config(self) -> GenerationConfig:
        """Get current configuration.

        Returns:
            Current configuration.

        Raises:
            ValueError: If error_mode is "get_config".
            RuntimeError: If error_mode is "runtime".

        """
        if self.error_mode == "get_config":
            msg = "Simulated get_config error"
            raise ValueError(msg)
        if self.error_mode == "runtime":
            msg = "Simulated runtime error"
            raise RuntimeError(msg)

        return self.config

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.

        Raises:
            ValueError: If error_mode is "update_config".
            RuntimeError: If error_mode is "runtime".

        """
        if self.error_mode == "update_config":
            msg = "Simulated update_config error"
            raise ValueError(msg)
        if self.error_mode == "runtime":
            msg = "Simulated runtime error"
            raise RuntimeError(msg)

        self.validate_config(config)
        self.config = config


def test_custom_provider_protocol_compliance() -> None:
    """Test that CustomLLMProvider implements LLMProvider protocol."""
    provider = CustomLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_custom_provider_generate_with_config() -> None:
    """Test generate method with different configurations."""
    provider = CustomLLMProvider()
    message = create_human_message("test message")

    # Test with default config
    response = provider.generate([message])
    assert response == "Response to: test message"

    # Test with high temperature
    config = GenerationConfig(model="test", temperature=0.8)
    response = provider.generate([message], config=config)
    assert response == "Response to: test message (high temperature)"

    # Test with low max_tokens
    config = GenerationConfig(model="test", max_tokens=50)
    response = provider.generate([message], config=config)
    assert len(response) <= 100


@pytest.mark.asyncio
async def test_custom_provider_generate_stream_with_config() -> None:
    """Test generate_stream method with different configurations."""
    provider = CustomLLMProvider()
    message = create_human_message("test message")

    # Test with default config
    chunks = [chunk async for chunk in provider.generate_stream([message])]
    assert chunks == ["Chunk 1", "Chunk 2"]

    # Test with high temperature
    config = GenerationConfig(model="test", temperature=0.8)
    chunks = [chunk async for chunk in provider.generate_stream([message], config=config)]
    assert chunks == ["Chunk 1", "High temperature chunk", "Chunk 2"]

    # Test with high max_tokens
    config = GenerationConfig(model="test", max_tokens=200)
    chunks = [chunk async for chunk in provider.generate_stream([message], config=config)]
    assert chunks == ["Chunk 1", "Chunk 2"]


def test_custom_provider_count_tokens_with_formats() -> None:
    """Test count_tokens method with different text formats."""
    provider = CustomLLMProvider()

    # Test empty text
    assert provider.count_tokens("") == 0

    # Test plain text
    assert provider.count_tokens("Hello world") == 2

    # Test JSON-like text
    assert provider.count_tokens('{"key": "value"}') == 8

    # Test XML-like text
    assert provider.count_tokens("<tag>content</tag>") == 3  # 2 tags + 1 content word


def test_custom_provider_validate_config_comprehensive() -> None:
    """Test validate_config method with various configurations."""
    provider = CustomLLMProvider()

    # Test valid config
    config = GenerationConfig(
        model="test",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        top_k=40,
    )
    provider.validate_config(config)

    # Test invalid model
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        provider.validate_config(GenerationConfig(model=""))

    # Test invalid temperature
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        provider.validate_config(GenerationConfig(model="test", temperature=1.5))

    # Test invalid max_tokens
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        provider.validate_config(GenerationConfig(model="test", max_tokens=0))

    # Test invalid top_p
    with pytest.raises(ValueError, match="Top P must be between 0 and 1"):
        provider.validate_config(GenerationConfig(model="test", top_p=1.5))

    # Test invalid top_k
    with pytest.raises(ValueError, match="Top K must be positive"):
        provider.validate_config(GenerationConfig(model="test", top_k=0))


def test_custom_provider_error_handling() -> None:
    """Test error handling in all methods."""
    provider = CustomLLMProvider()
    message = create_human_message("test message")
    config = GenerationConfig(model="test")

    # Test generate error
    provider.error_mode = "generate"
    with pytest.raises(ValueError, match="Simulated generate error"):
        provider.generate([message])

    # Test count_tokens error
    provider.error_mode = "count"
    with pytest.raises(ValueError, match="Simulated count error"):
        provider.count_tokens("test")

    # Test validate_config error
    provider.error_mode = "validate"
    with pytest.raises(ValueError, match="Simulated validate error"):
        provider.validate_config(config)

    # Test get_config error
    provider.error_mode = "get_config"
    with pytest.raises(ValueError, match="Simulated get_config error"):
        provider.get_config()

    # Test update_config error
    provider.error_mode = "update_config"
    with pytest.raises(ValueError, match="Simulated update_config error"):
        provider.update_config(config)

    # Test runtime error in all methods
    provider.error_mode = "runtime"
    with pytest.raises(RuntimeError, match="Simulated runtime error"):
        provider.generate([message])
    with pytest.raises(RuntimeError, match="Simulated runtime error"):
        provider.count_tokens("test")
    with pytest.raises(RuntimeError, match="Simulated runtime error"):
        provider.validate_config(config)
    with pytest.raises(RuntimeError, match="Simulated runtime error"):
        provider.get_config()
    with pytest.raises(RuntimeError, match="Simulated runtime error"):
        provider.update_config(config)


@pytest.mark.asyncio
async def test_custom_provider_stream_error_handling() -> None:
    """Test error handling in generate_stream method."""
    provider = CustomLLMProvider()
    message = create_human_message("test message")

    # Test stream error
    provider.error_mode = "stream"

    async def _test_stream_error() -> None:
        async for _ in provider.generate_stream([message]):
            pass

    with pytest.raises(ValueError, match="Simulated stream error"):
        await _test_stream_error()

    # Test runtime error
    provider.error_mode = "runtime"

    async def _test_runtime_error() -> None:
        async for _ in provider.generate_stream([message]):
            pass

    with pytest.raises(RuntimeError, match="Simulated runtime error"):
        await _test_runtime_error()
