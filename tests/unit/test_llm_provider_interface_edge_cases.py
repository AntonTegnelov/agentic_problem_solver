"""Edge case and error handling tests for LLM provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig
from src.messages.creation import create_human_message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Message


class EdgeCaseLLMProvider:
    """LLM provider implementation for testing edge cases."""

    def __init__(self) -> None:
        """Initialize provider."""
        self.config = GenerationConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            top_k=40,
        )
        self.should_raise = False

    def generate(
        self,
        messages: list[Message],
        *,
        _config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages with error handling."""
        if self.should_raise:
            msg = "Simulated error in generate"
            raise RuntimeError(msg)
        if not messages:
            return ""
        return f"Response to: {messages[0].content}"

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages with error handling."""
        if self.should_raise:
            msg = "Simulated error in generate_stream"
            raise RuntimeError(msg)
        if not messages:
            return
        yield "Chunk 1"
        yield "Chunk 2"
        if config and config.max_tokens > 100:
            yield "Extra chunk for high max_tokens"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with error handling."""
        if self.should_raise:
            msg = "Simulated error in count_tokens"
            raise RuntimeError(msg)
        if not text:
            return 0
        # Handle different text formats
        if text.startswith(("{", "[")):
            # JSON-like text
            return len(text) // 2
        return len(text.split())

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration with comprehensive checks."""
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
        """Get current configuration."""
        if self.should_raise:
            msg = "Simulated error in get_config"
            raise RuntimeError(msg)
        return self.config

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration with validation."""
        if self.should_raise:
            msg = "Simulated error in update_config"
            raise RuntimeError(msg)
        self.validate_config(config)
        self.config = config


def test_protocol_compliance() -> None:
    """Test that EdgeCaseLLMProvider implements LLMProvider protocol."""
    provider = EdgeCaseLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_generate_empty_messages() -> None:
    """Test generate method with empty messages."""
    provider = EdgeCaseLLMProvider()
    assert provider.generate([]) == ""


def test_generate_error_handling() -> None:
    """Test generate method error handling."""
    provider = EdgeCaseLLMProvider()
    provider.should_raise = True
    with pytest.raises(RuntimeError, match="Simulated error in generate"):
        provider.generate([create_human_message("test")])


@pytest.mark.asyncio
async def test_generate_stream_chunks() -> None:
    """Test generate_stream method chunk handling."""
    provider = EdgeCaseLLMProvider()
    messages = [create_human_message("test")]

    # Test with default config
    chunks = [chunk async for chunk in provider.generate_stream(messages)]
    assert chunks == ["Chunk 1", "Chunk 2"]

    # Test with high max_tokens
    config = GenerationConfig(model="test", max_tokens=200)
    chunks = [chunk async for chunk in provider.generate_stream(messages, config=config)]
    assert chunks == ["Chunk 1", "Chunk 2", "Extra chunk for high max_tokens"]


@pytest.mark.asyncio
async def test_generate_stream_error_handling() -> None:
    """Test generate_stream method error handling."""
    provider = EdgeCaseLLMProvider()
    provider.should_raise = True
    stream = provider.generate_stream([create_human_message("test")])
    with pytest.raises(RuntimeError, match="Simulated error in generate_stream"):
        await anext(stream)


def test_count_tokens_different_formats() -> None:
    """Test count_tokens method with different text formats."""
    provider = EdgeCaseLLMProvider()

    # Empty text
    assert provider.count_tokens("") == 0

    # Normal text
    assert provider.count_tokens("Hello world") == 2

    # JSON-like text
    assert provider.count_tokens('{"key": "value"}') == 8
    assert provider.count_tokens("[1, 2, 3]") == 4


def test_count_tokens_error_handling() -> None:
    """Test count_tokens method error handling."""
    provider = EdgeCaseLLMProvider()
    provider.should_raise = True
    with pytest.raises(RuntimeError, match="Simulated error in count_tokens"):
        provider.count_tokens("test")


def test_validate_config_comprehensive() -> None:
    """Test validate_config method with various invalid configurations."""
    provider = EdgeCaseLLMProvider()

    # Empty model name
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        provider.validate_config(GenerationConfig(model=""))

    # Invalid temperature
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        provider.validate_config(GenerationConfig(model="test", temperature=1.5))

    # Invalid max_tokens
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        provider.validate_config(GenerationConfig(model="test", max_tokens=0))

    # Invalid top_p
    with pytest.raises(ValueError, match="Top P must be between 0 and 1"):
        provider.validate_config(GenerationConfig(model="test", top_p=1.5))

    # Invalid top_k
    with pytest.raises(ValueError, match="Top K must be positive"):
        provider.validate_config(GenerationConfig(model="test", top_k=0))


def test_get_config_error_handling() -> None:
    """Test get_config method error handling."""
    provider = EdgeCaseLLMProvider()
    provider.should_raise = True
    with pytest.raises(RuntimeError, match="Simulated error in get_config"):
        provider.get_config()


def test_update_config_validation() -> None:
    """Test update_config method with validation."""
    provider = EdgeCaseLLMProvider()

    # Valid update
    new_config = GenerationConfig(
        model="new-model",
        temperature=0.5,
        max_tokens=2048,
        top_p=0.8,
        top_k=50,
    )
    provider.update_config(new_config)
    assert provider.get_config() == new_config

    # Invalid update
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        provider.update_config(GenerationConfig(model="test", temperature=1.5))


def test_update_config_error_handling() -> None:
    """Test update_config method error handling."""
    provider = EdgeCaseLLMProvider()
    provider.should_raise = True
    with pytest.raises(RuntimeError, match="Simulated error in update_config"):
        provider.update_config(GenerationConfig(model="test"))
