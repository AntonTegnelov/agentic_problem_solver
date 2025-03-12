"""Extended unit tests for LLM provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig
from src.messages.creation import create_human_message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Message


class MinimalLLMProvider:
    """Minimal LLM provider implementation for testing protocol compliance."""

    def generate(
        self,
        _messages: list[Message],
        *,
        _config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages."""
        return ""

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages."""
        if not messages:
            return
        if config is not None:
            pass
        yield ""

    def count_tokens(self, _text: str) -> int:
        """Count tokens in text."""
        return 0

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration."""

    def get_config(self) -> GenerationConfig:
        """Get current configuration."""
        return GenerationConfig(model="test")

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration."""


def test_minimal_provider_protocol_compliance() -> None:
    """Test that MinimalLLMProvider implements LLMProvider protocol."""
    provider = MinimalLLMProvider()
    assert isinstance(provider, LLMProvider)


class PartialLLMProvider:
    """Partial LLM provider implementation for testing protocol compliance."""

    def generate(
        self,
        _messages: list[Message],
        *,
        _config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages."""
        return ""

    def count_tokens(self, _text: str) -> int:
        """Count tokens in text."""
        return 0


def test_partial_provider_protocol_noncompliance() -> None:
    """Test that PartialLLMProvider does not implement LLMProvider protocol."""
    provider = PartialLLMProvider()
    assert not isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_generate_stream_empty_messages() -> None:
    """Test generate_stream method with empty messages."""
    provider = MinimalLLMProvider()

    # Test with empty messages
    chunks = [chunk async for chunk in provider.generate_stream([])]
    assert chunks == []


@pytest.mark.asyncio
async def test_generate_stream_with_config() -> None:
    """Test generate_stream method with config."""
    provider = MinimalLLMProvider()
    messages = [create_human_message("test")]
    config = GenerationConfig(model="test-model", temperature=0.5)

    chunks = [chunk async for chunk in provider.generate_stream(messages, config=config)]
    assert chunks == [""]


class CustomLLMProvider:
    """Custom LLM provider with specific implementation."""

    def __init__(self) -> None:
        """Initialize provider."""
        self._config = GenerationConfig(model="custom-model")

    def generate(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages."""
        if not messages:
            return ""

        actual_config = config or self._config
        return f"Response using {actual_config.model}"

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages."""
        if not messages:
            return

        actual_config = config or self._config
        yield f"Response using {actual_config.model}"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text)

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration."""
        if not config.model:
            msg = "Model is required"
            raise ValueError(msg)

    def get_config(self) -> GenerationConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration."""
        self._config = config


def test_custom_provider_protocol_compliance() -> None:
    """Test that CustomLLMProvider implements LLMProvider protocol."""
    provider = CustomLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_custom_provider_generate() -> None:
    """Test CustomLLMProvider generate method."""
    provider = CustomLLMProvider()
    messages = [create_human_message("test")]

    # Test with default config
    response = provider.generate(messages)
    assert response == "Response using custom-model"

    # Test with custom config
    config = GenerationConfig(model="test-model")
    response = provider.generate(messages, config=config)
    assert response == "Response using test-model"


@pytest.mark.asyncio
async def test_custom_provider_generate_stream() -> None:
    """Test CustomLLMProvider generate_stream method."""
    provider = CustomLLMProvider()
    messages = [create_human_message("test")]

    # Test with default config
    chunks = [chunk async for chunk in provider.generate_stream(messages)]
    assert "".join(chunks) == "Response using custom-model"

    # Test with custom config
    config = GenerationConfig(model="test-model")
    chunks = [chunk async for chunk in provider.generate_stream(messages, config=config)]
    assert "".join(chunks) == "Response using test-model"


def test_custom_provider_validate_config() -> None:
    """Test CustomLLMProvider validate_config method."""
    provider = CustomLLMProvider()

    # Valid config
    valid_config = GenerationConfig(model="test-model")
    provider.validate_config(valid_config)  # Should not raise

    # Invalid config
    invalid_config = GenerationConfig(model="")
    with pytest.raises(ValueError, match="Model is required"):
        provider.validate_config(invalid_config)


def test_custom_provider_update_config() -> None:
    """Test CustomLLMProvider update_config method."""
    provider = CustomLLMProvider()

    # Initial config
    assert provider.get_config().model == "custom-model"

    # Update config
    new_config = GenerationConfig(model="new-model")
    provider.update_config(new_config)
    assert provider.get_config().model == "new-model"
