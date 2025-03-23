"""Advanced tests for LLM provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig
from src.messages.creation import create_human_message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Message


class MinimalLLMProvider:
    """Minimal implementation of LLMProvider protocol."""

    def generate(
        self,
        messages: list[Message],  # noqa: ARG002
        *,
        config: GenerationConfig | None = None,  # noqa: ARG002
    ) -> str:
        """Generate response from messages."""
        return "minimal"

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages."""
        if not messages:
            return
        yield "minimal"

    def count_tokens(self, text: str) -> int:  # noqa: ARG002
        """Count tokens in text."""
        return 1

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration."""

    def get_config(self) -> GenerationConfig:
        """Get current configuration."""
        return GenerationConfig(model="minimal")

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration."""


class PartialLLMProvider:
    """Partial implementation missing some methods."""

    def generate(
        self,
        messages: list[Message],  # noqa: ARG002
        *,
        config: GenerationConfig | None = None,  # noqa: ARG002
    ) -> str:
        """Generate response from messages."""
        return "partial"


class ComplexLLMProvider:
    """Complex implementation with advanced features."""

    def __init__(self) -> None:
        """Initialize provider."""
        self.config = GenerationConfig(model="complex")
        self.history: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate response with history tracking."""
        response = f"complex: {len(messages)} messages"
        self.history.append({"type": "generate", "messages": messages, "config": config})
        return response

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream with history tracking."""
        if not messages:
            return
        self.history.append({"type": "stream", "messages": messages, "config": config})
        for i in range(3):
            yield f"chunk{i}"

    def count_tokens(self, text: str) -> int:
        """Count tokens with history tracking."""
        count = len(text.split())
        self.history.append({"type": "count", "text": text, "count": count})
        return count

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration with history tracking."""
        self.history.append({"type": "validate", "config": config})
        if not config.model:
            msg = "Model is required"
            raise ValueError(msg)

    def get_config(self) -> GenerationConfig:
        """Get current configuration."""
        self.history.append({"type": "get_config"})
        return self.config

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration with history tracking."""
        self.history.append({"type": "update", "config": config})
        self.config = config


def test_minimal_provider_protocol_compliance() -> None:
    """Test that MinimalLLMProvider implements LLMProvider protocol."""
    provider = MinimalLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_partial_provider_noncompliance() -> None:
    """Test that PartialLLMProvider does not implement LLMProvider protocol."""
    provider = PartialLLMProvider()
    assert not isinstance(provider, LLMProvider)


def test_complex_provider_history_tracking() -> None:
    """Test history tracking in ComplexLLMProvider."""
    provider = ComplexLLMProvider()
    messages = [create_human_message("test")]
    config = GenerationConfig(model="test")

    # Test generate
    response = provider.generate(messages, config=config)
    assert response == "complex: 1 messages"
    assert provider.history[-1] == {"type": "generate", "messages": messages, "config": config}

    # Test count_tokens
    count = provider.count_tokens("test text")
    assert count == 2
    assert provider.history[-1] == {"type": "count", "text": "test text", "count": 2}

    # Test config operations
    provider.validate_config(config)
    assert provider.history[-1] == {"type": "validate", "config": config}

    provider.update_config(config)
    assert provider.history[-1] == {"type": "update", "config": config}

    provider.get_config()
    assert provider.history[-1] == {"type": "get_config"}


@pytest.mark.asyncio
async def test_complex_provider_stream_history() -> None:
    """Test stream history tracking in ComplexLLMProvider."""
    provider = ComplexLLMProvider()
    messages = [create_human_message("test")]
    config = GenerationConfig(model="test")

    chunks = [chunk async for chunk in provider.generate_stream(messages, config=config)]
    assert chunks == ["chunk0", "chunk1", "chunk2"]
    assert provider.history[-1] == {"type": "stream", "messages": messages, "config": config}


def test_complex_provider_config_validation() -> None:
    """Test configuration validation in ComplexLLMProvider."""
    provider = ComplexLLMProvider()

    # Test valid config
    valid_config = GenerationConfig(model="test")
    provider.validate_config(valid_config)  # Should not raise

    # Test invalid config
    invalid_config = GenerationConfig(model="")
    with pytest.raises(ValueError, match="Model is required"):
        provider.validate_config(invalid_config)


def test_complex_provider_config_update() -> None:
    """Test configuration update in ComplexLLMProvider."""
    provider = ComplexLLMProvider()
    initial_config = provider.get_config()
    assert initial_config.model == "complex"

    new_config = GenerationConfig(model="new")
    provider.update_config(new_config)
    updated_config = provider.get_config()
    assert updated_config.model == "new"
    assert provider.history[-2:] == [
        {"type": "update", "config": new_config},
        {"type": "get_config"},
    ]
