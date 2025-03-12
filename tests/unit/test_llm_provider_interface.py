"""Unit tests for LLM provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage

from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self) -> None:
        """Initialize provider."""
        self.config = GenerationConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            top_k=40,
        )

    def generate(
        self,
        messages: list[HumanMessage],
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
        if not messages:
            return ""
        return f"Response to: {messages[0].content}"

    async def generate_stream(
        self,
        messages: list[HumanMessage],
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

        words = f"Response to: {messages[0].content}".split()
        for word in words:
            yield word + " "

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Token count.

        """
        # Simple mock implementation
        return len(text.split())

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If configuration is invalid.

        """
        if config.temperature < 0 or config.temperature > 1:
            msg = "Temperature must be between 0 and 1"
            raise ValueError(msg)
        if config.max_tokens < 1:
            msg = "Max tokens must be positive"
            raise ValueError(msg)

    def get_config(self) -> GenerationConfig:
        """Get current configuration.

        Returns:
            Current configuration.

        """
        return self.config

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.

        """
        self.config = config


def test_llm_provider_protocol_compliance() -> None:
    """Test that MockLLMProvider implements LLMProvider protocol."""
    provider = MockLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_generate_method() -> None:
    """Test generate method."""
    provider = MockLLMProvider()
    messages = [HumanMessage(content="Hello")]
    response = provider.generate(messages)
    assert response == "Response to: Hello"

    # Test with empty messages
    response = provider.generate([])
    assert response == ""

    # Test with custom config
    config = GenerationConfig(model="test-model", temperature=0.5, max_tokens=100)
    response = provider.generate(messages, config=config)
    assert response == "Response to: Hello"


@pytest.mark.asyncio
async def test_generate_stream_method() -> None:
    """Test generate_stream method."""
    provider = MockLLMProvider()
    messages = [HumanMessage(content="Hello")]

    # Collect stream chunks
    chunks = []
    async for chunk in provider.generate_stream(messages):
        chunks.append(chunk)

    assert "".join(chunks).strip() == "Response to: Hello"

    # Test with empty messages
    chunks = []
    async for chunk in provider.generate_stream([]):
        chunks.append(chunk)

    assert chunks == []

    # Test with custom config
    config = GenerationConfig(model="test-model", temperature=0.5, max_tokens=100)
    chunks = []
    async for chunk in provider.generate_stream(messages, config=config):
        chunks.append(chunk)

    assert "".join(chunks).strip() == "Response to: Hello"


def test_count_tokens_method() -> None:
    """Test count_tokens method."""
    provider = MockLLMProvider()
    count = provider.count_tokens("Hello world")
    assert count == 2

    count = provider.count_tokens("")
    assert count == 0

    count = provider.count_tokens("This is a longer text with multiple words")
    assert count == 8


def test_validate_config_method() -> None:
    """Test validate_config method."""
    provider = MockLLMProvider()

    # Valid config
    valid_config = GenerationConfig(model="test-model", temperature=0.7, max_tokens=100)
    provider.validate_config(valid_config)  # Should not raise

    # Invalid temperature
    invalid_temp_config = GenerationConfig(model="test-model", temperature=1.5, max_tokens=100)
    with pytest.raises(ValueError):
        provider.validate_config(invalid_temp_config)

    # Invalid max_tokens
    invalid_tokens_config = GenerationConfig(model="test-model", temperature=0.7, max_tokens=0)
    with pytest.raises(ValueError):
        provider.validate_config(invalid_tokens_config)


def test_get_config_method() -> None:
    """Test get_config method."""
    provider = MockLLMProvider()
    config = provider.get_config()
    assert isinstance(config, GenerationConfig)
    assert config.model == "test-model"
    assert config.temperature == 0.7
    assert config.max_tokens == 1024
    assert config.top_p == 0.9
    assert config.top_k == 40


def test_update_config_method() -> None:
    """Test update_config method."""
    provider = MockLLMProvider()
    new_config = GenerationConfig(model="new-model", temperature=0.5, max_tokens=500)
    provider.update_config(new_config)

    config = provider.get_config()
    assert config.model == "new-model"
    assert config.temperature == 0.5
    assert config.max_tokens == 500

    # Check default values for parameters not explicitly set
    assert config.top_p == 0.95  # Default value from GenerationConfig
    assert config.top_k == 40  # Default value from GenerationConfig
