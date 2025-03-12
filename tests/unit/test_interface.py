"""Tests for the LLM provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.llm_providers.interface import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Message
    from src.llm_providers.type_defs import GenerationConfig


# Create a mock message class for testing
class MockMessage:
    """Mock message for testing."""

    def __init__(self, content: str) -> None:
        """Initialize mock message.

        Args:
            content: Message content.

        """
        self.content = content


# Create a mock config class for testing
class MockConfig:
    """Mock generation config for testing."""

    def __init__(self, temperature: float = 0.7) -> None:
        """Initialize mock config.

        Args:
            temperature: Temperature value.

        """
        self.temperature = temperature


# Create a concrete implementation of the LLMProvider protocol for testing
class TestProvider:
    """Test implementation of LLMProvider protocol."""

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

        """
        return "Test response"

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
        yield "Test"
        yield " response"
        yield " stream"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Token count.

        """
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

    def get_config(self) -> GenerationConfig:
        """Get current configuration.

        Returns:
            Current configuration.

        """
        return MockConfig()

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.

        """


class TestLLMProviderInterface:
    """Tests for the LLMProvider interface."""

    def test_provider_implements_protocol(self) -> None:
        """Test that TestProvider implements LLMProvider protocol."""
        provider = TestProvider()
        assert isinstance(provider, LLMProvider)

    def test_generate_method(self) -> None:
        """Test generate method."""
        provider = TestProvider()
        messages = [MockMessage("Hello")]
        result = provider.generate(messages)
        assert result == "Test response"

        # Test with config
        config = MockConfig()
        result = provider.generate(messages, config=config)
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_stream_method(self) -> None:
        """Test generate_stream method."""
        provider = TestProvider()
        messages = [MockMessage("Hello")]

        # Collect all chunks
        chunks = []
        async for chunk in provider.generate_stream(messages):
            chunks.append(chunk)

        assert chunks == ["Test", " response", " stream"]

        # Test with config
        config = MockConfig()
        chunks = []
        async for chunk in provider.generate_stream(messages, config=config):
            chunks.append(chunk)

        assert chunks == ["Test", " response", " stream"]

        # Test with empty messages
        chunks = []
        async for chunk in provider.generate_stream([]):
            chunks.append(chunk)

        assert chunks == []

    def test_count_tokens_method(self) -> None:
        """Test count_tokens method."""
        provider = TestProvider()
        result = provider.count_tokens("Hello world")
        assert result == 2

    def test_validate_config_method(self) -> None:
        """Test validate_config method."""
        provider = TestProvider()

        # Valid config
        config = MockConfig(temperature=0.5)
        provider.validate_config(config)  # Should not raise

        # Invalid config
        config = MockConfig(temperature=1.5)
        with pytest.raises(ValueError):
            provider.validate_config(config)

    def test_get_config_method(self) -> None:
        """Test get_config method."""
        provider = TestProvider()
        config = provider.get_config()
        assert isinstance(config, MockConfig)
        assert config.temperature == 0.7

    def test_update_config_method(self) -> None:
        """Test update_config method."""
        provider = TestProvider()
        config = MockConfig(temperature=0.5)
        provider.update_config(config)  # Should not raise
