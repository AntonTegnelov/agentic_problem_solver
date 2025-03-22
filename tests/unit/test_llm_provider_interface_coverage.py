"""Tests for improving coverage of the LLMProvider interface."""

import asyncio
from collections.abc import AsyncGenerator

import pytest
from langchain_core.messages import HumanMessage

from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig


class ComprehensiveLLMProvider:
    """A comprehensive implementation of LLMProvider for testing."""

    def __init__(self) -> None:
        """Initialize the provider with default configuration."""
        self._config = GenerationConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            top_k=40,
        )

    def generate(
        self,
        messages: list[HumanMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages."""
        if not messages:
            return ""

        message_content = messages[0].content if messages else ""

        if config:
            return f"Response for: {message_content} (with custom config)"

        return f"Response for: {message_content}"

    async def generate_stream(
        self,
        messages: list[HumanMessage],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages."""
        if not messages:
            return

        if config:
            self.validate_config(config)

        response = "Test response"
        for char in response:
            yield char
            await asyncio.sleep(0.01)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        # Simple mock implementation - 1 token per word
        return len(text.split())

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration parameters.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If any configuration parameter is invalid.

        """
        if config.temperature < 0 or config.temperature > 1:
            msg = "Temperature must be between 0 and 1"
            raise ValueError(msg)

        if config.max_tokens <= 0:
            msg = "Max tokens must be positive"
            raise ValueError(msg)

        if config.top_p < 0 or config.top_p > 1:
            msg = "Top p must be between 0 and 1"
            raise ValueError(msg)

        if config.top_k <= 0:
            msg = "Top k must be positive"
            raise ValueError(msg)

    def get_config(self) -> GenerationConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration."""
        self.validate_config(config)
        self._config = config


@pytest.fixture
def provider() -> ComprehensiveLLMProvider:
    """Fixture providing a ComprehensiveLLMProvider instance."""
    return ComprehensiveLLMProvider()


def test_provider_implements_protocol() -> None:
    """Test that ComprehensiveLLMProvider implements LLMProvider protocol."""
    provider = ComprehensiveLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_generate_empty_messages(provider: ComprehensiveLLMProvider) -> None:
    """Test generate with empty messages."""
    result = provider.generate([])
    assert result == ""


def test_generate_with_messages(provider: ComprehensiveLLMProvider) -> None:
    """Test generate with messages."""
    messages = [
        HumanMessage(content="Hello, world!"),
    ]
    result = provider.generate(messages)
    assert result == "Response for: Hello, world!"


def test_generate_with_config(provider: ComprehensiveLLMProvider) -> None:
    """Test generate with custom config."""
    messages = [
        HumanMessage(content="Hello, world!"),
    ]
    config = GenerationConfig(
        model="custom-model",
        temperature=0.5,
        max_tokens=200,
    )
    result = provider.generate(messages, config=config)
    assert result == "Response for: Hello, world! (with custom config)"


@pytest.mark.asyncio
async def test_generate_stream_empty_messages(provider: ComprehensiveLLMProvider) -> None:
    """Test generate_stream with empty messages."""
    chunks = [chunk async for chunk in provider.generate_stream([])]
    assert chunks == []


@pytest.mark.asyncio
async def test_generate_stream_with_messages(provider: ComprehensiveLLMProvider) -> None:
    """Test generate_stream with messages."""
    messages = [
        HumanMessage(content="Hello, world!"),
    ]
    chunks = [chunk async for chunk in provider.generate_stream(messages)]
    assert chunks == ["T", "e", "s", "t", " ", "r", "e", "s", "p", "o", "n", "s", "e"]


@pytest.mark.asyncio
async def test_generate_stream_with_config(provider: ComprehensiveLLMProvider) -> None:
    """Test generate_stream with custom config."""
    messages = [
        HumanMessage(content="Hello, world!"),
    ]
    config = GenerationConfig(
        model="custom-model",
        temperature=0.5,
        max_tokens=50,
    )
    chunks = [chunk async for chunk in provider.generate_stream(messages, config=config)]
    assert chunks == ["T", "e", "s", "t", " ", "r", "e", "s", "p", "o", "n", "s", "e"]


def test_count_tokens_empty(provider: ComprehensiveLLMProvider) -> None:
    """Test count_tokens with empty text."""
    count = provider.count_tokens("")
    assert count == 0


def test_count_tokens_single_word(provider: ComprehensiveLLMProvider) -> None:
    """Test count_tokens with single word."""
    count = provider.count_tokens("hello")
    assert count == 1


def test_count_tokens_multiple_words(provider: ComprehensiveLLMProvider) -> None:
    """Test count_tokens with multiple words."""
    count = provider.count_tokens("hello world")
    assert count == 2


def test_validate_config_valid(provider: ComprehensiveLLMProvider) -> None:
    """Test validate_config with valid config."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        top_k=40,
    )
    provider.validate_config(config)
    # No exception raised means test passed


def test_validate_config_invalid_temperature(provider: ComprehensiveLLMProvider) -> None:
    """Test validate_config with invalid temperature."""
    config = GenerationConfig(
        model="test-model",
        temperature=2.0,  # Invalid: > 1.0
        max_tokens=100,
        top_p=0.9,
        top_k=40,
    )
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        provider.validate_config(config)


def test_validate_config_invalid_max_tokens(provider: ComprehensiveLLMProvider) -> None:
    """Test validate_config with invalid max_tokens."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=-1,  # Invalid: < 0
        top_p=0.9,
        top_k=40,
    )
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        provider.validate_config(config)


def test_validate_config_invalid_top_p(provider: ComprehensiveLLMProvider) -> None:
    """Test validate_config with invalid top_p."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100,
        top_p=1.5,  # Invalid: > 1.0
        top_k=40,
    )
    with pytest.raises(ValueError, match="Top p must be between 0 and 1"):
        provider.validate_config(config)


def test_validate_config_invalid_top_k(provider: ComprehensiveLLMProvider) -> None:
    """Test validate_config with invalid top_k."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        top_k=-5,  # Invalid: < 0
    )
    with pytest.raises(ValueError, match="Top k must be positive"):
        provider.validate_config(config)


def test_get_config(provider: ComprehensiveLLMProvider) -> None:
    """Test get_config returns current configuration."""
    config = provider.get_config()
    assert config.model == "test-model"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 0.9
    assert config.top_k == 40


def test_update_config(provider: ComprehensiveLLMProvider) -> None:
    """Test update_config updates configuration."""
    new_config = GenerationConfig(
        model="new-model",
        temperature=0.5,
        max_tokens=200,
        top_p=0.8,
        top_k=30,
    )
    provider.update_config(new_config)

    updated_config = provider.get_config()
    assert updated_config.model == "new-model"
    assert updated_config.temperature == 0.5
    assert updated_config.max_tokens == 200
    assert updated_config.top_p == 0.8
    assert updated_config.top_k == 30


def test_update_config_invalid(provider: ComprehensiveLLMProvider) -> None:
    """Test update_config with invalid configuration."""
    new_config = GenerationConfig(
        model="new-model",
        temperature=1.5,  # Invalid: > 1.0
        max_tokens=200,
        top_p=0.8,
        top_k=30,
    )
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        provider.update_config(new_config)
