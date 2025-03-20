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
        if config:
            self.validate_config(config)
        return "Test response"

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
        for chunk in ["Test", " stream", " response"]:
            yield chunk
            await asyncio.sleep(0.1)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        # Simple mock implementation - 1 token per word
        return len(text.split())

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration."""
        if config.temperature < 0 or config.temperature > 1:
            msg = "Temperature must be between 0 and 1"
            raise ValueError(msg)
        if config.max_tokens < 1:
            msg = "Max tokens must be positive"
            raise ValueError(msg)
        if config.top_p < 0 or config.top_p > 1:
            msg = "Top p must be between 0 and 1"
            raise ValueError(msg)
        if config.top_k < 1:
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
def provider():
    """Fixture providing a ComprehensiveLLMProvider instance."""
    return ComprehensiveLLMProvider()


def test_provider_implements_protocol() -> None:
    """Test that ComprehensiveLLMProvider implements LLMProvider protocol."""
    provider = ComprehensiveLLMProvider()
    assert isinstance(provider, LLMProvider)


def test_generate_empty_messages(provider) -> None:
    """Test generate with empty messages."""
    result = provider.generate([])
    assert result == ""


def test_generate_with_messages(provider) -> None:
    """Test generate with messages."""
    messages = [
        HumanMessage(content="Test message"),
    ]
    result = provider.generate(messages)
    assert result == "Test response"


def test_generate_with_config(provider) -> None:
    """Test generate with custom config."""
    messages = [
        HumanMessage(content="Test message"),
    ]
    config = GenerationConfig(
        model="test-model",
        temperature=0.5,
        max_tokens=50,
        top_p=0.8,
        top_k=30,
    )
    result = provider.generate(messages, config=config)
    assert result == "Test response"


@pytest.mark.asyncio
async def test_generate_stream_empty_messages(provider) -> None:
    """Test generate_stream with empty messages."""
    chunks = [chunk async for chunk in provider.generate_stream([])]
    assert not chunks


@pytest.mark.asyncio
async def test_generate_stream_with_messages(provider) -> None:
    """Test generate_stream with messages."""
    messages = [
        HumanMessage(content="Test message"),
    ]
    chunks = [chunk async for chunk in provider.generate_stream(messages)]
    assert chunks == ["Test", " stream", " response"]


@pytest.mark.asyncio
async def test_generate_stream_with_config(provider) -> None:
    """Test generate_stream with custom config."""
    messages = [
        HumanMessage(content="Test message"),
    ]
    config = GenerationConfig(
        model="test-model",
        temperature=0.5,
        max_tokens=50,
        top_p=0.8,
        top_k=30,
    )
    chunks = [chunk async for chunk in provider.generate_stream(messages, config=config)]
    assert chunks == ["Test", " stream", " response"]


def test_count_tokens_empty(provider) -> None:
    """Test count_tokens with empty text."""
    count = provider.count_tokens("")
    assert count == 0


def test_count_tokens_single_word(provider) -> None:
    """Test count_tokens with single word."""
    count = provider.count_tokens("hello")
    assert count == 1


def test_count_tokens_multiple_words(provider) -> None:
    """Test count_tokens with multiple words."""
    count = provider.count_tokens("hello world")
    assert count == 2


def test_validate_config_valid(provider) -> None:
    """Test validate_config with valid config."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        top_k=40,
    )
    provider.validate_config(config)  # Should not raise


def test_validate_config_invalid_temperature(provider) -> None:
    """Test validate_config with invalid temperature."""
    config = GenerationConfig(
        model="test-model",
        temperature=1.5,  # Invalid
        max_tokens=100,
        top_p=0.9,
        top_k=40,
    )
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        provider.validate_config(config)


def test_validate_config_invalid_max_tokens(provider) -> None:
    """Test validate_config with invalid max_tokens."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=0,  # Invalid
        top_p=0.9,
        top_k=40,
    )
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        provider.validate_config(config)


def test_validate_config_invalid_top_p(provider) -> None:
    """Test validate_config with invalid top_p."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100,
        top_p=1.5,  # Invalid
        top_k=40,
    )
    with pytest.raises(ValueError, match="Top p must be between 0 and 1"):
        provider.validate_config(config)


def test_validate_config_invalid_top_k(provider) -> None:
    """Test validate_config with invalid top_k."""
    config = GenerationConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        top_k=0,  # Invalid
    )
    with pytest.raises(ValueError, match="Top k must be positive"):
        provider.validate_config(config)


def test_get_config(provider) -> None:
    """Test get_config returns current configuration."""
    config = provider.get_config()
    assert isinstance(config, GenerationConfig)
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 0.9
    assert config.top_k == 40


def test_update_config(provider) -> None:
    """Test update_config updates configuration."""
    new_config = GenerationConfig(
        model="test-model",
        temperature=0.5,
        max_tokens=50,
        top_p=0.8,
        top_k=30,
    )
    provider.update_config(new_config)
    current_config = provider.get_config()
    assert current_config.temperature == 0.5
    assert current_config.max_tokens == 50
    assert current_config.top_p == 0.8
    assert current_config.top_k == 30


def test_update_config_invalid(provider) -> None:
    """Test update_config with invalid configuration."""
    new_config = GenerationConfig(
        model="test-model",
        temperature=1.5,  # Invalid
        max_tokens=50,
        top_p=0.8,
        top_k=30,
    )
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        provider.update_config(new_config)
