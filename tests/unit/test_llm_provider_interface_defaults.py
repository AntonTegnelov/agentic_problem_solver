"""Tests for the default implementations of the LLMProvider protocol."""

from collections.abc import AsyncGenerator

import pytest
from langchain_core.messages import BaseMessage as Message
from langchain_core.messages import HumanMessage

from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig


class DefaultLLMProvider:
    """A provider that uses all default implementations from the protocol."""

    def __init__(self) -> None:
        """Initialize provider."""
        self.config = GenerationConfig(model="test-model")

    async def generate(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages."""
        if not messages:
            return ""
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

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        raise NotImplementedError

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration."""
        raise NotImplementedError

    def get_config(self) -> GenerationConfig:
        """Get current configuration."""
        raise NotImplementedError

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration."""
        raise NotImplementedError


@pytest.fixture
def provider():
    """Fixture providing a DefaultLLMProvider instance."""
    return DefaultLLMProvider()


def test_provider_implements_protocol() -> None:
    """Test that DefaultLLMProvider implements LLMProvider protocol."""
    provider = DefaultLLMProvider()
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_generate_stream_default_empty_messages(provider) -> None:
    """Test generate_stream with empty messages returns empty string."""
    chunks = []
    async for chunk in provider.generate_stream([]):
        chunks.append(chunk)
    assert not chunks


@pytest.mark.asyncio
async def test_generate_stream_default_with_config(provider) -> None:
    """Test generate_stream with empty messages and config returns empty string."""
    config = GenerationConfig(model="test-model")
    chunks = []
    async for chunk in provider.generate_stream([], config=config):
        chunks.append(chunk)
    assert not chunks


@pytest.mark.asyncio
async def test_generate_stream_default_with_messages(provider) -> None:
    """Test generate_stream with messages returns empty string."""
    messages = [HumanMessage(content="test")]
    async for chunk in provider.generate_stream(messages):
        assert chunk == ""


@pytest.mark.asyncio
async def test_generate_stream_default_with_messages_and_config(provider) -> None:
    """Test generate_stream with both messages and config returns empty string."""
    messages = [HumanMessage(content="test")]
    config = GenerationConfig(model="test-model")
    async for chunk in provider.generate_stream(messages, config=config):
        assert chunk == ""


@pytest.mark.asyncio
async def test_generate_default_empty_messages(provider) -> None:
    """Test generate with empty messages returns empty string."""
    result = await provider.generate([])
    assert result == ""


@pytest.mark.asyncio
async def test_generate_default_with_config(provider) -> None:
    """Test generate with config returns empty string."""
    config = GenerationConfig(model="test-model")
    result = await provider.generate([], config=config)
    assert result == ""


@pytest.mark.asyncio
async def test_generate_default_with_messages(provider) -> None:
    """Test generate with messages returns empty string."""
    messages = [HumanMessage(content="test")]
    result = await provider.generate(messages)
    assert result == ""


def test_count_tokens_default(provider) -> None:
    """Test count_tokens raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        provider.count_tokens("test")


def test_validate_config_default(provider) -> None:
    """Test validate_config raises NotImplementedError."""
    config = GenerationConfig(model="test-model")
    with pytest.raises(NotImplementedError):
        provider.validate_config(config)


def test_get_config_default(provider) -> None:
    """Test get_config raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        provider.get_config()


def test_update_config_default(provider) -> None:
    """Test update_config raises NotImplementedError."""
    config = GenerationConfig(model="test-model")
    with pytest.raises(NotImplementedError):
        provider.update_config(config)
