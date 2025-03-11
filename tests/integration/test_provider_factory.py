"""Integration tests for the LLM provider factory.

These tests verify that the provider factory can properly initialize and manage
different LLM providers, handle configuration changes, and manage provider lifecycle events.
"""

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest

from src.agent.result import Result
from src.exceptions import ConfigError, InvalidModelError
from src.llm_providers.config.provider_config import ProviderConfig
from src.llm_providers.factory import LLMProviderFactory
from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.providers.gemini import GeminiProvider


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Fixture to set up mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test_key",
            "OPENAI_API_KEY": "test_key",
            "ANTHROPIC_API_KEY": "test_key",
        },
    ):
        yield


@pytest.fixture
def provider_factory() -> LLMProviderFactory:
    """Fixture to create a provider factory instance."""
    return LLMProviderFactory()


def test_provider_registration(provider_factory: LLMProviderFactory) -> None:
    """Test that providers can be registered and retrieved correctly."""

    # Register a mock provider
    class MockProvider(BaseLLMProvider):
        def process(self, prompt: str) -> Result[str]:
            return Result.ok("Mock response")

        def process_stream(self, prompt: str) -> Generator[Result[str], None, None]:
            yield Result.ok("Mock stream response")

    provider_factory.register_provider("mock", MockProvider)

    # Verify provider is registered
    assert "mock" in provider_factory._providers
    assert provider_factory._providers["mock"] == MockProvider


def test_provider_configuration(provider_factory: LLMProviderFactory, mock_env_vars: None) -> None:
    """Test provider configuration loading and validation."""
    config = ProviderConfig(
        provider_name="gemini",
        model="gemini-pro",
        temperature=0.7,
        api_key="test_key",
    )

    # Initialize provider
    provider = provider_factory.create_provider("gemini", config)
    assert isinstance(provider, GeminiProvider)
    assert provider.config.model == "gemini-pro"
    assert provider.config.temperature == 0.7


def test_provider_fallback(provider_factory: LLMProviderFactory, mock_env_vars: None) -> None:
    """Test provider fallback mechanism."""
    # Configure primary and fallback providers
    primary_config = ProviderConfig(
        provider_name="invalid_provider",  # This should trigger fallback
        model="invalid-model",
        temperature=0.7,
        api_key="test_key",
    )

    fallback_config = ProviderConfig(
        provider_name="gemini",
        model="gemini-pro",
        temperature=0.7,
        api_key="test_key",
    )

    # Test fallback behavior
    with pytest.raises(ConfigError):
        provider_factory.create_provider("invalid_provider", primary_config)

    # Verify fallback works
    fallback_provider = provider_factory.create_provider("gemini", fallback_config)
    assert isinstance(fallback_provider, GeminiProvider)


def test_invalid_model_handling(provider_factory: LLMProviderFactory, mock_env_vars: None) -> None:
    """Test handling of invalid model configurations."""
    config = ProviderConfig(
        provider_name="gemini",
        model="invalid-model",  # Invalid model name
        temperature=0.7,
        api_key="test_key",
    )

    with pytest.raises(InvalidModelError):
        provider_factory.create_provider("gemini", config)


def test_provider_lifecycle(provider_factory: LLMProviderFactory, mock_env_vars: None) -> None:
    """Test provider lifecycle management."""
    config = ProviderConfig(
        provider_name="gemini",
        model="gemini-pro",
        temperature=0.7,
        api_key="test_key",
    )

    # Initialize provider
    provider = provider_factory.create_provider("gemini", config)

    # Test provider is properly initialized
    assert provider.is_initialized

    # Test provider can be reused with same config
    same_provider = provider_factory.create_provider("gemini", config)
    assert provider is same_provider  # Should return cached instance

    # Test provider with different config creates new instance
    different_config = ProviderConfig(
        provider_name="gemini",
        model="gemini-pro",
        temperature=0.8,  # Different temperature
        api_key="test_key",
    )
    different_provider = provider_factory.create_provider("gemini", different_config)
    assert provider is not different_provider
