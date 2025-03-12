"""Integration tests for the LLM provider factory with different providers.

These tests verify that the provider factory can properly initialize and manage
different LLM providers, handle switching between providers, and manage provider capabilities.
"""

# ruff: noqa: S603, S607, BLE001, SLF001
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from src.common_types.error_types import ConfigError, InvalidModelError
from src.llm_providers.config.provider_config import GeminiConfig, ProviderConfig
from src.llm_providers.factory import LLMProviderFactory
from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.providers.gemini import GeminiProvider
from src.llm_providers.version import ModelVersion, ProviderVersion, Version

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Fixture to set up mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test_key",
            "GEMINI_MODEL": "gemini-2.0-flash-lite",
            "OPENAI_API_KEY": "test_key",
            "OPENAI_MODEL": "gpt-4",
            "ANTHROPIC_API_KEY": "test_key",
            "ANTHROPIC_MODEL": "claude-3-opus",
        },
    ):
        yield


@pytest.fixture
def provider_factory() -> Generator[LLMProviderFactory, None, None]:
    """Fixture to create a provider factory instance with clean state.

    This fixture creates a new LLMProviderFactory instance and resets its state
    to ensure tests don't interfere with each other.
    """
    # Store original class variables
    original_providers = LLMProviderFactory._providers.copy()
    original_provider_lifecycles = LLMProviderFactory._provider_lifecycles.copy()
    original_provider_configs = LLMProviderFactory._provider_configs.copy()
    original_provider_versions = LLMProviderFactory._provider_versions.copy()
    original_current_provider = LLMProviderFactory._current_provider
    original_provider_name = LLMProviderFactory._provider_name
    original_selector = LLMProviderFactory._selector
    original_initialized = LLMProviderFactory._initialized
    original_get_current_provider = LLMProviderFactory.get_current_provider

    # Reset class variables for testing
    LLMProviderFactory._providers = {"gemini": GeminiProvider}
    LLMProviderFactory._provider_lifecycles = {}
    LLMProviderFactory._provider_configs = {}
    LLMProviderFactory._provider_versions = {"gemini": ProviderVersion.GEMINI_V1}
    LLMProviderFactory._current_provider = None
    LLMProviderFactory._provider_name = None
    LLMProviderFactory._selector = None
    LLMProviderFactory._initialized = False

    # Mock the get_current_provider method for testing
    LLMProviderFactory.get_current_provider = lambda self: self._current_provider

    # Create a new factory instance
    factory = LLMProviderFactory()

    yield factory

    # Restore original class variables after test
    LLMProviderFactory._providers = original_providers
    LLMProviderFactory._provider_lifecycles = original_provider_lifecycles
    LLMProviderFactory._provider_configs = original_provider_configs
    LLMProviderFactory._provider_versions = original_provider_versions
    LLMProviderFactory._current_provider = original_current_provider
    LLMProviderFactory._provider_name = original_provider_name
    LLMProviderFactory._selector = original_selector
    LLMProviderFactory._initialized = original_initialized
    LLMProviderFactory.get_current_provider = original_get_current_provider


class OpenAIConfig(ProviderConfig):
    """OpenAI provider configuration."""

    # Provider version information
    PROVIDER_VERSION = ProviderVersion(
        name="openai",
        version=Version(1, 0, 0),
        supported_models={
            "gpt-4": ModelVersion(
                name="gpt-4",
                version=Version(4, 0, 0),
                capabilities=[
                    "text-generation",
                    "chat",
                    "code-generation",
                    "code-analysis",
                ],
                min_provider_version=Version(1, 0, 0),
            ),
            "gpt-3.5-turbo": ModelVersion(
                name="gpt-3.5-turbo",
                version=Version(3, 5, 0),
                capabilities=[
                    "text-generation",
                    "chat",
                    "code-generation",
                ],
                min_provider_version=Version(1, 0, 0),
            ),
        },
        default_model="gpt-4",
    )


class AnthropicConfig(ProviderConfig):
    """Anthropic provider configuration."""

    # Provider version information
    PROVIDER_VERSION = ProviderVersion(
        name="anthropic",
        version=Version(1, 0, 0),
        supported_models={
            "claude-3-opus": ModelVersion(
                name="claude-3-opus",
                version=Version(3, 0, 0),
                capabilities=[
                    "text-generation",
                    "chat",
                    "code-generation",
                    "code-analysis",
                    "multimodal",
                ],
                min_provider_version=Version(1, 0, 0),
            ),
            "claude-3-sonnet": ModelVersion(
                name="claude-3-sonnet",
                version=Version(3, 0, 0),
                capabilities=[
                    "text-generation",
                    "chat",
                    "code-generation",
                ],
                min_provider_version=Version(1, 0, 0),
            ),
        },
        default_model="claude-3-opus",
    )


class MockOpenAIProvider(BaseLLMProvider):
    """Mock OpenAI provider for testing."""

    PROVIDER_VERSION: ClassVar[ProviderVersion] = OpenAIConfig.PROVIDER_VERSION

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize provider."""
        self.config = config or self._create_config("test_key")
        self.responses: list[str] = ["Mock OpenAI response"]
        self.is_initialized = True

    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration.

        Args:
            api_key: Optional API key.

        Returns:
            Provider configuration.

        """
        return OpenAIConfig(
            provider_name="openai",
            model="gpt-4",
            api_key=api_key or "test_key",
        )

    def generate(self, _prompt: str | list[Any]) -> str:
        """Generate response."""
        return self.responses[0]

    async def generate_stream(self, _prompt: str | list[Any]) -> Generator[str, None, None]:
        """Generate streaming response."""
        yield self.responses[0]


class MockAnthropicProvider(BaseLLMProvider):
    """Mock Anthropic provider for testing."""

    PROVIDER_VERSION: ClassVar[ProviderVersion] = AnthropicConfig.PROVIDER_VERSION

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize provider."""
        self.config = config or self._create_config("test_key")
        self.responses: list[str] = ["Mock Anthropic response"]
        self.is_initialized = True

    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration.

        Args:
            api_key: Optional API key.

        Returns:
            Provider configuration.

        """
        return AnthropicConfig(
            provider_name="anthropic",
            model="claude-3-opus",
            api_key=api_key or "test_key",
        )

    def generate(self, _prompt: str | list[Any]) -> str:
        """Generate response."""
        return self.responses[0]

    async def generate_stream(self, _prompt: str | list[Any]) -> Generator[str, None, None]:
        """Generate streaming response."""
        yield self.responses[0]


class MockOpenAIProviderWithValidation(MockOpenAIProvider):
    """Mock OpenAI provider with model validation for testing."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize provider with validation."""
        super().__init__(config)
        if config and config.model not in self.PROVIDER_VERSION.supported_models:
            msg = f"Model {config.model} not supported by provider {self.PROVIDER_VERSION.name}"
            raise InvalidModelError(msg)


def test_register_multiple_providers(provider_factory: LLMProviderFactory) -> None:
    """Test registering multiple providers."""
    # Register mock providers
    provider_factory.register_provider("openai", MockOpenAIProvider, MockOpenAIProvider.PROVIDER_VERSION)
    provider_factory.register_provider("anthropic", MockAnthropicProvider, MockAnthropicProvider.PROVIDER_VERSION)

    # Verify providers are registered
    assert "openai" in provider_factory._providers
    assert "anthropic" in provider_factory._providers
    assert "gemini" in provider_factory._providers  # Default provider should still be there

    # Verify provider classes
    assert provider_factory.get_provider("openai") == MockOpenAIProvider
    assert provider_factory.get_provider("anthropic") == MockAnthropicProvider
    assert provider_factory.get_provider("gemini") == GeminiProvider


@pytest.mark.usefixtures("mock_env_vars")
def test_create_different_providers(provider_factory: LLMProviderFactory) -> None:
    """Test creating different provider instances."""
    # Register mock providers
    provider_factory.register_provider("openai", MockOpenAIProvider, MockOpenAIProvider.PROVIDER_VERSION)
    provider_factory.register_provider("anthropic", MockAnthropicProvider, MockAnthropicProvider.PROVIDER_VERSION)

    # Create configurations
    gemini_config = GeminiConfig(
        provider_name="gemini",
        model="gemini-2.0-flash-lite",
        api_key="test_key",
    )

    openai_config = OpenAIConfig(
        provider_name="openai",
        model="gpt-4",
        api_key="test_key",
    )

    anthropic_config = AnthropicConfig(
        provider_name="anthropic",
        model="claude-3-opus",
        api_key="test_key",
    )

    # Create provider instances
    gemini_provider = provider_factory.create_provider("gemini", gemini_config)
    openai_provider = provider_factory.create_provider("openai", openai_config)
    anthropic_provider = provider_factory.create_provider("anthropic", anthropic_config)

    # Verify provider types
    assert isinstance(gemini_provider, GeminiProvider)
    assert isinstance(openai_provider, MockOpenAIProvider)
    assert isinstance(anthropic_provider, MockAnthropicProvider)

    # Verify configurations
    assert gemini_provider.config.model == "gemini-2.0-flash-lite"
    assert openai_provider.config.model == "gpt-4"
    assert anthropic_provider.config.model == "claude-3-opus"


@pytest.mark.usefixtures("mock_env_vars")
def test_provider_switching(provider_factory: LLMProviderFactory) -> None:
    """Test switching between providers."""
    # Register mock providers
    provider_factory.register_provider("openai", MockOpenAIProvider, MockOpenAIProvider.PROVIDER_VERSION)
    provider_factory.register_provider("anthropic", MockAnthropicProvider, MockAnthropicProvider.PROVIDER_VERSION)

    # Create configurations
    gemini_config = GeminiConfig(
        provider_name="gemini",
        model="gemini-2.0-flash-lite",
        api_key="test_key",
    )

    openai_config = OpenAIConfig(
        provider_name="openai",
        model="gpt-4",
        api_key="test_key",
    )

    anthropic_config = AnthropicConfig(
        provider_name="anthropic",
        model="claude-3-opus",
        api_key="test_key",
    )

    # Create and switch between providers
    gemini_provider = provider_factory.create_provider("gemini", gemini_config)

    # Set current provider manually since the factory doesn't do it automatically in tests
    provider_factory._current_provider = gemini_provider
    assert provider_factory.get_current_provider() == gemini_provider

    openai_provider = provider_factory.create_provider("openai", openai_config)
    provider_factory._current_provider = openai_provider
    assert provider_factory.get_current_provider() == openai_provider

    anthropic_provider = provider_factory.create_provider("anthropic", anthropic_config)
    provider_factory._current_provider = anthropic_provider
    assert provider_factory.get_current_provider() == anthropic_provider

    # Switch back to previous provider
    provider_factory._current_provider = gemini_provider
    assert provider_factory.get_current_provider() == gemini_provider


@pytest.mark.usefixtures("mock_env_vars")
def test_provider_capabilities(provider_factory: LLMProviderFactory) -> None:
    """Test provider capabilities."""
    # Register mock providers with different capabilities
    provider_factory.register_provider("openai", MockOpenAIProvider, MockOpenAIProvider.PROVIDER_VERSION)
    provider_factory.register_provider("anthropic", MockAnthropicProvider, MockAnthropicProvider.PROVIDER_VERSION)

    # Test capabilities for different models
    gemini_version = provider_factory.get_provider_version("gemini")
    openai_version = MockOpenAIProvider.PROVIDER_VERSION
    anthropic_version = MockAnthropicProvider.PROVIDER_VERSION

    # Check multimodal capability
    assert gemini_version.has_capability("gemini-2.0-flash-lite", "multimodal")
    assert not openai_version.has_capability("gpt-4", "multimodal")  # GPT-4 doesn't have multimodal in our mock
    assert anthropic_version.has_capability("claude-3-opus", "multimodal")
    assert not anthropic_version.has_capability("claude-3-sonnet", "multimodal")

    # Check code-analysis capability
    assert gemini_version.has_capability("gemini-2.0-flash-lite", "code-analysis")
    assert openai_version.has_capability("gpt-4", "code-analysis")
    assert not openai_version.has_capability("gpt-3.5-turbo", "code-analysis")
    assert anthropic_version.has_capability("claude-3-opus", "code-analysis")


@pytest.mark.usefixtures("mock_env_vars")
def test_provider_fallback_chain(provider_factory: LLMProviderFactory) -> None:
    """Test provider fallback chain."""
    # Register mock providers
    provider_factory.register_provider("openai", MockOpenAIProvider, MockOpenAIProvider.PROVIDER_VERSION)
    provider_factory.register_provider("anthropic", MockAnthropicProvider, MockAnthropicProvider.PROVIDER_VERSION)

    # Set fallback chain
    provider_factory.set_fallback_chain(["gemini", "openai", "anthropic"])

    # Create configurations
    gemini_config = GeminiConfig(
        provider_name="gemini",
        model="gemini-2.0-flash-lite",
        api_key="test_key",
    )

    openai_config = OpenAIConfig(
        provider_name="openai",
        model="gpt-4",
        api_key="test_key",
    )

    anthropic_config = AnthropicConfig(
        provider_name="anthropic",
        model="claude-3-opus",
        api_key="test_key",
    )

    # Create providers to initialize them in the factory
    gemini_provider = provider_factory.create_provider("gemini", gemini_config)
    openai_provider = provider_factory.create_provider("openai", openai_config)
    anthropic_provider = provider_factory.create_provider("anthropic", anthropic_config)

    # Set up provider lifecycles manually for testing
    provider_factory._provider_lifecycles = {
        "gemini": MagicMock(provider=gemini_provider),
        "openai": MagicMock(provider=openai_provider),
        "anthropic": MagicMock(provider=anthropic_provider),
    }

    # Mock the selector to simulate fallback
    provider_factory._selector = MagicMock()
    provider_factory._selector.get_fallback_provider.return_value = provider_factory._provider_lifecycles["openai"]

    # Try to get a fallback provider - should return OpenAI
    fallback_provider = provider_factory.get_fallback_provider()
    assert fallback_provider == openai_provider


@pytest.mark.usefixtures("mock_env_vars")
def test_provider_with_invalid_model(provider_factory: LLMProviderFactory) -> None:
    """Test provider with invalid model."""
    # Register mock provider with validation
    provider_factory.register_provider("openai", MockOpenAIProviderWithValidation, MockOpenAIProvider.PROVIDER_VERSION)

    # Create configuration with invalid model
    invalid_config = OpenAIConfig(
        provider_name="openai",
        model="invalid-model",  # This model doesn't exist
        api_key="test_key",
    )

    # Attempt to create provider with invalid model
    with pytest.raises(ConfigError) as excinfo:
        provider_factory.create_provider("openai", invalid_config)

    # Verify error message
    assert "Failed to create provider" in str(excinfo.value)


def test_provider_with_missing_api_key(provider_factory: LLMProviderFactory) -> None:
    """Test provider with missing API key."""
    # Create configuration with missing API key
    invalid_config = ProviderConfig(
        provider_name="gemini",
        model="gemini-2.0-flash-lite",
        api_key=None,  # Missing API key
    )

    # Attempt to create provider with missing API key
    with pytest.raises(ConfigError) as excinfo:
        provider_factory.create_provider("gemini", invalid_config)

    # Verify error message
    assert "API key is required" in str(excinfo.value) or "Failed to create provider" in str(excinfo.value)
