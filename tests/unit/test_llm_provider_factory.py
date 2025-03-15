"""Unit tests for the LLM provider factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.common_types.error_types import ConfigError, InvalidModelError
from src.llm_providers.config.provider_config import ProviderConfig
from src.llm_providers.factory import LLMProviderFactory, ProviderNotFoundError
from src.llm_providers.lifecycle import ProviderLifecycle, ProviderState
from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.version import ModelVersion, ProviderVersion, Version


class MockProviderConfig(ProviderConfig):
    """Mock provider configuration for testing."""

    PROVIDER_VERSION = ProviderVersion(
        name="mock",
        version=Version(1, 0, 0),
        supported_models={
            "mock-model": ModelVersion(
                name="mock-model",
                version=Version(1, 0, 0),
                capabilities=["text-generation"],
                min_provider_version=Version(1, 0, 0),
            ),
        },
        default_model="mock-model",
    )


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""

    PROVIDER_VERSION = MockProviderConfig.PROVIDER_VERSION

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize provider."""
        self.config = config or self._create_config("test_key")
        self.is_initialized = True

    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration."""
        return MockProviderConfig(
            provider_name="mock",
            model="mock-model",
            api_key=api_key or "test_key",
        )

    def generate(self, _prompt: str | list) -> str:
        """Generate response."""
        return "Mock response"

    async def generate_stream(self, _prompt: str | list) -> list[str]:
        """Generate streaming response."""
        yield "Mock stream response"


@pytest.fixture
def reset_factory() -> None:
    """Reset the factory singleton state between tests."""
    # Store original class variables
    original_providers = LLMProviderFactory._providers.copy()
    original_provider_lifecycles = LLMProviderFactory._provider_lifecycles.copy()
    original_provider_configs = LLMProviderFactory._provider_configs.copy()
    original_provider_versions = LLMProviderFactory._provider_versions.copy()
    original_current_provider = LLMProviderFactory._current_provider
    original_provider_name = LLMProviderFactory._provider_name
    original_selector = LLMProviderFactory._selector
    original_initialized = LLMProviderFactory._initialized

    # Reset class variables for testing
    LLMProviderFactory._providers = {}
    LLMProviderFactory._provider_lifecycles = {}
    LLMProviderFactory._provider_configs = {}
    LLMProviderFactory._provider_versions = {}
    LLMProviderFactory._current_provider = None
    LLMProviderFactory._provider_name = None
    LLMProviderFactory._selector = None
    LLMProviderFactory._initialized = False

    yield

    # Restore original class variables after test
    LLMProviderFactory._providers = original_providers
    LLMProviderFactory._provider_lifecycles = original_provider_lifecycles
    LLMProviderFactory._provider_configs = original_provider_configs
    LLMProviderFactory._provider_versions = original_provider_versions
    LLMProviderFactory._current_provider = original_current_provider
    LLMProviderFactory._provider_name = original_provider_name
    LLMProviderFactory._selector = original_selector
    LLMProviderFactory._initialized = original_initialized


class TestLLMProviderFactory:
    """Unit tests for the LLM provider factory."""

    def test_singleton_pattern(self) -> None:
        """Test that the factory follows the singleton pattern."""
        factory1 = LLMProviderFactory()
        factory2 = LLMProviderFactory()
        assert factory1 is factory2

    @pytest.mark.usefixtures("reset_factory")
    def test_validate_provider_class(self) -> None:
        """Test provider class validation."""
        # Valid provider class
        LLMProviderFactory._validate_provider_class("mock", MockProvider)

        # Invalid provider class (not a subclass of BaseLLMProvider)
        class InvalidProvider:
            pass

        with pytest.raises(InvalidModelError):
            LLMProviderFactory._validate_provider_class("invalid", InvalidProvider)  # type: ignore[arg-type]

        # Provider already registered
        LLMProviderFactory._providers["mock"] = MockProvider
        with pytest.raises(ConfigError):
            LLMProviderFactory._validate_provider_class("mock", MockProvider)

    @pytest.mark.usefixtures("reset_factory")
    def test_get_provider_version(self) -> None:
        """Test getting provider version."""
        # Register provider with version
        LLMProviderFactory.register_provider("mock", MockProvider, MockProvider.PROVIDER_VERSION)

        # Get version
        version = LLMProviderFactory.get_provider_version("mock")
        assert version == MockProvider.PROVIDER_VERSION

        # Provider not found
        with pytest.raises(ProviderNotFoundError):
            LLMProviderFactory.get_provider_version("nonexistent")

    @pytest.mark.usefixtures("reset_factory")
    def test_get_current_provider_name(self) -> None:
        """Test getting current provider name."""
        # No provider set
        assert LLMProviderFactory.get_current_provider_name() is None

        # Set provider name
        LLMProviderFactory._provider_name = "mock"
        assert LLMProviderFactory.get_current_provider_name() == "mock"

    @pytest.mark.usefixtures("reset_factory")
    @patch("src.llm_providers.factory.ConfigError")
    def test_validate_provider_health(self, mock_config_error: MagicMock) -> None:
        """Test provider health validation."""
        factory = LLMProviderFactory()

        # Create mock lifecycle
        lifecycle = MagicMock(spec=ProviderLifecycle)

        # Test ready state
        lifecycle.state = ProviderState.READY
        factory._validate_provider_health(lifecycle, "mock")

        # Test initializing state
        lifecycle.state = ProviderState.INITIALIZING
        factory._validate_provider_health(lifecycle, "mock")
        mock_config_error.assert_not_called()

        # Test error state
        lifecycle.state = ProviderState.ERROR
        lifecycle.error = "Test error"
        factory._validate_provider_health(lifecycle, "mock")
        mock_config_error.assert_not_called()

    @pytest.mark.usefixtures("reset_factory")
    def test_reset_fallback_chain(self) -> None:
        """Test resetting the fallback chain."""
        factory = LLMProviderFactory()

        # Set up selector mock
        factory._selector = MagicMock()

        # Test reset
        factory.reset_fallback_chain()

        # Verify selector reset_fallback_chain was called
        factory._selector.reset_fallback_chain.assert_called_once()
