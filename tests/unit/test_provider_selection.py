"""Test provider selection functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.agent.result import Result
from src.common_types import ConfigError, RetryError, TemperatureError
from src.llm_providers.config.provider_config import ProviderConfig
from src.llm_providers.lifecycle import ProviderLifecycle, ProviderState
from src.llm_providers.providers.base import Provider
from src.llm_providers.selection import (
    ProviderCapability,
    ProviderSelector,
)
from src.llm_providers.version import ModelVersion, ProviderVersion, Version
from tests.unit.test_utils import MockGenerationError, MockProcessingError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.common_types.message_types import Message


class MockProvider(Provider):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock", supports_temp: bool = False) -> None:
        """Initialize mock provider.

        Args:
            name: Provider name.
            supports_temp: Whether provider supports temperature.

        """
        super().__init__(name)
        self._supports_temp = supports_temp
        self.should_fail = False
        self.processed_messages: list[Message] = []
        self.config = ProviderConfig(
            provider_name=name,
            temperature=0.7,
            max_tokens=100,
            model="test-model",
            api_key="test-key",
        )

    async def process_message(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        Raises:
            MockProcessingError: If processing fails.

        """
        if self.should_fail:
            msg = f"Error processing message: {message.content}"
            raise MockProcessingError(msg)
        return Result.ok(f"Processed message: {message.content}")

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        Raises:
            MockGenerationError: If streaming fails.

        """
        if self.should_fail:
            msg = f"Error generating streaming response: {message.content}"
            raise MockGenerationError(msg)

        self.processed_messages.append(message)
        yield f"Processed by mock: {message.content}"

    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration.

        Args:
            api_key: Optional API key.

        Returns:
            Provider configuration.

        """
        return ProviderConfig(api_key=api_key or "test_key")

    async def generate(self, prompt: str) -> str:
        """Generate response.

        Args:
            prompt: Prompt to generate response for.

        Returns:
            Generated response.

        Raises:
            MockGenerationError: If should_fail is True.

        """
        if self.should_fail:
            msg = f"Error generating response: {prompt}"
            raise MockGenerationError(msg)
        return f"Response to: {prompt}"

    async def generate_stream(self, prompt: str) -> str:
        """Generate streaming response.

        Args:
            prompt: Prompt to generate response for.

        Returns:
            Generated response.

        Raises:
            MockGenerationError: If should_fail is True.

        """
        if self.should_fail:
            msg = f"Error generating streaming response: {prompt}"
            raise MockGenerationError(msg)
        return f"Streaming response to: {prompt}"

    def supports_temperature(self, _temperature: float) -> bool:
        """Check if provider supports temperature.

        Args:
            _temperature: Temperature to check.

        Returns:
            Whether temperature is supported.

        """
        return self._supports_temp


def create_test_version(capabilities: list[str]) -> ProviderVersion:
    """Create test provider version.

    Args:
        capabilities: List of capabilities to support.

    Returns:
        Provider version.

    """
    return ProviderVersion(
        name="test",
        version=Version(1, 0, 0),
        supported_models={
            "test-model": ModelVersion(
                name="test-model",
                version=Version(1, 0, 0),
                capabilities=capabilities,
                min_provider_version=Version(1, 0, 0),
            ),
        },
        default_model="test-model",
    )


def test_provider_capability_matching() -> None:
    """Test provider capability matching."""
    # Create providers with different capabilities
    provider1 = MockProvider(supports_temp=True)
    provider2 = MockProvider(supports_temp=True)

    version1 = create_test_version(["text", "chat", "function_calling"])
    version2 = create_test_version(["text", "code", "streaming"])

    lifecycle1 = ProviderLifecycle(provider1, version1)
    lifecycle2 = ProviderLifecycle(provider2, version2)

    # Initialize provider states
    lifecycle1.state = ProviderState.READY
    lifecycle2.state = ProviderState.READY

    # Create selector
    selector = ProviderSelector(
        providers={
            "provider1": lifecycle1,
            "provider2": lifecycle2,
        },
        versions={
            "provider1": version1,
            "provider2": version2,
        },
    )

    # Test 1: Basic capability matching - only require text which both providers have
    capabilities = [ProviderCapability("text", required=True)]
    selected = selector.select_provider(capabilities)
    assert selected in [lifecycle1, lifecycle2]  # Either provider is valid

    # Test 2: Match provider with specific capability (chat)
    capabilities = [ProviderCapability("chat", required=True)]
    selected = selector.select_provider(capabilities)
    assert selected == lifecycle1  # Only provider1 has chat capability

    # Test 3: Match provider with specific capability (code)
    capabilities = [ProviderCapability("code", required=True)]
    selected = selector.select_provider(capabilities)
    assert selected == lifecycle2  # Only provider2 has code capability

    # Test 4: Multiple required capabilities
    capabilities = [
        ProviderCapability("text", required=True),
        ProviderCapability("chat", required=True),
    ]
    selected = selector.select_provider(capabilities)
    assert selected == lifecycle1  # Only provider1 has both text and chat

    # Test 5: When no provider matches required capabilities
    capabilities = [ProviderCapability("image", required=True)]
    with pytest.raises(ConfigError):
        selector.select_provider(capabilities)

    # Test 6: Optional capabilities
    capabilities = [
        ProviderCapability("text", required=True),
        ProviderCapability("image", required=False),
    ]
    selected = selector.select_provider(capabilities)
    assert selected in [lifecycle1, lifecycle2]  # Should still match on text

    # Test 7: Provider state affects selection
    lifecycle1.state = ProviderState.ERROR
    capabilities = [ProviderCapability("text", required=True)]
    selected = selector.select_provider(capabilities)
    assert selected == lifecycle2  # Only provider2 is in READY state


def test_provider_temperature_filtering() -> None:
    """Test provider temperature filtering."""
    # Create providers with different temperatures
    provider1 = MockProvider(supports_temp=True)
    provider2 = MockProvider(supports_temp=False)

    version = create_test_version(["text"])

    lifecycle1 = ProviderLifecycle(provider1, version)
    lifecycle2 = ProviderLifecycle(provider2, version)

    # Initialize provider states
    lifecycle1.state = ProviderState.READY
    lifecycle2.state = ProviderState.READY

    # Create selector
    selector = ProviderSelector(
        providers={"p1": lifecycle1, "p2": lifecycle2},
        versions={"p1": version, "p2": version},
    )

    # Test temperature matching
    selected = selector.select_provider(temperature=0.7)
    assert selected == lifecycle1

    # Test when no provider matches temperature
    selector = ProviderSelector(
        providers={"p2": lifecycle2},
        versions={"p2": version},
    )
    with pytest.raises(TemperatureError):
        selector.select_provider(temperature=0.7)


def test_provider_fallback_chain() -> None:
    """Test provider fallback chain."""
    # Create providers
    provider1 = MockProvider(supports_temp=True)
    provider2 = MockProvider(supports_temp=True)

    version = create_test_version(["text"])

    lifecycle1 = ProviderLifecycle(provider1, version)
    lifecycle2 = ProviderLifecycle(provider2, version)

    # Initialize provider states
    lifecycle1.state = ProviderState.ERROR  # First provider is unhealthy
    lifecycle2.state = ProviderState.READY

    # Create selector with fallback chain
    selector = ProviderSelector(
        providers={"p1": lifecycle1, "p2": lifecycle2},
        versions={"p1": version, "p2": version},
        fallback_chain=["p1", "p2"],
    )

    # Test fallback
    fallback = selector.get_fallback_provider()
    assert fallback == lifecycle2  # Should skip unhealthy provider1

    # Test when all providers exhausted
    selector.reset_fallback_chain()
    lifecycle2.state = ProviderState.ERROR
    with pytest.raises(RetryError):
        selector.get_fallback_provider()


def test_provider_load_balancing() -> None:
    """Test provider load balancing."""
    # Create providers
    provider1 = MockProvider(name="provider1", supports_temp=True)
    provider2 = MockProvider(name="provider2", supports_temp=True)
    provider3 = MockProvider(name="provider3", supports_temp=True)

    version = create_test_version(["text"])

    lifecycle1 = ProviderLifecycle(provider1, version)
    lifecycle2 = ProviderLifecycle(provider2, version)
    lifecycle3 = ProviderLifecycle(provider3, version)

    # Initialize provider states
    lifecycle1.state = ProviderState.READY
    lifecycle2.state = ProviderState.READY
    lifecycle3.state = ProviderState.READY

    # Create selector
    selector = ProviderSelector(
        providers={
            "provider1": lifecycle1,
            "provider2": lifecycle2,
            "provider3": lifecycle3,
        },
        versions={
            "provider1": version,
            "provider2": version,
            "provider3": version,
        },
    )

    # Test 1: Initial state - no load distribution
    selected = selector.select_provider()
    assert selected in [lifecycle1, lifecycle2, lifecycle3]  # Any provider is valid

    # Test 2: Load distribution affects selection
    selector.update_load_distribution("provider1", 10.0)  # High load
    selector.update_load_distribution("provider2", 5.0)  # Medium load
    selector.update_load_distribution("provider3", 1.0)  # Low load

    selected = selector.select_provider()
    assert selected.provider.name == "provider3"  # Should select least loaded provider

    # Test 3: Health affects selection more than load
    lifecycle3.health.error_count = 5  # Make provider3 unhealthy
    selected = selector.select_provider()
    assert selected.provider.name == "provider2"  # Should select provider2 (medium load but healthy)

    # Test 4: Both health and load affect selection
    lifecycle2.health.error_count = 3  # Make provider2 somewhat unhealthy
    selector.update_load_distribution(
        "provider1",
        1.0,
    )  # Update provider1 to low load
    selected = selector.select_provider()
    assert selected == lifecycle1  # Should select provider1 (now has low load and is healthy)

    # Test 5: Reset load distribution
    selector.update_load_distribution("provider1", 0.0)
    selector.update_load_distribution("provider2", 0.0)
    selector.update_load_distribution("provider3", 0.0)
    selected = selector.select_provider()
    assert selected == lifecycle1  # Should still select provider1 (healthiest)
