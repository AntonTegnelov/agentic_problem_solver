"""Test provider selection system."""

import pytest

from src.agent.agent_types.agent_types import Message, StepResult
from src.exceptions import ConfigError, RetryError, TemperatureError
from src.llm_providers.config.provider_config import ProviderConfig
from src.llm_providers.lifecycle import ProviderLifecycle, ProviderState
from src.llm_providers.selection import ProviderCapability, ProviderSelector
from src.llm_providers.version import ModelVersion, ProviderVersion, Version


class MockProvider:
    """Mock provider for testing."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize mock agent.

        Args:
            should_fail: Whether agent should fail processing.

        """
        self.should_fail = should_fail
        self.processed_messages: list[Message] = []

    async def process(self, message: Message) -> StepResult:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Step result.

        Raises:
            Exception: If should_fail is True.

        """
        if self.should_fail:
            msg = "Processing failed"
            raise Exception(msg)
        self.processed_messages.append(message)
        return StepResult(success=True, message="Success")

    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration.

        Args:
            api_key: Optional API key.

        Returns:
            Provider configuration.

        """
        return ProviderConfig(api_key=api_key or "test_key")

    async def generate(self, prompt: str) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.

        """
        if self.should_fail:
            msg = "Generation failed"
            raise Exception(msg)
        return f"Response to: {prompt}"

    async def generate_stream(self, prompt: str) -> str:
        """Generate text from prompt as stream.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text stream.

        """
        if self.should_fail:
            msg = "Generation failed"
            raise Exception(msg)
        yield f"Streaming response to: {prompt}"


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
    provider1 = MockProvider()
    provider2 = MockProvider()

    version1 = create_test_version(["text", "chat"])
    version2 = create_test_version(["text", "code"])

    lifecycle1 = ProviderLifecycle(provider1, version1)
    lifecycle2 = ProviderLifecycle(provider2, version2)

    # Initialize provider states
    lifecycle1.state = ProviderState.READY
    lifecycle2.state = ProviderState.READY

    # Create selector
    selector = ProviderSelector(
        providers={"p1": lifecycle1, "p2": lifecycle2},
        versions={"p1": version1, "p2": version2},
    )

    # Test capability matching
    capabilities = [
        ProviderCapability("text", required=True),
        ProviderCapability("chat", required=True),
    ]
    selected = selector.select_provider(capabilities)
    assert selected == lifecycle1  # Only provider1 has both text and chat

    # Test when no provider matches
    capabilities = [ProviderCapability("image", required=True)]
    with pytest.raises(ConfigError):
        selector.select_provider(capabilities)


def test_provider_temperature_filtering() -> None:
    """Test provider temperature filtering."""
    # Create providers with different temperatures
    provider1 = MockProvider()
    provider2 = MockProvider()

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
    with pytest.raises(TemperatureError):
        selector.select_provider(temperature=0.1)


def test_provider_fallback_chain() -> None:
    """Test provider fallback chain."""
    # Create providers
    provider1 = MockProvider()
    provider2 = MockProvider()

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
    provider1 = MockProvider()
    provider2 = MockProvider()

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

    # Update load distribution
    selector.update_load_distribution("p1", 100.0)  # High load
    selector.update_load_distribution("p2", 10.0)  # Low load

    # Test load-based selection
    selected = selector.select_provider()
    assert selected == lifecycle2  # Should select less loaded provider
