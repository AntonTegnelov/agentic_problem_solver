"""Test provider lifecycle management."""

from datetime import datetime

import pytest

from src.exceptions import EmptyResponseError
from src.llm_providers.config.provider_config import ProviderConfig
from src.llm_providers.lifecycle import (
    ProviderLifecycle,
    ProviderState,
)
from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.version import ModelVersion, ProviderVersion, Version


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""

    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration."""
        return ProviderConfig(api_key=api_key or "test_key")

    async def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        return "test response"

    async def generate_stream(self, prompt: str) -> str:
        """Generate text from prompt as stream."""
        yield "test response"


def create_test_version() -> ProviderVersion:
    """Create test provider version."""
    return ProviderVersion(
        name="test",
        version=Version(1, 0, 0),
        supported_models={
            "test-model": ModelVersion(
                name="test-model",
                version=Version(1, 0, 0),
                capabilities=["test"],
                min_provider_version=Version(1, 0, 0),
            ),
        },
        default_model="test-model",
    )


def test_provider_lifecycle_initialization() -> None:
    """Test provider lifecycle initialization."""
    provider = MockProvider()
    version = create_test_version()
    lifecycle = ProviderLifecycle(provider, version)

    assert lifecycle.state == ProviderState.INITIALIZING
    assert lifecycle.health.is_healthy
    assert lifecycle.health.error_count == 0
    assert isinstance(lifecycle.health.last_check, datetime)
    assert lifecycle.stats.total_tokens == 0
    assert lifecycle.stats.total_cost == 0.0


def test_provider_lifecycle_health_check() -> None:
    """Test provider health check."""
    provider = MockProvider()
    version = create_test_version()
    lifecycle = ProviderLifecycle(provider, version)

    # Test initial health
    assert lifecycle.check_health()
    assert lifecycle.health.is_healthy

    # Test health with errors
    lifecycle.health.failed_requests = 10
    lifecycle.health.total_requests = 20  # 50% error rate
    assert not lifecycle.check_health()
    assert not lifecycle.health.is_healthy
    assert lifecycle.health.last_error == "High error rate"


def test_provider_lifecycle_stats_update() -> None:
    """Test provider statistics update."""
    provider = MockProvider()
    version = create_test_version()
    lifecycle = ProviderLifecycle(provider, version)

    # Test successful request
    lifecycle.update_stats(tokens=100, cost=0.001, success=True)
    assert lifecycle.stats.total_tokens == 100
    assert lifecycle.stats.total_cost == 0.001
    assert lifecycle.health.total_requests == 1
    assert lifecycle.health.failed_requests == 0

    # Test failed request
    lifecycle.update_stats(tokens=50, cost=0.0005, success=False, error="Test error")
    assert lifecycle.stats.total_tokens == 150
    assert lifecycle.stats.total_cost == 0.0015
    assert lifecycle.health.total_requests == 2
    assert lifecycle.health.failed_requests == 1
    assert lifecycle.health.last_error == "Test error"


def test_provider_lifecycle_response_validation() -> None:
    """Test provider response validation."""
    provider = MockProvider()
    version = create_test_version()
    lifecycle = ProviderLifecycle(provider, version)

    # Test valid response
    lifecycle.validate_response("Valid response")
    assert lifecycle.health.error_count == 0

    # Test empty response
    with pytest.raises(EmptyResponseError):
        lifecycle.validate_response("")
    assert lifecycle.health.error_count == 1
    assert lifecycle.health.failed_requests == 1
    assert lifecycle.health.last_error == "Empty response"


def test_provider_lifecycle_cleanup() -> None:
    """Test provider cleanup."""
    provider = MockProvider()
    version = create_test_version()
    lifecycle = ProviderLifecycle(provider, version)

    # Add some test resources
    lifecycle._resources["test"] = "test"

    # Test cleanup
    lifecycle.cleanup()
    assert not lifecycle._resources
    assert lifecycle.state == ProviderState.SHUTDOWN
