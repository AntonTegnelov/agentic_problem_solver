"""Provider lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.exceptions import EmptyResponseError

from .providers.base import BaseLLMProvider
from .version import ProviderVersion

if TYPE_CHECKING:
    from src.llm_providers.providers.base import BaseLLMProvider
    from src.llm_providers.version import ProviderVersion

# Constants
MAX_ERROR_RATE = 0.2  # 20% error rate threshold
MAX_RESPONSE_TIME = 10.0  # 10 seconds response time threshold


class ProviderState(Enum):
    """Provider state."""

    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class HealthStatus:
    """Provider health status."""

    last_check: datetime
    is_healthy: bool
    error_count: int = 0
    last_error: str | None = None
    avg_response_time: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0


@dataclass
class ProviderStats:
    """Provider usage statistics."""

    total_tokens: int = 0
    total_cost: float = 0.0
    requests_per_minute: float = 0.0
    avg_tokens_per_request: float = 0.0
    last_request_time: datetime | None = None


@dataclass
class ProviderLifecycle:
    """Provider lifecycle management."""

    provider: BaseLLMProvider
    version: ProviderVersion
    state: ProviderState = ProviderState.INITIALIZING
    health: HealthStatus = field(
        default_factory=lambda: HealthStatus(datetime.now(timezone.utc), is_healthy=True),
    )
    stats: ProviderStats = field(default_factory=ProviderStats)
    _resources: dict[str, Any] = field(default_factory=dict)

    def initialize(self) -> None:
        """Initialize provider.

        Raises:
            ConfigError: If provider configuration is invalid.

        """
        try:
            self.provider.config.validate()
            self.state = ProviderState.READY
        except Exception as e:
            self.state = ProviderState.ERROR
            self.health.is_healthy = False
            self.health.last_error = str(e)
            raise

    def check_health(self) -> bool:
        """Check provider health.

        Returns:
            True if provider is healthy.

        """
        self.health.last_check = datetime.now(UTC)

        # Check error rate
        if self.health.total_requests > 0:
            error_rate = self.health.failed_requests / self.health.total_requests
            if error_rate > MAX_ERROR_RATE:
                self.health.is_healthy = False
                self.health.last_error = "High error rate"
                return False

        # Check response time
        if self.health.avg_response_time > MAX_RESPONSE_TIME:
            self.health.is_healthy = False
            self.health.last_error = "Slow response time"
            return False

        # Reset error count if all checks pass
        self.health.is_healthy = True
        self.health.error_count = 0
        self.health.last_error = None
        return True

    def update_stats(
        self,
        tokens: int = 0,
        cost: float = 0.0,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Update provider statistics.

        Args:
            tokens: Number of tokens used.
            cost: Cost of request.
            success: Whether request was successful.
            error: Error message if request failed.

        """
        now = datetime.now(timezone.utc)

        # Update request stats
        self.stats.total_tokens += tokens
        self.stats.total_cost += cost
        self.health.total_requests += 1

        if not success:
            self.health.failed_requests += 1
            self.health.error_count += 1
            self.health.last_error = error

        # Update timing stats
        if self.stats.last_request_time:
            time_diff = (now - self.stats.last_request_time).total_seconds()
            if time_diff > 0:
                self.stats.requests_per_minute = 60.0 / time_diff

        self.stats.last_request_time = now

        # Update averages
        if self.health.total_requests > 0:
            self.stats.avg_tokens_per_request = self.stats.total_tokens / self.health.total_requests

    def validate_response(self, response: str | None) -> None:
        """Validate provider response.

        Args:
            response: Provider response.

        Raises:
            EmptyResponseError: If response is empty.

        """
        if not response or not response.strip():
            self.health.error_count += 1
            self.health.failed_requests += 1
            self.health.last_error = "Empty response"
            msg = "Provider returned empty response"
            raise EmptyResponseError(msg)

    def cleanup(self) -> None:
        """Clean up provider resources."""
        try:
            # Clean up any resources
            self._resources.clear()
            self.state = ProviderState.SHUTDOWN
        except Exception as e:
            self.health.last_error = f"Cleanup failed: {e!s}"
            self.state = ProviderState.ERROR
            raise

    def record_usage(
        self,
        tokens: int = 0,
        cost: float = 0.0,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record request statistics.

        Args:
            tokens: Number of tokens used
            cost: Request cost
            success: Whether request was successful
            error: Error message if request failed

        """
        now = datetime.now(timezone.utc)

        # Update request stats
        self.stats.total_requests += 1
        self.stats.total_tokens += tokens
        self.stats.total_cost += cost

        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
            self.stats.last_error = error or "Unknown error"

        # Update request rate
        if self.stats.last_request:
            time_diff = (now - self.stats.last_request).total_seconds()
            if time_diff > 0:
                self.stats.requests_per_minute = 60 / time_diff

        self.stats.last_request = now

    def add_test_resource(self, name: str, value: Any) -> None:
        """Add test resource.

        Args:
            name: Resource name.
            value: Resource value.

        """
        self._resources[name] = value

    def has_resources(self) -> bool:
        """Check if provider has resources.

        Returns:
            True if provider has resources.

        """
        return bool(self._resources)

    @property
    def error_count(self) -> int:
        """Get error count.

        Returns:
            Error count.

        """
        return self.health.error_count
