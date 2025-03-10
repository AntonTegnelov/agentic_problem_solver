"""Provider selection and routing."""

from dataclasses import dataclass, field
from typing import Any

from src.exceptions import ConfigError, RetryError, TemperatureError
from src.llm_providers.lifecycle import ProviderLifecycle, ProviderState
from src.llm_providers.version import ProviderVersion


@dataclass
class ProviderCapability:
    """Provider capability information."""

    name: str
    required: bool = True
    min_version: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderSelector:
    """Provider selection and routing."""

    providers: dict[str, ProviderLifecycle]
    versions: dict[str, ProviderVersion]
    fallback_chain: list[str] = field(default_factory=list)
    _current_fallback_index: int = 0
    _load_distribution: dict[str, float] = field(default_factory=dict)

    def select_provider(
        self,
        capabilities: list[ProviderCapability] | None = None,
        temperature: float | None = None,
    ) -> ProviderLifecycle:
        """Select best provider based on capabilities and health.

        Args:
            capabilities: Required capabilities.
            temperature: Required temperature setting.

        Returns:
            Selected provider lifecycle.

        Raises:
            ConfigError: If no suitable provider found.
            TemperatureError: If temperature requirements not met.

        """
        candidates = self._filter_by_capabilities(capabilities or [])
        if not candidates:
            msg = "No provider found with required capabilities"
            raise ConfigError(msg)

        # Check temperature requirements
        if temperature is not None:
            candidates = self._filter_by_temperature(candidates, temperature)
            if not candidates:
                msg = f"No provider supports temperature {temperature}"
                raise TemperatureError(msg)

        # Sort by health and load
        candidates.sort(
            key=lambda x: (
                x.health.is_healthy,
                -x.health.error_count,
                -self._load_distribution.get(x.provider.__class__.__name__, 0.0),
            ),
            reverse=True,
        )

        return candidates[0]

    def _filter_by_capabilities(
        self,
        capabilities: list[ProviderCapability],
    ) -> list[ProviderLifecycle]:
        """Filter providers by required capabilities.

        Args:
            capabilities: Required capabilities.

        Returns:
            List of providers supporting all required capabilities.

        """
        result = []
        for lifecycle in self.providers.values():
            if lifecycle.state != ProviderState.READY:
                continue

            version = self.versions.get(lifecycle.provider.__class__.__name__)
            if not version:
                continue

            supports_all = True
            for cap in capabilities:
                if cap.required:
                    # Check capability support
                    if not version.supports_capability(cap.name):
                        supports_all = False
                        break

                    # Check version requirement
                    if cap.min_version and version.version < cap.min_version:
                        supports_all = False
                        break

            if supports_all:
                result.append(lifecycle)

        return result

    def _filter_by_temperature(
        self,
        providers: list[ProviderLifecycle],
        temperature: float,
    ) -> list[ProviderLifecycle]:
        """Filter providers by temperature requirement.

        Args:
            providers: List of providers to filter.
            temperature: Required temperature.

        Returns:
            List of providers supporting the temperature.

        """
        result = []
        for lifecycle in providers:
            if hasattr(lifecycle.provider.config, "temperature"):
                config_temp = lifecycle.provider.config.temperature
                if 0 <= temperature <= 1 and abs(config_temp - temperature) <= 0.1:
                    result.append(lifecycle)
        return result

    def update_load_distribution(self, provider_name: str, load: float) -> None:
        """Update provider load distribution.

        Args:
            provider_name: Provider name.
            load: Current load (requests per minute).

        """
        self._load_distribution[provider_name] = load

    def get_fallback_provider(self) -> ProviderLifecycle | None:
        """Get next provider in fallback chain.

        Returns:
            Next provider in fallback chain or None if exhausted.

        Raises:
            RetryError: If all fallbacks exhausted.

        """
        while self._current_fallback_index < len(self.fallback_chain):
            provider_name = self.fallback_chain[self._current_fallback_index]
            self._current_fallback_index += 1

            if provider_name in self.providers:
                lifecycle = self.providers[provider_name]
                if lifecycle.state == ProviderState.READY and lifecycle.check_health():
                    return lifecycle

        self._current_fallback_index = 0  # Reset for next time
        msg = "All fallback providers exhausted"
        raise RetryError(msg)

    def reset_fallback_chain(self) -> None:
        """Reset fallback chain index."""
        self._current_fallback_index = 0
