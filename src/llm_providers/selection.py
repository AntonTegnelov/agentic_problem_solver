"""Provider selection and routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.exceptions import ConfigError, RetryError, TemperatureError
from src.llm_providers.lifecycle import ProviderLifecycle, ProviderState

if TYPE_CHECKING:
    from src.llm_providers.version import ProviderVersion

# Constants
TEMPERATURE_TOLERANCE = 0.1  # Maximum allowed difference in temperature


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
        """Select provider based on capabilities and health.

        Args:
            capabilities: Required capabilities.
            temperature: Required temperature.

        Returns:
            Selected provider.

        Raises:
            ConfigError: If no provider found.
            TemperatureError: If no provider supports temperature.

        """
        # Filter by capabilities
        candidates = []
        for provider_name, lifecycle in self.providers.items():
            if lifecycle.state != ProviderState.READY:
                continue

            version = self.versions[provider_name]
            model_version = version.supported_models[version.default_model]

            # Check if provider has all required capabilities
            has_all_capabilities = True
            if capabilities:
                for capability in capabilities:
                    if capability.required and capability.name not in model_version.capabilities:
                        has_all_capabilities = False
                        break

            if has_all_capabilities:
                candidates.append(lifecycle)

        if not candidates:
            msg = "No provider found supporting required capabilities"
            raise ConfigError(msg)

        # Filter by temperature
        if temperature is not None:
            temp_candidates = [
                p for p in candidates if p.provider.supports_temperature(temperature)
            ]
            if not temp_candidates:
                msg = f"No provider found supporting temperature {temperature}"
                raise TemperatureError(msg)
            candidates = temp_candidates

        # Sort by health (lower error count is better) and load (lower load is better)
        candidates.sort(
            key=lambda p: (
                p.error_count,
                self._load_distribution.get(p.provider.name, 0.0),
            ),
        )

        return candidates[0]

    def _filter_by_capabilities(
        self,
        capabilities: list[str],
    ) -> list[ProviderLifecycle]:
        """Filter providers by capabilities.

        Args:
            capabilities: Required capabilities.

        Returns:
            List of providers with required capabilities.

        """
        candidates = []
        for provider_name, lifecycle in self.providers.items():
            if lifecycle.state != ProviderState.READY:
                continue

            version = self.versions[provider_name]
            model_version = version.supported_models[version.default_model]

            # If no capabilities required, include all ready providers
            if not capabilities:
                candidates.append(lifecycle)
                continue

            # Check if provider has all required capabilities
            has_all_capabilities = True
            for capability in capabilities:
                if capability not in model_version.capabilities:
                    has_all_capabilities = False
                    break

            if has_all_capabilities:
                candidates.append(lifecycle)

        return candidates

    def _filter_by_temperature(
        self,
        providers: list[ProviderLifecycle],
        temperature: float,
    ) -> list[ProviderLifecycle]:
        """Filter providers by temperature.

        Args:
            providers: List of provider lifecycles.
            temperature: Target temperature.

        Returns:
            List of providers matching temperature.

        """
        result = []
        for lifecycle in providers:
            if hasattr(lifecycle.provider.config, "temperature"):
                config_temp = lifecycle.provider.config.temperature
                if (
                    0 <= temperature <= 1
                    and abs(config_temp - temperature) <= TEMPERATURE_TOLERANCE
                ):
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
