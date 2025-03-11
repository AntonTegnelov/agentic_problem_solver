"""LLM provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from typing_extensions import Self

from src.config.utils import load_config_from_env
from src.exceptions import (
    APIKeyError,
    ConfigError,
    EmptyResponseError,
    InvalidModelError,
    RetryError,
)
from src.llm_providers.lifecycle import ProviderLifecycle
from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.providers.gemini import GeminiProvider
from src.llm_providers.selection import ProviderSelector
from src.llm_providers.version import ProviderVersion

if TYPE_CHECKING:
    from src.llm_providers.config.provider_config import ProviderConfig

# Error messages
PROVIDER_NOT_FOUND = "Provider {name} not found"
API_KEY_REQUIRED = "API key is required"
NO_PROVIDER_SET = "No provider set"
UNSUPPORTED_PROVIDER = "Unsupported provider: {}"
INVALID_PROVIDER = "Invalid provider class: {name}. Must implement BaseLLMProvider"
PROVIDER_EXISTS = "Provider {name} already registered"
PROVIDER_CONFIG_ERROR = "Failed to create config for provider {name}: {error}"
VERSION_MISMATCH = "Provider {name} version {version} does not match required version {required}"
PROVIDER_NOT_READY = "Provider {name} is not ready (state: {state})"
PROVIDER_UNHEALTHY = "Provider {name} is unhealthy: {error}"
NO_SUITABLE_PROVIDER = "No suitable provider found for capabilities: {capabilities}"


class ProviderNotFoundError(ValueError):
    """Raised when provider is not found."""

    def __init__(self, name: str) -> None:
        """Initialize error.

        Args:
            name: Provider name.

        """
        super().__init__(PROVIDER_NOT_FOUND.format(name=name))


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _instance: ClassVar[type[LLMProviderFactory] | None] = None
    _initialized: ClassVar[bool] = False
    _providers: ClassVar[dict[str, type[BaseLLMProvider]]] = {}
    _current_provider: ClassVar[BaseLLMProvider | None] = None
    _provider_name: ClassVar[str | None] = None
    _provider_configs: ClassVar[dict[str, ProviderConfig]] = {}
    _provider_versions: ClassVar[dict[str, ProviderVersion]] = {}
    _provider_lifecycles: ClassVar[dict[str, ProviderLifecycle]] = {}
    _selector: ClassVar[ProviderSelector | None] = None

    def __new__(cls) -> Self:
        """Create or return singleton instance.

        Returns:
            The singleton instance.

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize factory."""
        if not self._initialized:
            self._initialized = True
            self._selector = ProviderSelector(
                providers=self._provider_lifecycles,
                versions=self._provider_versions,
                fallback_chain=["gemini"],  # Default fallback chain
            )

    @classmethod
    def _validate_provider_class(
        cls,
        name: str,
        provider_cls: type[BaseLLMProvider],
    ) -> None:
        """Validate provider class.

        Args:
            name: Provider name.
            provider_cls: Provider class to validate.

        Raises:
            InvalidModelError: If provider class is invalid.
            ConfigError: If provider already registered.

        """
        if not issubclass(provider_cls, BaseLLMProvider):
            raise InvalidModelError(INVALID_PROVIDER.format(name=name))

        if not isinstance(provider_cls, type):
            raise InvalidModelError(INVALID_PROVIDER.format(name=name))

        if name in cls._providers:
            raise ConfigError(PROVIDER_EXISTS.format(name=name))

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_cls: type[BaseLLMProvider],
        version: ProviderVersion | None = None,
    ) -> None:
        """Register a provider.

        Args:
            name: Provider name.
            provider_cls: Provider class.
            version: Optional provider version info.

        Raises:
            InvalidModelError: If provider class is invalid.
            ConfigError: If provider already registered.

        """
        cls._validate_provider_class(name, provider_cls)
        cls._providers[name] = provider_cls
        if version:
            cls._provider_versions[name] = version

    @classmethod
    def get_provider(cls, name: str) -> type[BaseLLMProvider]:
        """Get a provider class.

        Args:
            name: Provider name.

        Returns:
            Provider class.

        Raises:
            ProviderNotFoundError: If provider not found.

        """
        if name not in cls._providers:
            raise ProviderNotFoundError(name)
        return cls._providers[name]

    @classmethod
    def get_provider_version(cls, name: str) -> ProviderVersion:
        """Get provider version information.

        Args:
            name: Provider name.

        Returns:
            Provider version information.

        Raises:
            ProviderNotFoundError: If provider not found.

        """
        if name not in cls._provider_versions:
            raise ProviderNotFoundError(name)
        return cls._provider_versions[name]

    @classmethod
    def get_current_provider(cls) -> BaseLLMProvider | None:
        """Get current provider instance.

        Returns:
            Current provider instance or None.

        """
        return cls._current_provider

    @classmethod
    def get_current_provider_name(cls) -> str | None:
        """Get current provider name.

        Returns:
            Current provider name or None.

        """
        return cls._provider_name

    @classmethod
    def _load_provider_config(cls, name: str) -> ProviderConfig:
        """Load provider configuration.

        Args:
            name: Provider name

        Returns:
            Provider configuration

        Raises:
            ConfigError: If configuration loading fails

        """
        try:
            # Create dummy instance to get config class
            dummy_instance = cls.get_provider(name)(None)  # type: ignore[arg-type]
            config_keys = dummy_instance.config.required_keys()
            env_vars = load_config_from_env(config_keys)
            return dummy_instance.config.__class__.from_env(env_vars)
        except (KeyError, ValueError, AttributeError) as e:
            raise ConfigError(PROVIDER_CONFIG_ERROR.format(name=name, error=str(e))) from e

    def _create_provider_config(self, name: str) -> ProviderConfig:
        """Create provider configuration.

        Args:
            name: Provider name

        Returns:
            Provider configuration

        Raises:
            ConfigError: If configuration creation fails

        """
        try:
            # Create dummy instance to get config class
            dummy_instance = self._providers[name](None)  # type: ignore[arg-type]
            config_keys = dummy_instance.config.required_keys()
            env_vars = load_config_from_env(config_keys)
            return dummy_instance.config.__class__.from_env(env_vars)
        except (KeyError, ValueError, AttributeError) as e:
            raise ConfigError(PROVIDER_CONFIG_ERROR.format(name=name, error=str(e))) from e

    def _get_provider_by_name(
        self,
        name: str,
    ) -> tuple[BaseLLMProvider, ProviderVersion]:
        """Get provider by name.

        Args:
            name: Provider name.

        Returns:
            Tuple of provider and version.

        Raises:
            ValueError: If provider name is invalid.
            ConfigError: If provider version is missing.

        """
        if name not in self._providers:
            raise ValueError(UNSUPPORTED_PROVIDER.format(name))

        provider = self._providers[name]
        version = self._provider_versions.get(name)
        if not version:
            msg = f"No version information for provider {name}"
            raise ConfigError(msg)

        return provider, version

    def _get_provider_by_capabilities(
        self,
        capabilities: list[str] | None = None,
        temperature: float | None = None,
    ) -> tuple[BaseLLMProvider, ProviderVersion]:
        """Get provider by capabilities.

        Args:
            capabilities: Required capabilities.
            temperature: Temperature setting.

        Returns:
            Tuple of provider and version.

        Raises:
            ConfigError: If provider selector is not initialized.

        """
        if not self._selector:
            msg = "Provider selector not initialized"
            raise ConfigError(msg)

        lifecycle = self._selector.select_provider(capabilities, temperature)
        return lifecycle.provider, lifecycle.version

    def _validate_provider_health(
        self,
        lifecycle: ProviderLifecycle,
        name: str | None = None,
    ) -> None:
        """Validate provider health.

        Args:
            lifecycle: Provider lifecycle.
            name: Provider name.

        Raises:
            EmptyResponseError: If provider is unhealthy.

        """
        if not lifecycle.check_health():
            raise EmptyResponseError(
                PROVIDER_UNHEALTHY.format(
                    name=name or lifecycle.provider.__class__.__name__,
                    error=lifecycle.health.last_error or "Unknown error",
                ),
            )

    def get_provider_instance(
        self,
        name: str | None = None,
        capabilities: list[str] | None = None,
        temperature: float | None = None,
    ) -> BaseLLMProvider:
        """Get provider instance.

        Args:
            name: Provider name.
            capabilities: Required capabilities.
            temperature: Temperature setting.

        Returns:
            Provider instance.

        Raises:
            ConfigError: If provider creation fails.

        """
        try:
            # Get provider and version
            if name:
                provider, version = self._get_provider_by_name(name)
            else:
                provider, version = self._get_provider_by_capabilities(capabilities, temperature)

            # Create lifecycle
            lifecycle = ProviderLifecycle(provider, version)

            # Validate provider health
            self._validate_provider_health(lifecycle, name)

            # Set as current provider
            self._current_provider = lifecycle.provider
            self._provider_lifecycles[name] = lifecycle
        except (ValueError, ConfigError, EmptyResponseError) as e:
            msg = f"Failed to get provider: {e!s}"
            raise ConfigError(msg) from e
        else:
            return self._current_provider

    def get_fallback_provider(self) -> BaseLLMProvider:
        """Get next provider in fallback chain.

        Returns:
            Next provider in fallback chain.

        Raises:
            RetryError: If all fallbacks exhausted.

        """
        if not self._selector:
            msg = "Provider selector not initialized"
            raise ConfigError(msg)

        lifecycle = self._selector.get_fallback_provider()
        if not lifecycle:
            msg = "No fallback providers available"
            raise RetryError(msg)

        self._current_provider = lifecycle.provider
        self._provider_name = lifecycle.provider.__class__.__name__
        return lifecycle.provider

    def reset_fallback_chain(self) -> None:
        """Reset fallback chain."""
        if self._selector:
            self._selector.reset_fallback_chain()

    def set_fallback_chain(self, providers: list[str]) -> None:
        """Set fallback provider chain.

        Args:
            providers: List of provider names in fallback order.

        Raises:
            ConfigError: If any provider is not registered.

        """
        for name in providers:
            if name not in self._providers:
                msg = f"Provider {name} not registered"
                raise ConfigError(msg)

        if self._selector:
            self._selector.fallback_chain = providers.copy()
            self._selector.reset_fallback_chain()

    @classmethod
    def _get_cached_provider(cls, name: str, config: ProviderConfig | None) -> BaseLLMProvider | None:
        """Get cached provider if available.

        Args:
            name: Provider name.
            config: Provider configuration.

        Returns:
            Cached provider instance or None if not found.

        """
        if name in cls._provider_lifecycles:
            lifecycle = cls._provider_lifecycles[name]
            if lifecycle.provider.config == config:
                return lifecycle.provider
        return None

    @classmethod
    def _create_provider_instance(cls, name: str, config: ProviderConfig) -> BaseLLMProvider:
        """Create a new provider instance.

        Args:
            name: Provider name.
            config: Provider configuration.

        Returns:
            New provider instance.

        Raises:
            ConfigError: If provider creation fails.

        """
        provider_cls = cls._providers[name]
        try:
            provider = provider_cls(config=config)
            version = cls._provider_versions.get(name) or ProviderVersion(name)
            lifecycle = ProviderLifecycle(provider, version)
            lifecycle.initialize()
            cls._provider_lifecycles[name] = lifecycle
        except Exception as e:
            msg = f"Failed to create provider {name}: {e}"
            raise ConfigError(msg) from e
        else:
            return provider

    @classmethod
    def create_provider(
        cls,
        name: str,
        config: ProviderConfig | None = None,
    ) -> BaseLLMProvider:
        """Create provider instance.

        Args:
            name: Provider name.
            config: Provider configuration.

        Returns:
            Provider instance.

        Raises:
            ConfigError: If provider creation fails.
            APIKeyError: If API key is missing.

        """

        def _raise_unsupported_provider(name: str) -> None:
            raise ValueError(UNSUPPORTED_PROVIDER.format(name))

        def _raise_api_key_error() -> None:
            raise APIKeyError(API_KEY_REQUIRED)

        try:
            # Get provider class
            if name not in cls._providers:
                _raise_unsupported_provider(name)

            # Check if we have a cached provider
            cached_provider = cls._get_cached_provider(name, config)
            if cached_provider:
                return cached_provider

            # Create configuration if not provided
            if not config:
                api_key = load_config_from_env(["API_KEY"])
                if not api_key:
                    _raise_api_key_error()

                provider_cls = cls._providers[name]
                config = provider_cls.create_config(api_key)

            # Create provider instance
            return cls._create_provider_instance(name, config)

        except (ValueError, APIKeyError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            msg = f"Failed to create provider {name}: {e}"
            raise ConfigError(msg) from e

    @classmethod
    def cleanup_provider(cls, name: str) -> None:
        """Clean up provider resources.

        Args:
            name: Provider name

        Raises:
            ConfigError: If cleanup fails

        """
        try:
            # Remove provider lifecycle
            del cls._provider_lifecycles[name]
        except (KeyError, AttributeError) as e:
            msg = f"Failed to clean up provider {name}: {e!s}"
            raise ConfigError(msg) from e


# Register default providers
LLMProviderFactory.register_provider(
    "gemini",
    GeminiProvider,
    version=ProviderVersion.GEMINI_V1,
)
