"""LLM provider factory."""

from typing import ClassVar

from src.config.utils import load_config_from_env
from src.exceptions import (
    APIKeyError,
    ConfigError,
    EmptyResponseError,
    InvalidModelError,
    RetryError,
)
from src.llm_providers.config.provider_config import ProviderConfig
from src.llm_providers.lifecycle import ProviderLifecycle, ProviderState
from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.providers.gemini import GeminiProvider
from src.llm_providers.selection import ProviderCapability, ProviderSelector
from src.llm_providers.version import ProviderVersion

# Error messages
PROVIDER_NOT_FOUND = "Provider {name} not found"
API_KEY_REQUIRED = "API key is required"
NO_PROVIDER_SET = "No provider set"
UNSUPPORTED_PROVIDER = "Unsupported provider: {}"
INVALID_PROVIDER = "Invalid provider class: {name}. Must implement BaseLLMProvider"
PROVIDER_EXISTS = "Provider {name} already registered"
PROVIDER_CONFIG_ERROR = "Failed to create config for provider {name}: {error}"
VERSION_MISMATCH = (
    "Provider {name} version {version} does not match required version {required}"
)
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

    _instance: ClassVar[type["LLMProviderFactory"] | None] = None
    _initialized: ClassVar[bool] = False
    _providers: ClassVar[dict[str, type[BaseLLMProvider]]] = {}
    _current_provider: ClassVar[BaseLLMProvider | None] = None
    _provider_name: ClassVar[str | None] = None
    _provider_configs: ClassVar[dict[str, ProviderConfig]] = {}
    _provider_versions: ClassVar[dict[str, ProviderVersion]] = {}
    _provider_lifecycles: ClassVar[dict[str, ProviderLifecycle]] = {}
    _selector: ClassVar[ProviderSelector | None] = None

    def __new__(cls) -> "LLMProviderFactory":
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
            name: Provider name.

        Returns:
            Provider configuration.

        Raises:
            ConfigError: If configuration loading fails.

        """
        try:
            provider_cls = cls.get_provider(name)
            dummy_instance = provider_cls(
                None,
            )  # Create temporary instance to get config class
            config_keys = dummy_instance.config.required_keys()
            env_vars = load_config_from_env(config_keys)
            return dummy_instance.config.__class__.from_env(env_vars)
        except Exception as e:
            raise ConfigError(PROVIDER_CONFIG_ERROR.format(name=name, error=str(e)))

    def set_provider(
        self,
        name: str | None = None,
        capabilities: list[ProviderCapability] | None = None,
        temperature: float | None = None,
    ) -> None:
        """Set the active provider.

        Args:
            name: Optional provider name. If not provided, selects based on capabilities.
            capabilities: Required capabilities.
            temperature: Required temperature setting.

        Raises:
            ValueError: If provider not supported.
            ConfigError: If provider configuration is invalid.
            EmptyResponseError: If provider is unhealthy.
            TemperatureError: If temperature requirements not met.

        """
        try:
            if name:
                if name not in self._providers:
                    raise ValueError(UNSUPPORTED_PROVIDER.format(name))

                # Use specific provider
                if name in self._provider_lifecycles:
                    lifecycle = self._provider_lifecycles[name]
                else:
                    # Create new provider instance
                    config = self._load_provider_config(name)
                    provider_cls = self._providers[name]
                    provider = provider_cls(config.api_key)

                    # Create and initialize lifecycle
                    version = self._provider_versions.get(name)
                    if not version:
                        msg = f"No version information for provider {name}"
                        raise ConfigError(msg)

                    lifecycle = ProviderLifecycle(provider, version)
                    lifecycle.initialize()
                    self._provider_lifecycles[name] = lifecycle

            else:
                # Select provider based on capabilities
                if not self._selector:
                    msg = "Provider selector not initialized"
                    raise ConfigError(msg)

                lifecycle = self._selector.select_provider(capabilities, temperature)

            # Validate provider health
            if not lifecycle.check_health():
                raise EmptyResponseError(
                    PROVIDER_UNHEALTHY.format(
                        name=name or lifecycle.provider.__class__.__name__,
                        error=lifecycle.health.last_error or "Unknown error",
                    ),
                )

            # Set as current provider
            self._current_provider = lifecycle.provider
            self._provider_name = name or lifecycle.provider.__class__.__name__
            self._provider_configs[self._provider_name] = lifecycle.provider.config

            # Update load distribution
            if self._selector:
                self._selector.update_load_distribution(
                    self._provider_name,
                    lifecycle.stats.requests_per_minute,
                )

        except Exception as e:
            if name and name in self._provider_lifecycles:
                self._provider_lifecycles[name].state = ProviderState.ERROR
            msg = f"Failed to initialize provider {name}: {e!s}"
            raise ConfigError(msg)

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
    def create_provider(cls, name: str, api_key: str | None = None) -> BaseLLMProvider:
        """Create provider instance.

        Args:
            name: Provider name.
            api_key: Optional API key. If not provided, loaded from config.

        Returns:
            Provider instance.

        Raises:
            APIKeyError: If API key is missing.
            ConfigError: If provider configuration is invalid.
            ProviderNotFoundError: If provider not found.

        """
        if name not in cls._providers:
            raise ProviderNotFoundError(name)

        try:
            if api_key is None:
                config = cls._load_provider_config(name)
                api_key = config.api_key

            if not api_key:
                raise APIKeyError(API_KEY_REQUIRED)

            provider_cls = cls._providers[name]
            provider = provider_cls(api_key)

            # Create and initialize lifecycle
            version = cls._provider_versions.get(name)
            if version:
                lifecycle = ProviderLifecycle(provider, version)
                lifecycle.initialize()
                cls._provider_lifecycles[name] = lifecycle

            return provider

        except APIKeyError:
            raise
        except Exception as e:
            if name in cls._provider_lifecycles:
                cls._provider_lifecycles[name].state = ProviderState.ERROR
            msg = f"Failed to create provider {name}: {e!s}"
            raise ConfigError(msg)

    @classmethod
    def cleanup_provider(cls, name: str) -> None:
        """Clean up provider resources.

        Args:
            name: Provider name.

        Raises:
            ProviderNotFoundError: If provider not found.

        """
        if name not in cls._provider_lifecycles:
            raise ProviderNotFoundError(name)

        try:
            lifecycle = cls._provider_lifecycles[name]
            lifecycle.cleanup()

            if name == cls._provider_name:
                cls._current_provider = None
                cls._provider_name = None

            del cls._provider_lifecycles[name]

        except Exception as e:
            msg = f"Failed to clean up provider {name}: {e!s}"
            raise ConfigError(msg)


# Register default providers
LLMProviderFactory.register_provider(
    "gemini",
    GeminiProvider,
    version=ProviderVersion.GEMINI_V1,
)
