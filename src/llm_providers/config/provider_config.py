"""Provider configuration module."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.config import BaseConfig
from src.exceptions import ConfigError, InvalidModelError
from src.llm_providers.version import ProviderVersion, Version


@dataclass
class ProviderConfig(BaseConfig):
    """Base provider configuration."""

    api_key: str | None = None
    model: str | None = None
    version: Version | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    # Required environment variables
    REQUIRED_ENV_VARS: ClassVar[list[str]] = ["API_KEY", "MODEL"]

    # Provider version information
    PROVIDER_VERSION: ClassVar[ProviderVersion | None] = None

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If configuration is invalid.
            InvalidModelError: If model is not supported.

        """
        if not self.api_key:
            msg = "API key is required"
            raise ConfigError(msg)

        if not self.model:
            msg = "Model is required"
            raise ConfigError(msg)

        if self.PROVIDER_VERSION is None:
            msg = "Provider version not set"
            raise ConfigError(msg)

        # Validate model is supported
        try:
            model_version = self.PROVIDER_VERSION.get_model(self.model)
            if self.version and self.version < model_version.min_provider_version:
                msg = (
                    f"Provider version {self.version} is too old for model {self.model}. "
                    f"Minimum required version is {model_version.min_provider_version}"
                )
                raise InvalidModelError(
                    msg,
                )
        except InvalidModelError as e:
            msg = f"Invalid model configuration: {e!s}"
            raise InvalidModelError(msg)

    def required_keys(self) -> list[str]:
        """Get required environment variable keys.

        Returns:
            List of required keys with provider prefix.

        """
        provider_name = self.__class__.__name__.replace("Config", "").upper()
        return [f"{provider_name}_{key}" for key in self.REQUIRED_ENV_VARS]


@dataclass
class GeminiConfig(ProviderConfig):
    """Gemini provider configuration."""

    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40

    # Required environment variables
    REQUIRED_ENV_VARS: ClassVar[list[str]] = [
        "API_KEY",
        "MODEL",
        "TEMPERATURE",
        "MAX_OUTPUT_TOKENS",
        "TOP_P",
        "TOP_K",
    ]

    # Provider version information
    PROVIDER_VERSION = ProviderVersion.GEMINI_V1

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If configuration is invalid.

        """
        try:
            super().validate()

            if not 0 <= self.temperature <= 1:
                msg = "Temperature must be between 0 and 1"
                raise ValueError(msg)

            if self.max_output_tokens <= 0:
                msg = "Max tokens must be positive"
                raise ValueError(msg)

            if not 0 <= self.top_p <= 1:
                msg = "Top P must be between 0 and 1"
                raise ValueError(msg)

            if self.top_k <= 0:
                msg = "Top K must be positive"
                raise ValueError(msg)

        except ValueError as e:
            raise ConfigError(str(e))

    @classmethod
    def from_env(cls, env_vars: dict[str, str]) -> "GeminiConfig":
        """Create config from environment variables."""
        try:
            return cls(
                api_key=env_vars["GEMINI_API_KEY"],
                model=env_vars.get("GEMINI_MODEL", "gemini-pro"),  # Ensure this is set
                version=Version(1, 0, 0),  # Current version
                temperature=float(env_vars.get("GEMINI_TEMPERATURE", "0.7")),
                max_output_tokens=int(env_vars.get("GEMINI_MAX_OUTPUT_TOKENS", "2048")),
                top_p=float(env_vars.get("GEMINI_TOP_P", "0.95")),
                top_k=int(env_vars.get("GEMINI_TOP_K", "40")),
            )
        except (KeyError, ValueError) as e:
            msg = f"Invalid configuration: {e!s}"
            raise ConfigError(msg)
