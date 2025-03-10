"""Provider configuration module."""

from __future__ import annotations

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

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ConfigError: If configuration is invalid.

        """

        def _raise_invalid_model_error(msg: str) -> None:
            raise InvalidModelError(msg)

        def _raise_value_error(msg: str) -> None:
            raise ValueError(msg)

        try:
            # Validate model version
            model_version = self.get_model_version()
            if model_version.min_provider_version > self.provider_version:
                msg = (
                    f"Model {self.model_name} requires provider version "
                    f"{model_version.min_provider_version} or higher. "
                    f"Current version is {self.provider_version}"
                )
                _raise_invalid_model_error(msg)

            # Validate parameters
            if not 0 <= self.temperature <= 1:
                _raise_value_error("Temperature must be between 0 and 1")

            if self.max_output_tokens <= 0:
                _raise_value_error("Max tokens must be positive")

            if not 0 <= self.top_p <= 1:
                _raise_value_error("Top P must be between 0 and 1")

            if self.top_k <= 0:
                _raise_value_error("Top K must be positive")
        except (ValueError, KeyError, InvalidModelError) as e:
            msg = f"Invalid configuration: {e!s}"
            raise ConfigError(msg) from e
        else:
            return True

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

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ConfigError: If configuration is invalid.

        """

        def _raise_invalid_model_error(msg: str) -> None:
            raise InvalidModelError(msg)

        def _raise_value_error(msg: str) -> None:
            raise ValueError(msg)

        try:
            # Validate model version
            model_version = self.get_model_version()
            if model_version.min_provider_version > self.provider_version:
                msg = (
                    f"Model {self.model_name} requires provider version "
                    f"{model_version.min_provider_version} or higher. "
                    f"Current version is {self.provider_version}"
                )
                _raise_invalid_model_error(msg)

            # Validate parameters
            if not 0 <= self.temperature <= 1:
                _raise_value_error("Temperature must be between 0 and 1")

            if self.max_output_tokens <= 0:
                _raise_value_error("Max tokens must be positive")

            if not 0 <= self.top_p <= 1:
                _raise_value_error("Top P must be between 0 and 1")

            if self.top_k <= 0:
                _raise_value_error("Top K must be positive")
        except (ValueError, KeyError, InvalidModelError) as e:
            msg = f"Invalid configuration: {e!s}"
            raise ConfigError(msg) from e
        else:
            return True

    @classmethod
    def from_env(cls, env_vars: dict[str, str]) -> GeminiConfig:
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
            raise ConfigError(msg) from e
