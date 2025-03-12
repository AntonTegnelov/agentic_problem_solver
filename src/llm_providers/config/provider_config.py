"""Provider configuration module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.common_types.error_types import ConfigError, InvalidModelError
from src.config import BaseConfig
from src.llm_providers.version import ModelVersion, ProviderVersion, Version


@dataclass
class ProviderConfig(BaseConfig):
    """Provider configuration."""

    provider_name: str = "default"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 100
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    api_type: str | None = None
    deployment_name: str | None = None
    organization_id: str | None = None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

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
        if not self.api_key:
            msg = "API key is required"
            raise ConfigError(msg)

        if not self.model:
            msg = "Model name is required"
            raise ConfigError(msg)

        # Validate version if provided
        if self.api_version:
            try:
                # Add version validation logic here
                pass
            except ValueError as e:
                msg = f"Invalid version format: {e}"
                raise ConfigError(msg) from e

        return True

    def get_model_version(self) -> ModelVersion:
        """Get model version.

        Returns:
            Model version.

        Raises:
            InvalidModelError: If model is not supported.

        """
        if not self.PROVIDER_VERSION:
            msg = "Provider version not set"
            raise ConfigError(msg)

        return self.PROVIDER_VERSION.get_model(self.model)

    def required_keys(self) -> list[str]:
        """Get required environment variable keys.

        Returns:
            List of required keys with provider prefix.

        """
        provider_name = self.provider_name.upper()
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
            if model_version.min_provider_version > self.PROVIDER_VERSION.version:
                msg = (
                    f"Model {self.model} requires provider version "
                    f"{model_version.min_provider_version} or higher. "
                    f"Current version is {self.PROVIDER_VERSION.version}"
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
                api_key=env_vars.get("GEMINI_API_KEY"),
                model=env_vars.get("GEMINI_MODEL", "gemini-pro"),  # Ensure this is set
                api_version=Version(1, 0, 0),  # Current version
                temperature=float(env_vars.get("GEMINI_TEMPERATURE", "0.7")),
                max_output_tokens=int(env_vars.get("GEMINI_MAX_OUTPUT_TOKENS", "2048")),
                top_p=float(env_vars.get("GEMINI_TOP_P", "0.95")),
                top_k=int(env_vars.get("GEMINI_TOP_K", "40")),
            )
        except (KeyError, ValueError) as e:
            msg = f"Invalid configuration: {e!s}"
            raise ConfigError(msg) from e
