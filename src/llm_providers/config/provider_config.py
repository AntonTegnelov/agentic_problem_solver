"""Provider configuration module."""

from dataclasses import dataclass, field
from typing import Any

from src.config import BaseConfig


@dataclass
class ProviderConfig(BaseConfig):
    """Base provider configuration."""

    api_key: str | None = None
    model: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid.

        """
        error_msg: str
        if not self.api_key:
            error_msg = "API key is required"
            raise ValueError(error_msg)


@dataclass
class GeminiConfig(ProviderConfig):
    """Gemini provider configuration."""

    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid.

        """
        error_msg: str
        super().validate()

        if not 0 <= self.temperature <= 1:
            error_msg = "Temperature must be between 0 and 1"
            raise ValueError(error_msg)

        if self.max_output_tokens <= 0:
            error_msg = "Max tokens must be positive"
            raise ValueError(error_msg)

        if not 0 <= self.top_p <= 1:
            error_msg = "Top P must be between 0 and 1"
            raise ValueError(error_msg)

        if self.top_k <= 0:
            error_msg = "Top K must be positive"
            raise ValueError(error_msg)
