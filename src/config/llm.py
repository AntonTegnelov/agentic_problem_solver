"""LLM provider configuration."""

from dataclasses import asdict, dataclass, field
from typing import Any

from .base import BaseConfig
from .constants import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)


@dataclass
class LLMConfig(BaseConfig):
    """LLM provider configuration."""

    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid.

        """
        error_msg: str

        if not 0 <= self.temperature <= 1:
            error_msg = "Temperature must be between 0 and 1"
            raise ValueError(error_msg)

        if self.max_output_tokens <= 0:
            error_msg = "Max tokens must be positive"
            raise ValueError(error_msg)

        if not self.model:
            error_msg = "Model name cannot be empty"
            raise ValueError(error_msg)

        if not 0 <= self.top_p <= 1:
            error_msg = "Top P must be between 0 and 1"
            raise ValueError(error_msg)

        if self.top_k <= 0:
            error_msg = "Top K must be positive"
            raise ValueError(error_msg)

    def dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation.

        """
        config_dict = asdict(self)
        config_dict.update(self.extra_params)
        return config_dict
