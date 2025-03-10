"""Agent configuration."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.exceptions import ConfigError

from .base import BaseConfig
from .constants import (
    DEFAULT_MAX_STEPS,
    DEFAULT_TASK_TIMEOUT,
)
from .llm import LLMConfig


class NumericValidation(Enum):
    """Numeric validation options."""

    ALLOW_ZERO = auto()
    DISALLOW_ZERO = auto()


@dataclass
class AgentConfig(BaseConfig):
    """Agent configuration.

    The default model is 'gemini-2.0-flash-lite', which is optimized for our use case.
    This model provides the best balance of speed and quality for the problem solver.
    """

    model: str | LLMConfig = (
        "gemini-2.0-flash-lite"  # Default model - DO NOT CHANGE without updating docs
    )
    temperature: float = 0.7
    max_tokens: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    context: dict[str, Any] | None = field(default_factory=dict)
    task_timeout: int = DEFAULT_TASK_TIMEOUT
    max_steps: int = DEFAULT_MAX_STEPS
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def _validate_numeric_field(
        self,
        field_name: str,
        value: float,
        min_value: float | None = None,
        max_value: float | None = None,
        validation_type: NumericValidation = NumericValidation.DISALLOW_ZERO,
    ) -> None:
        """Validate a numeric field.

        Args:
            field_name: Name of the field being validated.
            value: Value to validate.
            min_value: Minimum allowed value (inclusive).
            max_value: Maximum allowed value (inclusive).
            validation_type: Type of validation to perform.

        Raises:
            ConfigError: If validation fails.

        """
        if min_value is not None and value < min_value:
            msg = f"{field_name} must be greater than {min_value}"
            raise ConfigError(msg)

        if max_value is not None and value > max_value:
            msg = f"{field_name} must be less than {max_value}"
            raise ConfigError(msg)

        if validation_type == NumericValidation.DISALLOW_ZERO and value == 0:
            msg = f"{field_name} must be non-zero"
            raise ConfigError(msg)

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigError: If configuration is invalid.

        """
        # Validate numeric fields
        self._validate_numeric_field("task_timeout", self.task_timeout, min_value=1)
        self._validate_numeric_field(
            "max_retries",
            self.max_retries,
            min_value=0,
            validation_type=NumericValidation.ALLOW_ZERO,
        )
        self._validate_numeric_field("max_steps", self.max_steps, min_value=1)
        self._validate_numeric_field(
            "temperature",
            self.temperature,
            min_value=0,
            max_value=1,
        )

        # Validate model
        if isinstance(self.model, dict):
            self.model = LLMConfig(**self.model)
        elif isinstance(self.model, LLMConfig):
            self.model.validate()
        elif not isinstance(self.model, str):
            msg = "Model must be a string or LLMConfig instance"
            raise ConfigError(msg)
