"""Agent configuration."""

from dataclasses import dataclass

from .base import BaseConfig
from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_TASK_TIMEOUT,
)
from .llm import LLMConfig


@dataclass
class AgentConfig(BaseConfig):
    """Agent configuration."""

    llm: LLMConfig
    task_timeout: int = DEFAULT_TASK_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    max_steps: int = DEFAULT_MAX_STEPS
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        error_msg: str

        if self.task_timeout <= 0:
            error_msg = "Task timeout must be greater than 0"
            raise ValueError(error_msg)

        if self.max_retries < 0:
            error_msg = "Max retries must be non-negative"
            raise ValueError(error_msg)

        if self.max_steps <= 0:
            error_msg = "Max steps must be positive"
            raise ValueError(error_msg)

        # Validate LLM config if present
        if isinstance(self.llm, dict):
            self.llm = LLMConfig(**self.llm)
        self.llm.validate()
