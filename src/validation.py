"""Validation utilities."""

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from src.agent.agent_types.agent_types import Message

T = TypeVar("T")

# Error messages
TEMPERATURE_ERROR = "Temperature must be between 0 and 1"
MAX_TOKENS_ERROR = "Max tokens must be greater than 0"
MODEL_ERROR = "Model name cannot be empty"
TASK_TIMEOUT_ERROR = "Task timeout must be greater than 0"
MAX_RETRIES_ERROR = "Max retries must be greater than 0"
MAX_STEPS_ERROR = "Maximum steps must be greater than 0"
INVALID_ROLE_ERROR = "Invalid role: {role}"


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MessageRoles(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ProcessingStep(str, Enum):
    """Steps in the agent's workflow."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    END = "end"


def get_message_metadata(
    messages: list[Message],
    index: int,
    key: str,
    default: T = None,
) -> T:
    """Get message metadata.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        default: Default value.

    Returns:
        Metadata value.

    """
    try:
        message = messages[index]
        if key == "tool_call_id":
            return message.tool_call_id  # type: ignore[return-value]
        return message.metadata.get(key, default)
    except (IndexError, AttributeError):
        return default


def set_message_metadata(
    messages: list[Message],
    index: int,
    key: str,
    value: str | None,
) -> None:
    """Set message metadata.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        value: Metadata value.

    """
    try:
        message = messages[index]
        if key == "tool_call_id":
            message.tool_call_id = value  # type: ignore[assignment]
        else:
            message.metadata[key] = value
    except (IndexError, AttributeError):
        pass


class Config(BaseModel):
    """Configuration model."""

    temperature: float | None = Field(
        None,
        description="Temperature for response generation",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int | None = Field(
        None,
        description="Maximum tokens in response",
        gt=0,
    )
    top_p: float | None = Field(
        None,
        description="Top p for response generation",
        ge=0.0,
        le=1.0,
    )
    top_k: int | None = Field(
        None,
        description="Top k for response generation",
        gt=0,
    )
    presence_penalty: float | None = Field(
        None,
        description="Presence penalty for response generation",
    )
    frequency_penalty: float | None = Field(
        None,
        description="Frequency penalty for response generation",
    )
    stop: str | list[str] | None = Field(
        None,
        description="Stop sequences for response generation",
    )

    def dict(self, *args: list[Any], **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert to dictionary, excluding None values.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dictionary representation.

        """
        d = super().dict(*args, **kwargs)
        return {k: v for k, v in d.items() if v is not None}
