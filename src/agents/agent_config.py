"""Agent configuration."""

from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TASK_TIMEOUT,
    DEFAULT_TEMPERATURE,
)
from src.validation import (
    MAX_RETRIES_ERROR,
    MAX_STEPS_ERROR,
    MAX_TOKENS_ERROR,
    MODEL_ERROR,
    TASK_TIMEOUT_ERROR,
    TEMPERATURE_ERROR,
)


class AgentConfig(BaseModel):
    """Agent configuration."""

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=1.0,
        json_schema_extra={"error_messages": {"le": TEMPERATURE_ERROR}},
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        gt=0,
        json_schema_extra={"error_messages": {"gt": MAX_TOKENS_ERROR}},
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        json_schema_extra={"error_messages": {"min_length": MODEL_ERROR}},
    )
    task_timeout: int = Field(
        default=DEFAULT_TASK_TIMEOUT,
        gt=0,
        json_schema_extra={"error_messages": {"gt": TASK_TIMEOUT_ERROR}},
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        json_schema_extra={"error_messages": {"ge": MAX_RETRIES_ERROR}},
    )
    max_steps: int = Field(
        default=DEFAULT_MAX_STEPS,
        gt=0,
        json_schema_extra={"error_messages": {"gt": MAX_STEPS_ERROR}},
    )
