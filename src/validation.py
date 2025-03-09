"""Validation utilities."""

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field, field_validator

# Type variables
T = TypeVar("T")

# Error messages
TEMPERATURE_ERROR = "Temperature must be between 0 and 1"
MAX_TOKENS_ERROR = "Max tokens must be greater than 0"
MODEL_ERROR = "Model name cannot be empty"
TASK_TIMEOUT_ERROR = "Task timeout must be greater than 0"
MAX_RETRIES_ERROR = "Max retries must be greater than 0"
MAX_STEPS_ERROR = "Max steps must be greater than 0"


class MessageRoles(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStep(str, Enum):
    """Agent processing steps."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    END = "end"


class Message(BaseModel):
    """Message model."""

    role: MessageRoles
    content: str
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Agent state."""

    messages: list[Message] = Field(default_factory=list)
    current_step: AgentStep = AgentStep.UNDERSTAND
    step_count: int = 0
    task_completed: bool = False
    error: str | None = None

    def get_message_metadata(self, index: int, key: str, default: T = None) -> T:
        """Get message metadata.

        Args:
            index: Message index.
            key: Metadata key.
            default: Default value.

        Returns:
            Metadata value.
        """
        if index < 0 or index >= len(self.messages):
            return default
        return self.messages[index].response_metadata.get(key, default)


class ProcessingStep(str, Enum):
    """Steps in the agent's workflow."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    END = "end"


class GraphState(BaseModel):
    """The state of the agent."""

    messages: list[Message] = Field(
        default_factory=list,
        description="The conversation history using LangChain message types",
    )
    current_step: ProcessingStep = Field(
        default=ProcessingStep.UNDERSTAND,
        description="The current step in the workflow",
    )
    error: str | None = Field(None, description="Error message if any")
    result: str | None = Field(None, description="Final result if any")

    def get_message_metadata(self, index: int, key: str, default: T = None) -> T:
        """Get metadata from a message at the specified index.

        Args:
            index: The index of the message.
            key: The metadata key to get.
            default: The default value to return if the key is not found.

        Returns:
            The metadata value or default if not found.
        """
        try:
            message = self.messages[index]
            if key == "content":
                return message.content  # type: ignore[return-value]
            if key == "type":
                return message.type  # type: ignore[return-value]
            if key == "tool_call_id" and isinstance(message, ToolMessage):
                return message.tool_call_id  # type: ignore[return-value]
            return message.additional_kwargs.get(key, default)
        except (IndexError, AttributeError):
            return default

    def set_message_metadata(self, index: int, key: str, value: T) -> None:
        """Set metadata for a message at the specified index.

        Args:
            index: The index of the message.
            key: The metadata key to set.
            value: The value to set.
        """
        try:
            message = self.messages[index]
            if key == "content":
                message.content = value  # type: ignore[assignment]
            elif key == "type":
                message.type = value  # type: ignore[assignment]
            elif key == "tool_call_id" and isinstance(message, ToolMessage):
                message.tool_call_id = value  # type: ignore[assignment]
            else:
                message.additional_kwargs[key] = value
        except (IndexError, AttributeError):
            pass

    @classmethod
    @field_validator("messages")
    def validate_messages(cls, v: list[Any]) -> list[Any]:
        """Validate messages list.

        Args:
            v: The messages list to validate.

        Returns:
            The validated messages list.

        Raises:
            ValueError: If messages list is empty.
        """
        if not v:
            msg = "Messages list cannot be empty"
            raise ValueError(msg)
        return v

    model_config = {
        "arbitrary_types_allowed": True,  # Allow LangChain message types
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "content": "Agent initialized",
                            "type": "system",
                            "additional_kwargs": {"metadata": {}},
                        }
                    ],
                    "current_step": "UNDERSTAND",
                    "error": None,
                    "result": None,
                }
            ]
        },
    }


class Task(BaseModel):
    """A task to be solved by the agent."""

    content: str = Field(description="The task content")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context for the task"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Constraints for the task"
    )
    requirements: list[str] = Field(
        default_factory=list, description="Requirements for the task"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "Write a Python function to calculate Fibonacci numbers",
                    "context": {"max_number": 100},
                    "constraints": [
                        "Must be efficient",
                        "Must handle negative numbers",
                    ],
                    "requirements": ["Return type must be List[int]"],
                }
            ]
        }
    }


class TaskResult(BaseModel):
    """The result of a task."""

    result: str = Field(..., description="The task result")
    success: bool = Field(..., description="Whether the task was successful")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the task"
    )


class Step(BaseModel):
    """A step in the agent's workflow."""

    name: ProcessingStep = Field(..., description="The name of the step")
    status: str = Field(
        default="pending",
        description="The status of the step (pending, in_progress, completed, failed)",
    )

    @classmethod
    @field_validator("name")
    def validate_name(cls, v: ProcessingStep) -> ProcessingStep:
        """Validate the name field."""
        valid_steps = list(ProcessingStep)
        if v not in valid_steps:
            msg = f"Step name must be one of {valid_steps}"
            raise ValueError(msg)
        return v

    @classmethod
    @field_validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate the status field."""
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        if v not in valid_statuses:
            msg = f"Status must be one of {valid_statuses}"
            raise ValueError(msg)
        return v


class AgentConfig(BaseModel):
    """Agent configuration."""

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
        error_messages={"le": TEMPERATURE_ERROR},
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of tokens to generate",
        error_messages={"gt": MAX_TOKENS_ERROR},
    )
    model: str = Field(
        default="gemini-pro",
        min_length=1,
        description="Model name",
        error_messages={"min_length": MODEL_ERROR},
    )
    task_timeout: int = Field(
        default=300,
        gt=0,
        description="Task timeout in seconds",
        error_messages={"gt": TASK_TIMEOUT_ERROR},
    )
    max_retries: int = Field(
        default=3,
        gt=0,
        description="Maximum number of retries",
        error_messages={"gt": MAX_RETRIES_ERROR},
    )
    max_steps: int = Field(
        default=10,
        gt=0,
        description="Maximum number of processing steps",
        error_messages={"gt": MAX_STEPS_ERROR},
    )
