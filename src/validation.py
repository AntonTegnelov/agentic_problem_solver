"""Validation module for input validation."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field, field_validator

from src.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TASK_TIMEOUT,
    DEFAULT_TEMPERATURE,
)


class AgentStep(str, Enum):
    """Possible steps in the agent's workflow."""

    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    END = "END"


class AgentState(BaseModel):
    """The state of the agent."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]] = Field(
        default_factory=list,
        description="The conversation history using LangChain message types",
    )
    current_step: str = Field(
        default=AgentStep.UNDERSTAND,
        description="The current step in the workflow",
    )
    error: Optional[str] = Field(None, description="Error message if any")
    result: Optional[str] = Field(None, description="Final result if any")

    def get_message_metadata(self, index: int, key: str, default: Any = None) -> Any:
        """Get metadata from a message at the specified index.

        Args:
            index: Index of the message
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        try:
            message = self.messages[index]
            # First check standard message attributes
            if key == "content":
                return message.content
            elif key == "type":
                return message.type
            elif key == "tool_call_id" and isinstance(message, ToolMessage):
                return message.tool_call_id
            # Then check additional_kwargs metadata
            metadata = message.additional_kwargs.get("metadata", {})
            return metadata.get(key, default)
        except IndexError:
            return default

    def set_message_metadata(self, index: int, key: str, value: Any) -> None:
        """Set metadata for a message at the specified index.

        Args:
            index: Index of the message
            key: Metadata key
            value: Metadata value
        """
        try:
            message = self.messages[index]
            # Handle standard message attributes
            if key == "content":
                message.content = value
                return
            # Store other metadata in additional_kwargs
            if "metadata" not in message.additional_kwargs:
                message.additional_kwargs["metadata"] = {}
            message.additional_kwargs["metadata"][key] = value
        except IndexError:
            pass

    @field_validator("messages")
    @classmethod
    def messages_not_empty(
        cls, v: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
    ) -> List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]:
        """Validate that the messages list is not empty."""
        if not v:
            raise ValueError("Messages list cannot be empty")
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


class AgentConfig(BaseModel):
    """Agent configuration model."""

    temperature: float = Field(default=DEFAULT_TEMPERATURE)
    max_tokens: Optional[int] = Field(default=None)
    model: str = Field(default=DEFAULT_MODEL)
    task_timeout: int = Field(default=DEFAULT_TASK_TIMEOUT)
    max_retries: int = Field(default=DEFAULT_MAX_RETRIES)
    max_steps: int = Field(default=DEFAULT_MAX_STEPS)

    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature value.

        Args:
            v: Temperature value

        Returns:
            Validated temperature value

        Raises:
            ValueError: If temperature is not between 0 and 1
        """
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @field_validator("max_tokens")
    def validate_max_tokens(cls, v: Optional[int]) -> Optional[int]:
        """Validate max_tokens value.

        Args:
            v: Max tokens value

        Returns:
            Validated max tokens value

        Raises:
            ValueError: If max_tokens is not positive
        """
        if v is not None and v <= 0:
            raise ValueError("Max tokens must be positive")
        return v

    @field_validator("task_timeout")
    def validate_task_timeout(cls, v: int) -> int:
        """Validate task_timeout value.

        Args:
            v: Task timeout value

        Returns:
            Validated task timeout value

        Raises:
            ValueError: If task_timeout is not positive
        """
        if v <= 0:
            raise ValueError("Task timeout must be positive")
        return v

    @field_validator("max_retries")
    def validate_max_retries(cls, v: int) -> int:
        """Validate max_retries value.

        Args:
            v: Max retries value

        Returns:
            Validated max retries value

        Raises:
            ValueError: If max_retries is negative
        """
        if v < 0:
            raise ValueError("Max retries cannot be negative")
        return v

    @field_validator("max_steps")
    def validate_max_steps(cls, v: int) -> int:
        """Validate max_steps value.

        Args:
            v: Max steps value

        Returns:
            Validated max steps value

        Raises:
            ValueError: If max_steps is not positive
        """
        if v <= 0:
            raise ValueError("Max steps must be positive")
        return v


class TaskInput(BaseModel):
    """Input for a task."""

    content: str = Field(description="The task content")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for the task"
    )
    constraints: List[str] = Field(
        default_factory=list, description="Constraints for the task"
    )
    requirements: List[str] = Field(
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


class TaskOutput(BaseModel):
    """Output from a task."""

    result: str = Field(..., description="The task result")
    success: bool = Field(..., description="Whether the task was successful")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the task"
    )


class WorkflowStep(BaseModel):
    """A step in the agent's workflow."""

    name: str = Field(..., description="The name of the step")
    description: str = Field(..., description="Description of what the step does")
    input: TaskInput = Field(..., description="The input for this step")
    output: TaskOutput = Field(..., description="The output from this step")
    status: str = Field(..., description="The status of this step")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate the name field."""
        valid_steps = [step for step in AgentStep]
        if v not in valid_steps:
            raise ValueError(f"Step name must be one of {valid_steps}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate the status field."""
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v
