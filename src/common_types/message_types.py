"""Message type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Type alias for Message
Message = Union["Message", "HumanMessage", "AIMessage", "SystemMessage", "ToolMessage"]

# Type alias for message content
MessageValue = Union[str, int, float, bool, dict[str, Any], list[Any], None]

# Type alias for message criteria
CriteriaValue = Union[str, int, float, bool, None]
CriteriaDict = dict[str, CriteriaValue]

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Message:
    """Base message class."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Initialize message.

        Args:
            content: Message content.
            metadata: Optional message metadata.

        """
        self.content = content
        self.metadata = metadata or {}

    @property
    def type(self) -> str:
        """Get message type."""
        return self.__class__.__name__.lower().replace("message", "")


@dataclass
class SystemMessage(Message):
    """System message."""


@dataclass
class HumanMessage(Message):
    """Human message."""


@dataclass
class AIMessage(Message):
    """AI message."""


@dataclass
class ToolMessage(Message):
    """Tool message."""

    tool_call_id: str = field(default="")

    def __post_init__(self) -> None:
        """Post init hook."""
        if not self.tool_call_id:
            msg = "tool_call_id is required"
            raise ValueError(msg)
        if "tool_call_id" not in self.metadata:
            self.metadata["tool_call_id"] = self.tool_call_id
