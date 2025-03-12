"""Message type definitions."""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    BaseMessage as Message,
)

# Type alias for message content
MessageValue = str | int | float | bool | dict[str, Any] | list[Any] | None

# Type alias for message criteria
CriteriaValue = str | int | float | bool | None
CriteriaDict = dict[str, CriteriaValue]

T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "AIMessage",
    "CriteriaDict",
    "CriteriaValue",
    "HumanMessage",
    "Message",
    "MessageValue",
    "SystemMessage",
    "ToolMessage",
]
