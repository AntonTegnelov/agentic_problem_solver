"""Message type definitions."""

from __future__ import annotations

from typing import Any, TypeVar, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

# Type alias for Message
Message = Union[BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage]

# Type alias for message content
MessageValue = Union[str, int, float, bool, dict[str, Any], list[Any], None]

# Type alias for message criteria
CriteriaValue = Union[str, int, float, bool, None]
CriteriaDict = dict[str, CriteriaValue]

T = TypeVar("T")
U = TypeVar("U")
