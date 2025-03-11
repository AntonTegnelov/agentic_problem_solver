"""Message types and utilities."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

# Import submodules at the top to avoid E402 errors
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    BaseMessage as Message,
)

# Import local modules at the top
from .chain import MessageChain
from .creation import (
    create_ai_message,
    create_human_message,
    create_message,
    create_message_chain,
    create_system_message,
    create_tool_message,
)
from .handler import MessageHandler
from .processor import MessageProcessor
from .router import MessageRouter
from .utils import (
    get_message_at_index,
    get_message_metadata,
    get_metadata_at_index,
    parse_structured_content,
    set_message_metadata,
    set_metadata_at_index,
    validate_message_content,
)

T = TypeVar("T")


class MessagePriority(Enum):
    """Message priority."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


def create_structured_message(
    role: str,
    content: str | dict[str, object],
    metadata: dict[str, object] | None = None,
) -> Message:
    """Create structured message.

    Args:
        role: Message role.
        content: Message content (string or dictionary).
        metadata: Optional message metadata.

    Returns:
        Message instance.

    Raises:
        ConfigError: If role is invalid.

    """
    # Import here to avoid circular imports
    import json

    from src.exceptions import ConfigError

    # Convert dictionary content to JSON string
    if isinstance(content, dict):
        content = json.dumps(content)

    # Initialize metadata if None
    if metadata is None:
        metadata = {}

    additional_kwargs = {"metadata": metadata} if metadata else {}

    if role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "human":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "ai":
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "tool":
        if "tool_call_id" not in metadata:
            msg = "Tool messages require a tool_call_id in metadata"
            raise ConfigError(msg)
        return ToolMessage(content=content, tool_call_id=metadata["tool_call_id"], additional_kwargs=additional_kwargs)
    msg = f"Invalid message role: {role}"
    raise ConfigError(msg)


__all__ = [
    "AIMessage",
    "HumanMessage",
    "Message",
    "MessageChain",
    "MessageHandler",
    "MessagePriority",
    "MessageProcessor",
    "MessageRouter",
    "SystemMessage",
    "ToolMessage",
    "create_ai_message",
    "create_human_message",
    "create_message",
    "create_message_chain",
    "create_structured_message",
    "create_system_message",
    "create_tool_message",
    "get_message_at_index",
    "get_message_metadata",
    "get_metadata_at_index",
    "parse_structured_content",
    "set_message_metadata",
    "set_metadata_at_index",
    "validate_message_content",
]
