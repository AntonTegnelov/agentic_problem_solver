"""Message types and utilities."""

from __future__ import annotations

import json
from enum import Enum
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

from src.exceptions import ConfigError

T = TypeVar("T")


class MessagePriority(Enum):
    """Message priority."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


def get_message_at_index(messages: list[Message], index: int) -> Message:
    """Get message at index.

    Args:
        messages: List of messages.
        index: Message index.

    Returns:
        Message at index.

    Raises:
        IndexError: If index is out of range.

    """
    try:
        return messages[index]
    except IndexError as e:
        msg = f"Message index out of range: {index}"
        raise IndexError(msg) from e


def get_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    default: T | None = None,
) -> T | None:
    """Get metadata from a message at the specified index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        default: Default value if key not found.

    Returns:
        Message metadata value.

    """
    try:
        message = messages[index]
        return message.metadata.get(key, default)
    except IndexError:
        return default


def set_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    value: dict[str, Any],
) -> None:
    """Set metadata for a message at the specified index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        value: Metadata value.

    Raises:
        IndexError: If index is out of range.

    """
    try:
        message = messages[index]
        message.metadata[key] = value
    except IndexError as e:
        msg = f"Message index out of range: {index}"
        raise IndexError(msg) from e


def get_message_metadata(message: Message, key: str, default: Any = None) -> Any:
    """Get message metadata.

    Args:
        message: Message to get metadata from
        key: Metadata key
        default: Default value if key not found

    Returns:
        Metadata value

    """
    return message.metadata.get(key, default)


def set_message_metadata(message: Message, key: str, value: Any) -> None:
    """Set message metadata.

    Args:
        message: Message to set metadata on
        key: Metadata key
        value: Metadata value

    """
    message.metadata[key] = value


def parse_structured_content(
    message: Message,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse structured content from message.

    Args:
        message: Message to parse.
        default: Default value if parsing fails.

    Returns:
        Parsed content.

    Raises:
        ValueError: If content is not valid JSON.

    """
    try:
        if not message.content:
            if default is not None:
                return default
            msg = "Empty message content"
            raise ValueError(msg)
        return json.loads(message.content)
    except json.JSONDecodeError as e:
        if default is not None:
            return default
        msg = f"Invalid JSON content: {e}"
        raise ValueError(msg) from e


def validate_message_content(message: Message, required_fields: list[str] | None = None) -> bool:
    """Validate message content.

    Args:
        message: Message to validate
        required_fields: Optional list of required metadata fields

    Returns:
        True if content is valid

    Raises:
        ConfigError: If content is invalid

    """
    if not message.content:
        msg = "Empty message content"
        raise ConfigError(msg)

    # Check required metadata fields
    if required_fields:
        for field in required_fields:
            if not get_message_metadata(message, field):
                msg = f"Missing required metadata field: {field}"
                raise ConfigError(msg)

    # For structured messages, validate JSON content
    if get_message_metadata(message, "structured"):
        try:
            if isinstance(message.content, str):
                json.loads(message.content)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON content: {e!s}"
            raise ConfigError(msg) from e

    return True


def create_structured_message(role: str, content: str) -> Message:
    """Create a structured message based on role.

    Args:
        role: Message role.
        content: Message content.

    Returns:
        Structured message.

    Raises:
        ConfigError: If role is invalid.

    """
    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        return ToolMessage(content=content)
    msg = f"Invalid message role: {role}"
    raise ConfigError(msg)


from .chain import MessageChain
from .creation import (
    create_ai_message,
    create_human_message,
    create_message,
    create_message_chain,
    create_structured_message,
    create_system_message,
    create_tool_message,
)
from .handler import MessageHandler
from .processor import MessageProcessor
from .router import MessageRouter

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
