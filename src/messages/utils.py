"""Message utility functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar

from src.exceptions import ConfigError

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage as Message

T = TypeVar("T")


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
    return messages[index]


def get_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    default: T | None = None,
) -> T | None:
    """Get metadata value from message at index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        default: Default value if key not found.

    Returns:
        Metadata value.

    Raises:
        IndexError: If index is out of range.

    """
    return messages[index].metadata.get(key, default)


def set_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    value: Any,
) -> None:
    """Set metadata value for message at index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        value: Metadata value.

    Raises:
        IndexError: If index is out of range.

    """
    messages[index].metadata[key] = value


def get_message_metadata(message: Message, key: str, default: Any = None) -> Any:
    """Get message metadata.

    Args:
        message: Message to get metadata from.
        key: Metadata key.
        default: Default value if key not found.

    Returns:
        Metadata value.

    """
    return message.metadata.get(key, default)


def set_message_metadata(message: Message, key: str, value: Any) -> None:
    """Set message metadata.

    Args:
        message: Message to set metadata on.
        key: Metadata key.
        value: Metadata value.

    """
    message.metadata[key] = value


def parse_structured_content(
    message: Message,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse structured message content.

    Args:
        message: Message to parse.
        default: Default value if parsing fails.

    Returns:
        Parsed content as dictionary.

    Raises:
        ConfigError: If content cannot be parsed and no default provided.

    """
    try:
        if isinstance(message.content, dict):
            return message.content
        return json.loads(message.content)
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        if default is not None:
            return default
        msg = f"Failed to parse message content: {e}"
        raise ConfigError(msg) from e


def validate_message_content(message: Message, required_fields: list[str] | None = None) -> bool:
    """Validate message content.

    Args:
        message: Message to validate.
        required_fields: Optional list of required fields.

    Returns:
        True if message is valid.

    Raises:
        ConfigError: If message validation fails.

    """
    # Check for empty content
    if not message.content:
        msg = "Message content cannot be empty"
        raise ConfigError(msg)

    # Check required fields in content
    if required_fields and isinstance(message.content, dict):
        for field in required_fields:
            if field not in message.content:
                msg = f"Missing required field: {field}"
                raise ConfigError(msg)

    # Check required fields in metadata
    if required_fields and not isinstance(message.content, dict):
        for field in required_fields:
            if field not in message.metadata:
                msg = f"Missing required metadata field: {field}"
                raise ConfigError(msg)

    return True
