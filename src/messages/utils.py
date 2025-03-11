"""Message utility functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

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
    return get_message_metadata(messages[index], key, default)


def set_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    value: object,
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
    set_message_metadata(messages[index], key, value)


def get_message_metadata(message: Message, key: str, default: object = None) -> object:
    """Get message metadata.

    Args:
        message: Message to get metadata from.
        key: Metadata key.
        default: Default value if key not found.

    Returns:
        Metadata value.

    """
    # Ensure the message has additional_kwargs and metadata
    if not hasattr(message, "additional_kwargs"):
        return default

    metadata = message.additional_kwargs.get("metadata", {})
    return metadata.get(key, default)


def set_message_metadata(message: Message, key: str, value: object) -> None:
    """Set message metadata.

    Args:
        message: Message to set metadata on.
        key: Metadata key.
        value: Metadata value.

    """
    # Ensure the message has additional_kwargs
    if not hasattr(message, "additional_kwargs"):
        message.additional_kwargs = {}

    # Ensure the message has a metadata dict
    if "metadata" not in message.additional_kwargs:
        message.additional_kwargs["metadata"] = {}

    # Set the metadata value
    message.additional_kwargs["metadata"][key] = value


def parse_structured_content(
    message: Message,
    default: dict[str, object] | None = None,
) -> dict[str, object]:
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
            if get_message_metadata(message, field) is None:
                msg = f"Missing required metadata field: {field}"
                raise ConfigError(msg)

    return True
