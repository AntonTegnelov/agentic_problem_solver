"""Message utility functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

from src.common_types.error_types import ConfigError

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


# Sender/Receiver Hierarchy Information


def get_sender_id(message: Message) -> str | None:
    """Get sender agent ID from message metadata.

    Args:
        message: Message to get sender ID from.

    Returns:
        Sender agent ID or None if not set.

    """
    return get_message_metadata(message, "sender_id")


def set_sender_id(message: Message, agent_id: str) -> None:
    """Set sender agent ID in message metadata.

    Args:
        message: Message to set sender ID on.
        agent_id: Sender agent ID.

    """
    set_message_metadata(message, "sender_id", agent_id)


def get_receiver_id(message: Message) -> str | None:
    """Get receiver agent ID from message metadata.

    Args:
        message: Message to get receiver ID from.

    Returns:
        Receiver agent ID or None if not set.

    """
    return get_message_metadata(message, "receiver_id")


def set_receiver_id(message: Message, agent_id: str) -> None:
    """Set receiver agent ID in message metadata.

    Args:
        message: Message to set receiver ID on.
        agent_id: Receiver agent ID.

    """
    set_message_metadata(message, "receiver_id", agent_id)


def get_sender_parent_id(message: Message) -> str | None:
    """Get sender's parent agent ID from message metadata.

    Args:
        message: Message to get sender's parent ID from.

    Returns:
        Sender's parent agent ID or None if not set.

    """
    return get_message_metadata(message, "sender_parent_id")


def set_sender_parent_id(message: Message, parent_id: str | None) -> None:
    """Set sender's parent agent ID in message metadata.

    Args:
        message: Message to set sender's parent ID on.
        parent_id: Sender's parent agent ID or None if no parent.

    """
    set_message_metadata(message, "sender_parent_id", parent_id)


def get_receiver_parent_id(message: Message) -> str | None:
    """Get receiver's parent agent ID from message metadata.

    Args:
        message: Message to get receiver's parent ID from.

    Returns:
        Receiver's parent agent ID or None if not set.

    """
    return get_message_metadata(message, "receiver_parent_id")


def set_receiver_parent_id(message: Message, parent_id: str | None) -> None:
    """Set receiver's parent agent ID in message metadata.

    Args:
        message: Message to set receiver's parent ID on.
        parent_id: Receiver's parent agent ID or None if no parent.

    """
    set_message_metadata(message, "receiver_parent_id", parent_id)


def get_hierarchy_path(message: Message) -> list[str] | None:
    """Get the hierarchical path the message has traveled through.

    The path is a list of agent IDs representing the chain of delegation,
    from the original sender down to the current receiver.

    Args:
        message: Message to get hierarchy path from.

    Returns:
        List of agent IDs representing the hierarchy path or None if not set.

    """
    return get_message_metadata(message, "hierarchy_path")


def set_hierarchy_path(message: Message, path: list[str]) -> None:
    """Set the hierarchical path the message has traveled through.

    Args:
        message: Message to set hierarchy path on.
        path: List of agent IDs representing the hierarchy path.

    """
    set_message_metadata(message, "hierarchy_path", path.copy())


def add_to_hierarchy_path(message: Message, agent_id: str) -> None:
    """Add an agent ID to the hierarchy path.

    Args:
        message: Message to update hierarchy path on.
        agent_id: Agent ID to add to the path.

    """
    path = get_hierarchy_path(message) or []
    if agent_id not in path:
        path.append(agent_id)
        set_hierarchy_path(message, path)


def is_hierarchical_message(message: Message) -> bool:
    """Check if message contains hierarchy information.

    Args:
        message: Message to check.

    Returns:
        True if message contains sender and receiver IDs.

    """
    return get_sender_id(message) is not None and get_receiver_id(message) is not None
