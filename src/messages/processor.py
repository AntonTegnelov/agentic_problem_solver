"""Message processor module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.common_types import Message
from src.exceptions import ConfigError

if TYPE_CHECKING:
    from src.common_types import Message

T = TypeVar("T")


def set_metadata_at_index(messages: list[Message], index: int, key: str, value: Any) -> None:
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


def get_metadata_at_index(messages: list[Message], index: int, key: str) -> Any:
    """Get metadata value from message at index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.

    Returns:
        Metadata value.

    Raises:
        IndexError: If index is out of range.

    """
    return messages[index].metadata.get(key)


def parse_structured_content(message: Message) -> dict[str, Any]:
    """Parse structured message content.

    Args:
        message: Message to parse.

    Returns:
        Parsed content as dictionary.

    Raises:
        ConfigError: If content cannot be parsed.

    """
    try:
        if isinstance(message.content, dict):
            return message.content
        return json.loads(message.content)
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        msg = f"Failed to parse message content: {e}"
        raise ConfigError(msg) from e


def validate_message_content(message: Message, required_fields: list[str] | None = None) -> bool:
    """Validate message content.

    Args:
        message: Message to validate.
        required_fields: Optional list of required metadata fields.

    Returns:
        True if message is valid.

    Raises:
        ConfigError: If message validation fails.

    """
    # Check for empty content
    if not message.content:
        msg = "Message content cannot be empty"
        raise ConfigError(msg)

    # Check required metadata fields
    if required_fields:
        for field in required_fields:
            if field not in message.metadata:
                msg = f"Missing required metadata field: {field}"
                raise ConfigError(msg)

    return True


class MessageProcessor(Protocol):
    """Message processor protocol."""

    def process(self, message: Message) -> Message:
        """Process a message.

        Args:
            message: Message to process.

        Returns:
            Processed message.

        Raises:
            ConfigError: If message processing fails.

        """
        ...

    def validate(self, message: Message) -> bool:
        """Validate a message.

        Args:
            message: Message to validate.

        Returns:
            True if message is valid.

        Raises:
            ConfigError: If message validation fails.

        """
        ...


def create_message_from_dict(data: dict[str, Any]) -> Message:
    """Create message from dictionary.

    Args:
        data: Message data.

    Returns:
        Created message.

    Raises:
        ConfigError: If message creation fails.

    """
    if not isinstance(data, dict):
        msg = f"Invalid message data type: {type(data)}"
        raise ConfigError(msg)

    required_fields = ["role", "content"]
    for field in required_fields:
        if field not in data:
            msg = f"Missing required field: {field}"
            raise ConfigError(msg)

    role = data["role"]
    content = data["content"]
    metadata = data.get("metadata", {})

    if role == "human":
        return HumanMessage(content=content, metadata=metadata)
    if role == "ai":
        return AIMessage(content=content, metadata=metadata)
    if role == "system":
        return SystemMessage(content=content, metadata=metadata)
    if role == "tool":
        return ToolMessage(content=content, metadata=metadata)
    msg = f"Invalid message role: {role}"
    raise ConfigError(msg)


def get_message_metadata(message: Message, key: str) -> Any:
    """Get message metadata value.

    Args:
        message: Message to get metadata from.
        key: Metadata key.

    Returns:
        Metadata value.

    """
    return message.metadata.get(key)


def set_message_metadata(message: Message, key: str, value: Any) -> None:
    """Set message metadata value.

    Args:
        message: Message to set metadata on.
        key: Metadata key.
        value: Metadata value.

    """
    message.metadata[key] = value
