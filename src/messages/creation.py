"""Message creation functions."""

from datetime import UTC, datetime
from typing import Any, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_types.agent_types import Message
from src.exceptions import ConfigError


def create_structured_message(
    role: str,
    content: Union[str, dict[str, Any]],
    metadata: Optional[dict[str, Any]] = None,
) -> Message:
    """Create a structured message.

    Args:
        role: Message role (human, ai, system, tool).
        content: Message content (string or dict).
        metadata: Optional message metadata.

    Returns:
        Created message.

    Raises:
        ConfigError: If role is invalid.

    """
    # Convert dict content to string if needed
    if isinstance(content, dict):
        content = str(content)

    # Create message based on role
    if role == "human":
        return HumanMessage(content=content, metadata=metadata or {})
    if role == "ai":
        return AIMessage(content=content, metadata=metadata or {})
    if role == "system":
        return SystemMessage(content=content, metadata=metadata or {})
    if role == "tool":
        return ToolMessage(content=content, metadata=metadata or {})

    msg = f"Invalid message role: {role}"
    raise ConfigError(msg)


def create_ai_message(content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
    """Create an AI message.

    Args:
        content: Message content.
        metadata: Optional message metadata.

    Returns:
        AI message.

    """
    return Message(
        content=content,
        role="assistant",
        metadata=metadata or {},
        created_at=datetime.now(UTC),
    )


def create_human_message(content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
    """Create a human message.

    Args:
        content: Message content.
        metadata: Optional message metadata.

    Returns:
        Human message.

    """
    return Message(
        content=content,
        role="user",
        metadata=metadata or {},
        created_at=datetime.now(UTC),
    )


def create_tool_message(content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
    """Create a tool message.

    Args:
        content: Message content.
        metadata: Optional message metadata.

    Returns:
        Tool message.

    """
    return Message(
        content=content,
        role="tool",
        metadata=metadata or {},
        created_at=datetime.now(UTC),
    )


def create_system_message(content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
    """Create a system message.

    Args:
        content: Message content.
        metadata: Optional message metadata.

    Returns:
        System message.

    """
    return Message(
        content=content,
        role="system",
        metadata=metadata or {},
        created_at=datetime.now(UTC),
    )


def get_message_at_index(messages: list[Message], index: int) -> Message:
    """Get message at specified index.

    Args:
        messages: List of messages.
        index: Message index.

    Returns:
        Message at index.

    Raises:
        IndexError: If index is out of range.

    """
    return messages[index]
