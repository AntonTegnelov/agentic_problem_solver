"""Message creation utilities."""

from __future__ import annotations

import json
from typing import Any

from src.common_types.message_types import (
    AIMessage,
    HumanMessage,
    Message,
    SystemMessage,
    ToolMessage,
)
from src.exceptions import ConfigError
from src.messages.chain import MessageChain


def create_message(role: str, content: str, metadata: dict[str, Any] | None = None) -> Message:
    """Create message.

    Args:
        role: Message role.
        content: Message content.
        metadata: Optional metadata.

    Returns:
        Created message.

    Raises:
        ConfigError: If role is invalid.

    """
    if metadata is None:
        metadata = {}

    if role == "human":
        return HumanMessage(content=content, metadata=metadata)
    if role == "ai":
        return AIMessage(content=content, metadata=metadata)
    if role == "system":
        return SystemMessage(content=content, metadata=metadata)
    if role == "tool":
        if "tool_call_id" not in metadata:
            msg = "tool_call_id is required for tool messages"
            raise ConfigError(msg)
        return ToolMessage(
            content=content,
            tool_call_id=metadata["tool_call_id"],
            metadata=metadata,
        )

    msg = f"Invalid role: {role}"
    raise ConfigError(msg)


def create_human_message(content: str, **kwargs: Any) -> HumanMessage:
    """Create human message.

    Args:
        content: Message content.
        **kwargs: Additional message metadata.

    Returns:
        Human message.

    """
    return HumanMessage(content=content, metadata=kwargs)


def create_ai_message(content: str, **kwargs: Any) -> AIMessage:
    """Create AI message.

    Args:
        content: Message content.
        **kwargs: Additional message metadata.

    Returns:
        AI message.

    """
    return AIMessage(content=content, metadata=kwargs)


def create_system_message(content: str, **kwargs: Any) -> SystemMessage:
    """Create system message.

    Args:
        content: Message content.
        **kwargs: Additional message metadata.

    Returns:
        System message.

    """
    return SystemMessage(content=content, metadata=kwargs)


def create_tool_message(content: str, tool_call_id: str, **kwargs: Any) -> ToolMessage:
    """Create tool message.

    Args:
        content: Message content.
        tool_call_id: Tool call ID.
        **kwargs: Additional message metadata.

    Returns:
        Tool message.

    """
    metadata = kwargs.copy()
    metadata["tool_call_id"] = tool_call_id
    return ToolMessage(content=content, tool_call_id=tool_call_id, metadata=metadata)


def create_structured_message(
    role: str,
    content: str | dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Create structured message.

    Args:
        role: Message role
        content: Message content
        metadata: Optional metadata

    Returns:
        Structured message

    """
    if metadata is None:
        metadata = {}

    metadata["structured"] = True

    # Convert dict content to JSON string
    if isinstance(content, dict):
        content = json.dumps(content)

    return create_message(role, content, metadata)


def create_message_chain() -> MessageChain:
    """Create message chain.

    Returns:
        Created message chain.

    """
    return MessageChain()
