"""Message creation utilities."""

from __future__ import annotations

import json
from typing import Any

from src.common_types.error_types import ConfigError
from src.common_types.message_types import (
    AIMessage,
    HumanMessage,
    Message,
    SystemMessage,
    ToolMessage,
)
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

    additional_kwargs = {"metadata": metadata} if metadata else {}

    if role == "human":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "ai":
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "tool":
        if "tool_call_id" not in metadata:
            msg = "Tool messages require a tool_call_id in metadata"
            raise ConfigError(msg)
        return ToolMessage(
            content=content,
            tool_call_id=metadata["tool_call_id"],
            additional_kwargs=additional_kwargs,
        )

    msg = f"Invalid message role: {role}"
    raise ConfigError(msg)


def create_human_message(content: str, metadata: dict[str, Any] | None = None, **kwargs: object) -> HumanMessage:
    """Create human message.

    Args:
        content: Message content.
        metadata: Optional metadata.
        **kwargs: Additional keyword arguments.

    Returns:
        Human message.

    """
    if metadata is None:
        metadata = {}

    # Add any kwargs to metadata
    if kwargs:
        metadata.update(kwargs)

    additional_kwargs = {"metadata": metadata} if metadata else {}

    return HumanMessage(content=content, additional_kwargs=additional_kwargs)


def create_ai_message(content: str, metadata: dict[str, Any] | None = None, **kwargs: object) -> AIMessage:
    """Create AI message.

    Args:
        content: Message content.
        metadata: Optional metadata.
        **kwargs: Additional keyword arguments.

    Returns:
        AI message.

    """
    if metadata is None:
        metadata = {}

    # Add any kwargs to metadata
    if kwargs:
        metadata.update(kwargs)

    additional_kwargs = {"metadata": metadata} if metadata else {}

    return AIMessage(content=content, additional_kwargs=additional_kwargs)


def create_system_message(content: str, metadata: dict[str, Any] | None = None, **kwargs: object) -> SystemMessage:
    """Create system message.

    Args:
        content: Message content.
        metadata: Optional metadata.
        **kwargs: Additional keyword arguments.

    Returns:
        System message.

    """
    if metadata is None:
        metadata = {}

    # Add any kwargs to metadata
    if kwargs:
        metadata.update(kwargs)

    additional_kwargs = {"metadata": metadata} if metadata else {}

    return SystemMessage(content=content, additional_kwargs=additional_kwargs)


def create_tool_message(
    content: str,
    tool_call_id: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: object,
) -> ToolMessage:
    """Create tool message.

    Args:
        content: Message content.
        tool_call_id: Tool call ID.
        metadata: Optional metadata.
        **kwargs: Additional keyword arguments.

    Returns:
        Tool message.

    """
    if metadata is None:
        metadata = {}

    # Add tool_call_id to metadata
    metadata["tool_call_id"] = tool_call_id

    # Add any kwargs to metadata
    if kwargs:
        metadata.update(kwargs)

    additional_kwargs = {"metadata": metadata} if metadata else {}

    return ToolMessage(content=content, tool_call_id=tool_call_id, additional_kwargs=additional_kwargs)


def create_structured_message(
    role: str,
    content: str | dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Create structured message.

    Args:
        role: Message role.
        content: Message content.
        metadata: Optional metadata.

    Returns:
        Message instance.

    Raises:
        ConfigError: If role is invalid.

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
