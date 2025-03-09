"""Message wrapper module for LangChain message types."""

from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def create_system_message(
    content: str, metadata: Optional[Dict[str, Any]] = None
) -> SystemMessage:
    """Create a SystemMessage with proper initialization.

    Args:
        content: Message content
        metadata: Optional metadata dictionary

    Returns:
        Initialized SystemMessage
    """
    return SystemMessage(
        content=content,
        additional_kwargs={
            "metadata": metadata or {},
            "tool_calls": None,
            "function_call": None,
        },
    )


def create_human_message(
    content: str, metadata: Optional[Dict[str, Any]] = None
) -> HumanMessage:
    """Create a HumanMessage with proper initialization.

    Args:
        content: Message content
        metadata: Optional metadata dictionary

    Returns:
        Initialized HumanMessage
    """
    return HumanMessage(
        content=content,
        additional_kwargs={
            "metadata": metadata or {},
            "tool_calls": None,
            "function_call": None,
        },
    )


def create_ai_message(
    content: str, metadata: Optional[Dict[str, Any]] = None
) -> AIMessage:
    """Create an AIMessage with proper initialization.

    Args:
        content: Message content
        metadata: Optional metadata dictionary

    Returns:
        Initialized AIMessage
    """
    return AIMessage(
        content=content,
        additional_kwargs={
            "metadata": metadata or {},
            "tool_calls": None,
            "function_call": None,
        },
    )


def create_tool_message(
    content: str, tool_call_id: str, metadata: Optional[Dict[str, Any]] = None
) -> ToolMessage:
    """Create a ToolMessage with proper initialization.

    Args:
        content: Message content
        tool_call_id: ID of the tool call
        metadata: Optional metadata dictionary

    Returns:
        Initialized ToolMessage
    """
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        additional_kwargs={
            "metadata": metadata or {},
            "tool_calls": None,
            "function_call": None,
        },
    )


def get_message_metadata(message: Any, key: str, default: Any = None) -> Any:
    """Get metadata from a message.

    Args:
        message: Message object
        key: Metadata key
        default: Default value if key not found

    Returns:
        Metadata value or default
    """
    if message is None:
        return default

    if key == "content":
        return message.content
    elif key == "type":
        return message.type
    elif key == "tool_call_id" and isinstance(message, ToolMessage):
        return message.tool_call_id
    metadata = message.additional_kwargs.get("metadata", {})
    return metadata.get(key, default)


def set_message_metadata(message: Any, key: str, value: Any) -> None:
    """Set metadata for a message.

    Args:
        message: Message object
        key: Metadata key
        value: Metadata value
    """
    if key == "content":
        message.content = value
        return
    if "metadata" not in message.additional_kwargs:
        message.additional_kwargs["metadata"] = {}
    message.additional_kwargs["metadata"][key] = value
