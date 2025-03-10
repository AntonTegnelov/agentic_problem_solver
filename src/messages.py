"""Message handling module."""

from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_types.agent_types import Message

T = TypeVar("T")
MessageValue = (
    str | int | float | bool | dict[str, "MessageValue"] | list["MessageValue"] | None
)


def create_system_message(
    content: str,
    metadata: dict[str, object] | None = None,
) -> SystemMessage:
    """Create a SystemMessage with proper initialization.

    Args:
        content: The message content.
        metadata: Optional metadata to attach to the message.

    Returns:
        A SystemMessage instance.

    """
    if metadata is None:
        metadata = {}
    return SystemMessage(content=content, additional_kwargs={"metadata": metadata})


def create_human_message(
    content: str,
    metadata: dict[str, object] | None = None,
) -> HumanMessage:
    """Create a HumanMessage with proper initialization.

    Args:
        content: The message content.
        metadata: Optional metadata to attach to the message.

    Returns:
        A HumanMessage instance.

    """
    if metadata is None:
        metadata = {}
    return HumanMessage(content=content, additional_kwargs={"metadata": metadata})


def create_ai_message(
    content: str,
    metadata: dict[str, object] | None = None,
) -> AIMessage:
    """Create an AIMessage with proper initialization.

    Args:
        content: The message content.
        metadata: Optional metadata to attach to the message.

    Returns:
        An AIMessage instance.

    """
    if metadata is None:
        metadata = {}
    return AIMessage(content=content, additional_kwargs={"metadata": metadata})


def create_tool_message(
    content: str,
    tool_call_id: str,
    metadata: dict[str, object] | None = None,
) -> ToolMessage:
    """Create a ToolMessage with proper initialization.

    Args:
        content: The message content.
        tool_call_id: The ID of the tool call.
        metadata: Optional metadata to attach to the message.

    Returns:
        A ToolMessage instance.

    """
    if metadata is None:
        metadata = {}
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        additional_kwargs={"metadata": metadata},
    )


def get_message_metadata(
    message: Message | HumanMessage | AIMessage | SystemMessage | ToolMessage,
    key: str,
    default: T | None = None,
) -> T | None:
    """Get metadata from a message.

    Args:
        message: The message to get metadata from.
        key: The metadata key to get.
        default: The default value to return if the key is not found.

    Returns:
        The metadata value or default if not found.

    """
    if key == "content":
        return message.content  # type: ignore[attr-defined]
    if key == "type":
        return message.type  # type: ignore[attr-defined]
    if key == "tool_call_id" and isinstance(message, ToolMessage):
        return message.tool_call_id  # type: ignore[attr-defined]

    if hasattr(message, "metadata"):
        metadata = getattr(message, "metadata", {})
        return metadata.get(key, default)
    if hasattr(message, "additional_kwargs"):
        return message.additional_kwargs.get("metadata", {}).get(key, default)  # type: ignore[union-attr]
    return default


def set_message_metadata(
    message: Message | HumanMessage | AIMessage | SystemMessage | ToolMessage,
    key: str,
    value: MessageValue,
) -> None:
    """Set metadata for a message.

    Args:
        message: The message to set metadata for.
        key: The metadata key to set.
        value: The value to set.

    """
    if key == "content":
        message.content = str(value)  # type: ignore[attr-defined]
        return
    if key == "type":
        message.type = str(value)  # type: ignore[attr-defined]
        return
    if key == "tool_call_id" and isinstance(message, ToolMessage):
        message.tool_call_id = str(value)  # type: ignore[attr-defined]
        return

    if hasattr(message, "metadata"):
        if not hasattr(message, "metadata"):
            message.metadata = {}  # type: ignore[attr-defined]
        message.metadata[key] = value  # type: ignore[attr-defined]
    elif hasattr(message, "additional_kwargs"):
        if "metadata" not in message.additional_kwargs:  # type: ignore[union-attr]
            message.additional_kwargs["metadata"] = {}  # type: ignore[union-attr]
        message.additional_kwargs["metadata"][key] = value  # type: ignore[union-attr]


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
    """Get metadata from a message at the specified index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        default: Default value if key not found.

    Returns:
        Message metadata value.

    """
    message = get_message_at_index(messages, index)
    return get_message_metadata(message, key, default)


def set_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    value: MessageValue,
) -> None:
    """Set metadata for a message at the specified index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        value: Metadata value.

    """
    message = get_message_at_index(messages, index)
    set_message_metadata(message, key, value)
