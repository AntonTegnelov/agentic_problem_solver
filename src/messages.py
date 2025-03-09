"""Message wrapper module for LangChain message types."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def create_system_message(
    content: str, metadata: dict[str, object] | None = None
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
    content: str, metadata: dict[str, object] | None = None
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
    content: str, metadata: dict[str, object] | None = None
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
    content: str, tool_call_id: str, metadata: dict[str, object] | None = None
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
    message: HumanMessage | AIMessage | SystemMessage | ToolMessage,
    key: str,
    default: object | None = None,
) -> object | None:
    """Get metadata from a message.

    Args:
        message: The message to get metadata from.
        key: The metadata key to get.
        default: The default value to return if the key is not found.

    Returns:
        The metadata value or default if not found.
    """
    if key == "content":
        return message.content
    if key == "type":
        return message.type
    if key == "tool_call_id" and isinstance(message, ToolMessage):
        return message.tool_call_id
    return message.additional_kwargs.get("metadata", {}).get(key, default)


def set_message_metadata(
    message: HumanMessage | AIMessage | SystemMessage | ToolMessage,
    key: str,
    value: object,
) -> None:
    """Set metadata for a message.

    Args:
        message: The message to set metadata for.
        key: The metadata key to set.
        value: The value to set.
    """
    if key == "content":
        message.content = str(value)
        return
    if key == "type":
        message.type = str(value)
        return
    if key == "tool_call_id" and isinstance(message, ToolMessage):
        message.tool_call_id = str(value)
        return

    if "metadata" not in message.additional_kwargs:
        message.additional_kwargs["metadata"] = {}
    message.additional_kwargs["metadata"][key] = value
