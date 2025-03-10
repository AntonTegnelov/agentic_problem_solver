"""Message handling module."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_types.agent_types import Message
from src.exceptions import ConfigError

T = TypeVar("T")
MessageValue = (
    str | int | float | bool | dict[str, "MessageValue"] | list["MessageValue"] | None
)


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MessageChain:
    """Message chain for tracking conversation history."""

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, MessageValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_message(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Add message to chain with metadata.

        Args:
            message: Message to add.
            priority: Message priority.

        """
        set_message_metadata(message, "timestamp", datetime.now(UTC).isoformat())
        set_message_metadata(message, "priority", priority.value)
        self.messages.append(message)
        self.last_updated = datetime.now(UTC)

    def validate_chain(self) -> bool:
        """Validate message chain structure and content.

        Returns:
            True if chain is valid.

        Raises:
            ConfigError: If chain validation fails.

        """
        if not self.messages:
            return True

        # Check message sequence
        for i, current_msg in enumerate(self.messages[1:], 1):
            prev = self.messages[i - 1]

            # Validate message types follow expected pattern
            if isinstance(prev, HumanMessage) and not isinstance(
                current_msg,
                AIMessage | ToolMessage,
            ):
                error_msg = (
                    "Invalid message sequence: Human message must be followed by "
                    "AI or Tool message"
                )
                raise ConfigError(error_msg)

            if isinstance(prev, AIMessage) and not isinstance(
                current_msg,
                HumanMessage | ToolMessage,
            ):
                error_msg = (
                    "Invalid message sequence: AI message must be followed by "
                    "Human or Tool message"
                )
                raise ConfigError(error_msg)

            # Validate timestamps are sequential
            prev_time = get_message_metadata(prev, "timestamp")
            curr_time = get_message_metadata(current_msg, "timestamp")
            if prev_time and curr_time and prev_time > curr_time:
                error_msg = (
                    "Invalid message sequence: Messages must be in chronological order"
                )
                raise ConfigError(error_msg)

        return True

    def get_messages_by_type(self, msg_type: type[Message]) -> Iterator[Message]:
        """Get messages of specified type.

        Args:
            msg_type: Message type to filter by.

        Yields:
            Messages of specified type.

        """
        for msg in self.messages:
            if isinstance(msg, msg_type):
                yield msg

    def get_messages_by_priority(
        self,
        min_priority: MessagePriority = MessagePriority.LOW,
    ) -> Iterator[Message]:
        """Get messages with minimum priority level.

        Args:
            min_priority: Minimum priority level.

        Yields:
            Messages meeting priority threshold.

        """
        for msg in self.messages:
            priority = get_message_metadata(
                msg,
                "priority",
                MessagePriority.NORMAL.value,
            )
            if priority >= min_priority.value:
                yield msg

    def search_messages(
        self,
        query: str,
        metadata_key: str | None = None,
    ) -> Iterator[Message]:
        """Search messages by content or metadata.

        Args:
            query: Search query string.
            metadata_key: Optional metadata key to search in.

        Yields:
            Matching messages.

        """
        query = query.lower()
        for msg in self.messages:
            if metadata_key:
                value = get_message_metadata(msg, metadata_key)
                if value and str(value).lower().find(query) != -1:
                    yield msg
            elif msg.content.lower().find(query) != -1:  # type: ignore[union-attr]
                yield msg

    def validate_message_chain(self) -> bool:
        """Validate the message chain."""
        # Implement message chain validation using existing Message class
        # Placeholder for actual validation logic
        return True

    def filter_messages(self) -> list[Message]:
        """Filter messages based on provided criteria."""
        # Create message filtering and search utilities
        # Placeholder for actual filtering logic
        return []  # Placeholder for actual filtering logic


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


def create_message_chain() -> MessageChain:
    """Create a new message chain.

    Returns:
        New message chain instance.

    """
    return MessageChain()


def validate_message_content(
    message: Message,
    required_fields: list[str] | None = None,
) -> bool:
    """Validate message content structure.

    Args:
        message: Message to validate.
        required_fields: Optional list of required metadata fields.

    Returns:
        True if content is valid.

    Raises:
        ConfigError: If content validation fails.

    """
    if not message.content:  # type: ignore[union-attr]
        msg = "Message content cannot be empty"
        raise ConfigError(msg)

    if required_fields:
        for field in required_fields:
            if not get_message_metadata(message, field):
                msg = f"Required metadata field missing: {field}"
                raise ConfigError(msg)

    return True


class MessageHandler:
    """Handles message processing and validation."""

    def handle_message(self, message: Message) -> None:
        """Process a message."""
        # ... existing handling logic ...
        self.track_message_history(message)
        # ... existing handling logic ...

    def track_message_history(self, message: Message) -> None:
        """Track message history with metadata."""
        # Implement message history tracking with metadata
        # Store message in a history log

    def validate_message_chain(self) -> bool:
        """Validate the message chain."""
        # Implement message chain validation using existing Message class
        # Placeholder for actual validation logic
        return True

    def filter_messages(self) -> list[Message]:
        """Filter messages based on provided criteria."""
        # Create message filtering and search utilities
        # Placeholder for actual filtering logic
        return []
