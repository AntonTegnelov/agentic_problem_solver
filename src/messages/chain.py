"""Message chain module."""

# This is the primary implementation of MessageChain.
# The src/messages.py file is now a compatibility layer that imports from this file.
# Issue: #123

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage as Message

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.common_types import Message

from src.common_types.enums import MessagePriority
from src.common_types.error_types import ConfigError
from src.common_types.message_types import (
    AIMessage,
    CriteriaValue,
    HumanMessage,
)
from src.messages.utils import get_message_metadata, set_message_metadata


def create_message_chain() -> MessageChain:
    """Create a new message chain.

    Returns:
        Empty message chain.

    """
    return MessageChain()


@dataclass
class MessageChain:
    """Message chain for tracking conversation history."""

    def __init__(self, messages: list[Message] | None = None) -> None:
        """Initialize message chain.

        Args:
            messages: Optional list of initial messages.

        """
        self._messages: list[Message] = messages or []
        self.metadata: dict[str, Any] = {}
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)
        self.agent_id = str(uuid.uuid4())
        self.parent_agent_id = None

    def __getitem__(self, index: int) -> Message:
        """Get message at index.

        Args:
            index: Message index.

        Returns:
            Message at index.

        Raises:
            IndexError: If index out of range.

        """
        return self._messages[index]

    def __len__(self) -> int:
        """Get number of messages.

        Returns:
            Number of messages.

        """
        return len(self._messages)

    @property
    def messages(self) -> list[Message]:
        """Get messages list.

        Returns:
            List of messages.

        """
        return self._messages

    def add_message(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Add message to chain.

        Args:
            message: Message to add.
            priority: Message priority.

        """
        set_message_metadata(message, "priority", priority.value)
        set_message_metadata(message, "timestamp", datetime.now(UTC).isoformat())
        set_message_metadata(message, "sequence", len(self._messages) + 1)
        self._messages.append(message)
        self.updated_at = datetime.now(UTC)

    def get_messages(self) -> list[Message]:
        """Get all messages in chain.

        Returns:
            List of messages.

        """
        return self._messages

    def get_messages_by_priority(self, min_priority: MessagePriority) -> list[Message]:
        """Get messages with priority >= min_priority.

        Args:
            min_priority: Minimum priority

        Returns:
            List of messages

        """
        filtered = []
        for message in self._messages:
            priority = get_message_metadata(message, "priority")
            if priority and MessagePriority(priority).value >= min_priority.value:
                filtered.append(message)
        return filtered

    def get_message_history(self, limit: int | None = None) -> list[Message]:
        """Get message history.

        Args:
            limit: Optional limit on number of messages to return.

        Returns:
            List of messages, most recent first.

        """
        messages = self._messages.copy()
        if limit:
            messages = messages[-limit:]
        return messages

    def filter_messages(
        self,
        criteria: dict[str, Any] | None = None,
        filter_fn: Callable[[Message], bool] | None = None,
        **kwargs: CriteriaValue,
    ) -> list[Message]:
        """Filter messages by criteria.

        Args:
            criteria: Filter criteria.
            filter_fn: Custom filter function.
            **kwargs: Additional keyword criteria.

        Returns:
            Filtered messages.

        """
        filtered = self._messages

        # Combine dictionary and keyword criteria
        all_criteria = {}
        if criteria:
            all_criteria.update(criteria)
        if kwargs:
            all_criteria.update(kwargs)

        if all_criteria:
            filtered = [
                msg
                for msg in filtered
                if all(
                    (key == "type" and isinstance(msg, value)) or (get_message_metadata(msg, key) == value)
                    for key, value in all_criteria.items()
                )
            ]

        if filter_fn:
            filtered = [msg for msg in filtered if filter_fn(msg)]

        return filtered

    def validate_chain(self) -> bool:
        """Validate message chain.

        Returns:
            True if chain is valid.

        Raises:
            ConfigError: If chain is invalid.

        """
        if not self._messages:
            return True

        for i in range(len(self._messages) - 1):
            current = self._messages[i]
            next_msg = self._messages[i + 1]

            # Check for consecutive messages of same type
            if current.type == next_msg.type:
                msg = f"Invalid message sequence: consecutive {current.type} messages"
                raise ConfigError(msg)

            # Check valid sequences
            if current.type == "human":
                if next_msg.type not in ["ai", "tool"]:
                    msg = "Human message must be followed by AI or tool message"
                    raise ConfigError(msg)
            elif current.type == "ai":
                if next_msg.type not in ["human", "tool"]:
                    msg = "AI message must be followed by human or tool message"
                    raise ConfigError(msg)
            elif current.type == "tool" and next_msg.type not in ["human", "ai"]:
                msg = "Tool message must be followed by human or AI message"
                raise ConfigError(msg)

        return True

    def search_messages(self, query: str, field: str | None = None) -> list[Message]:
        """Search messages by content or metadata field.

        Args:
            query: Search query.
            field: Optional metadata field to search.

        Returns:
            List of matching messages.

        """
        return [
            msg
            for msg in self._messages
            if (field is None and query.lower() in str(msg.content).lower())
            or (field is not None and query.lower() in str(get_message_metadata(msg, field, "")).lower())
        ]

    def validate_message_chain(self) -> bool:
        """Validate message chain.

        Returns:
            True if chain is valid.

        Raises:
            ConfigError: If chain is invalid.

        """
        # Check for required metadata
        for message in self._messages:
            if not get_message_metadata(message, "timestamp"):
                msg = "Missing timestamp metadata"
                raise ConfigError(msg)
            if not get_message_metadata(message, "priority"):
                msg = "Missing priority metadata"
                raise ConfigError(msg)

        # Check for valid role sequences
        for i in range(len(self._messages) - 1):
            current = self._messages[i]
            next_msg = self._messages[i + 1]

            # Check for consecutive human messages
            if isinstance(current, HumanMessage) and isinstance(next_msg, HumanMessage):
                msg = "Invalid message sequence: consecutive human messages"
                raise ConfigError(msg)

            # Check for consecutive AI messages
            if isinstance(current, AIMessage) and isinstance(next_msg, AIMessage):
                msg = "Invalid message sequence: consecutive AI messages"
                raise ConfigError(msg)

        return True

    def get_message_content(self, index: int) -> str:
        """Get message content.

        Args:
            index: Message index.

        Returns:
            Message content.

        Raises:
            IndexError: If index is out of range.

        """
        return self._messages[index].content

    def clear(self) -> None:
        """Clear message chain."""
        self._messages.clear()
        self.metadata.clear()
        self.updated_at = datetime.now(UTC)
