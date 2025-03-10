"""Message chain module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from src.common_types import Message

MessageValue = str | int | float | bool | dict[str, "MessageValue"] | list["MessageValue"] | None
CriteriaValue = Union[str, int, bool, None]
CriteriaDict = dict[str, CriteriaValue]


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


def create_message_chain() -> MessageChain:
    """Create a new message chain.

    Returns:
        Empty message chain.

    """
    return MessageChain()


@dataclass
class MessageChain:
    """Message chain for tracking conversation history."""

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, MessageValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

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
        self.messages.append(message)
        self.last_updated = datetime.now(UTC)

    def get_messages(self) -> list[Message]:
        """Get all messages in chain.

        Returns:
            List of messages.

        """
        return self.messages

    def get_messages_by_priority(self, min_priority: MessagePriority) -> list[Message]:
        """Get messages with minimum priority.

        Args:
            min_priority: Minimum priority level.

        Returns:
            List of messages.

        """
        return [msg for msg in self.messages if msg.priority >= min_priority]

    def search_messages(self, query: str, field: str | None = None) -> list[Message]:
        """Search messages by content or metadata field.

        Args:
            query: Search query.
            field: Optional metadata field to search.

        Returns:
            List of matching messages.

        """
        results = []
        for msg in self.messages:
            if (field is None and query.lower() in str(msg.content).lower()) or (
                field is not None and query.lower() in str(msg.metadata.get(field, "")).lower()
            ):
                results.append(msg)
        return results

    def validate_chain(self) -> bool:
        """Validate message chain.

        Returns:
            True if chain is valid.

        Raises:
            ValueError: If chain is invalid.

        """
        if not self.messages:
            return True

        for i in range(len(self.messages) - 1):
            curr_msg = self.messages[i]
            next_msg = self.messages[i + 1]

            # Check message sequence
            if curr_msg.role == next_msg.role == "human":
                msg = "Invalid message sequence: consecutive human messages"
                raise ValueError(msg)

        return True
