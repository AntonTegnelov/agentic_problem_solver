"""Message handler module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from src.common_types import Message

T = TypeVar("T")


class MessageHandler(Protocol):
    """Message handler protocol."""

    def handle(self, message: Message) -> Message:
        """Handle a message.

        Args:
            message: Message to handle.

        Returns:
            Handled message.

        Raises:
            ConfigError: If message handling fails.

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
