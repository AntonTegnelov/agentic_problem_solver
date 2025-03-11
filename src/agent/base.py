"""Base agent class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.result import Result
    from src.common_types.message_types import Message


class Agent(ABC):
    """Base agent class."""

    @abstractmethod
    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        raise NotImplementedError

    @abstractmethod
    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        raise NotImplementedError

    @abstractmethod
    async def process(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        raise NotImplementedError

    @abstractmethod
    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        """
        raise NotImplementedError

    @abstractmethod
    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        raise NotImplementedError

    @abstractmethod
    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        raise NotImplementedError
