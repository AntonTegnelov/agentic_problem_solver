"""Agent adapters for migration between ABC and Protocol implementations.

This module provides adapter classes to facilitate the migration from the
abstract base class (ABC) implementation to the Protocol-based approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from src.agent.base import Agent as AgentABC

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Agent as AgentProtocol
    from src.agent.result import Result
    from src.common_types.message_types import Message

T = TypeVar("T")


class ProtocolToABCAdapter(AgentABC):
    """Adapter that wraps a Protocol-based Agent to implement the ABC interface.

    This adapter allows using Protocol-based Agent implementations in places
    where an ABC-based Agent is expected, facilitating gradual migration.
    """

    def __init__(self, protocol_agent: AgentProtocol[Any]) -> None:
        """Initialize adapter with a Protocol-based Agent.

        Args:
            protocol_agent: The Protocol-based Agent to adapt.

        """
        self._agent = protocol_agent

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        return self._agent.get_agent_id()

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        return self._agent.get_capabilities()

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        return self._agent.can_handle(task)

    async def process(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        return self._agent.process(message)

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        """
        async for chunk in self._agent.process_stream(message):
            yield chunk

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        return self._agent.send_message(message)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        return self._agent.receive_message(message)


class ABCToProtocolAdapter:
    """Adapter that wraps an ABC-based Agent to implement the Protocol interface.

    This adapter allows using ABC-based Agent implementations in places
    where a Protocol-based Agent is expected, facilitating gradual migration.
    """

    def __init__(self, abc_agent: AgentABC) -> None:
        """Initialize adapter with an ABC-based Agent.

        Args:
            abc_agent: The ABC-based Agent to adapt.

        """
        self._agent = abc_agent

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        return self._agent.get_agent_id()

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        return self._agent.get_capabilities()

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        return self._agent.can_handle(task)

    def process(self, message: Message) -> Result[T]:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        return self._agent.process(message)

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        """
        async for chunk in self._agent.process_stream(message):
            yield chunk

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        return self._agent.send_message(message)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        return self._agent.receive_message(message)
