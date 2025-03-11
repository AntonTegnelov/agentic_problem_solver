"""Message handler module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from src.common_types.error_types import AgentNotFoundError, RetryError
from src.common_types.message_types import Message
from src.messages.chain import MessageChain
from src.messages.router import MessageRouter
from src.messages.utils import set_message_metadata

if TYPE_CHECKING:
    from src.common_types import Message
    from src.common_types.result_types import Result

T = TypeVar("T")


@dataclass
class MessageHandler:
    """Message handler."""

    handlers: dict[str, Callable[[Message], None]] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    message_chain: MessageChain = field(default_factory=MessageChain)
    _sequence: int = field(default=0)
    router: MessageRouter = field(default_factory=MessageRouter)

    def register_handler(self, message_type: str, handler: Callable[[Message], None]) -> None:
        """Register message handler.

        Args:
            message_type: Message type.
            handler: Message handler.

        """
        self.handlers[message_type] = handler

    def handle_message(self, message: Message) -> None:
        """Handle message.

        Args:
            message: Message to handle.

        """
        handler = self.handlers.get(message.type)
        if handler:
            handler(message)
        self._sequence += 1
        set_message_metadata(message, "sequence", self._sequence)
        set_message_metadata(message, "timestamp", datetime.now(timezone.utc).isoformat())
        self.message_chain.messages.append(message)

    def handle(self, message: Message) -> Message:
        """Handle a message.

        Args:
            message: Message to handle.

        Returns:
            Handled message.

        Raises:
            ConfigError: If message handling fails.

        """

    def validate(self, message: Message) -> bool:
        """Validate a message.

        Args:
            message: Message to validate.

        Returns:
            True if message is valid.

        Raises:
            ConfigError: If message validation fails.

        """

    def register_agent(self, agent_id: str, agent: object) -> None:
        """Register agent.

        Args:
            agent_id: Agent ID.
            agent: Agent instance.

        """
        self.agents[agent_id] = agent
        self.router.register_agent(agent_id, agent)

    async def route_to_agent(self, message: Message, agent_id: str) -> Result[Any]:
        """Route message to agent.

        Args:
            message: Message to route.
            agent_id: Target agent ID.

        Returns:
            Result of message routing.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self.agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        self.agents[agent_id]
        return await self.router.route_message(message, agent_id)

    async def handle_message_with_retry(
        self,
        message: Message,
        agent_id: str,
        max_retries: int = 3,
    ) -> Result[Any]:
        """Handle message with retry.

        Args:
            message: Message to handle.
            agent_id: Agent ID.
            max_retries: Maximum number of retries.

        Returns:
            Step result.

        Raises:
            AgentNotFoundError: If agent not found.
            RetryError: If max retries exceeded.

        """
        # First check if agent exists to avoid try-except in loop
        if agent_id not in self.agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        retries = 0
        last_error = None

        # Define a helper function to attempt message routing
        async def attempt_route() -> Result[Any]:
            try:
                return await self.router.route_message(message, agent_id)
            except (
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
                IndexError,
                OSError,
                RuntimeError,
                ConnectionError,
            ) as e:
                nonlocal last_error
                last_error = e
                return None

        while retries <= max_retries:
            result = await attempt_route()
            if result is not None:
                return result

            retries += 1
            if retries > max_retries:
                msg = f"Max retries exceeded ({max_retries}). Last error: {last_error}"
                raise RetryError(msg) from last_error

            await asyncio.sleep(0.1 * (2**retries))  # Exponential backoff

        # This should never be reached due to the check above, but added for completeness
        if last_error:
            msg = f"Max retries exceeded ({max_retries})"
            raise RetryError(msg) from last_error
        return None
