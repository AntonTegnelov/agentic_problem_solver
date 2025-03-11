"""Message routing module for directing messages between agents.

This module provides functionality for routing messages between different agents,
handling retries, and managing message flow in the system.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from src.agent.errors import AgentNotFoundError
from src.exceptions import ConfigError, RetryError
from src.messages.utils import set_message_metadata

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types.agent_types import Agent, StepResult
    from src.common_types.message_types import Message
    from src.messages.chain import MessageChain


class MessageRouter:
    """Message router for directing messages between agents."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> None:
        """Initialize router.

        Args:
            max_retries: Maximum number of retries.
            retry_delay: Delay between retries in seconds.

        """
        self.agents: dict[str, Agent] = {}
        self.routes: dict[str, str] = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register agent.

        Args:
            agent_id: Agent ID.
            agent: Agent to register.

        """
        self.agents[agent_id] = agent

    def register_handler(self, message_type: str, handler: Callable[[Message], None]) -> None:
        """Register message handler.

        Args:
            message_type: Type of message to handle.
            handler: Handler function.

        """
        self.handlers[message_type] = handler

    def register_broadcast_handler(self, handler: Callable[[Message], None]) -> None:
        """Register broadcast message handler.

        Args:
            handler: Handler function.

        """
        self.broadcast_handlers.append(handler)

    async def route_message(
        self,
        message: Message,
        target_agent_id: str,
        chain: MessageChain | None = None,
    ) -> StepResult:
        """Route message to agent.

        Args:
            message: Message to route.
            target_agent_id: Target agent ID.
            chain: Optional message chain.

        Returns:
            Step result.

        Raises:
            AgentNotFoundError: If agent not found.
            RetryError: If max retries exceeded.

        """
        # Check if there's a route defined for this agent
        actual_target_id = self.routes.get(target_agent_id, target_agent_id)

        if actual_target_id not in self.agents:
            msg = f"Agent not found: {actual_target_id}"
            raise AgentNotFoundError(msg)

        agent = self.agents[actual_target_id]
        retries = 0
        set_message_metadata(message, "retry_count", 0)

        while retries <= self.max_retries:
            try:
                result = await agent.process(message)
                if chain is not None:
                    chain.add_message(message)
                if hasattr(agent, "processed_messages"):
                    agent.processed_messages.append(message)
                return result
            except Exception as e:
                retries += 1
                set_message_metadata(message, "retry_count", retries)
                if retries > self.max_retries:
                    msg = f"Max retries exceeded ({self.max_retries}) for message: {e}"
                    raise RetryError(msg) from e
                await asyncio.sleep(self.retry_delay)

        msg = "Max retries exceeded"
        raise RetryError(msg)

    async def broadcast_message(self, message: Message) -> list[StepResult]:
        """Broadcast message to all agents.

        Args:
            message: Message to broadcast.

        Returns:
            List of results from each agent.

        """
        tasks = []
        for agent in self.agents.values():
            tasks.append(agent.process(message))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    def get_handler(self, message_type: str) -> Callable[[Message], None] | None:
        """Get handler for message type.

        Args:
            message_type: Type of message.

        Returns:
            Handler function if registered, None otherwise.

        """
        return self.handlers.get(message_type)

    def add_route(self, source: str, target: str) -> None:
        """Add route between agents.

        Args:
            source: Source agent ID.
            target: Target agent ID.

        Raises:
            ConfigError: If agent not found.

        """
        if source not in self.agents or target not in self.agents:
            msg = f"Invalid route: {source} -> {target}"
            raise ConfigError(msg)
        self.routes[source] = target

    async def route_message_stream(
        self,
        message: Message,
        target_agent_id: str,
    ) -> AsyncGenerator[str, None]:
        """Route message to agent with streaming.

        Args:
            message: Message to route.
            target_agent_id: Target agent ID.

        Yields:
            Chunks of processed message.

        Raises:
            AgentNotFoundError: If agent not found.
            RetryError: If max retries exceeded.

        """
        if target_agent_id not in self.agents:
            msg = f"Agent not found: {target_agent_id}"
            raise AgentNotFoundError(msg)

        agent = self.agents[target_agent_id]
        retries = 0

        while retries <= self.max_retries:
            try:
                async for chunk in agent.process_stream(message):
                    yield chunk
                return
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    msg = f"Max retries exceeded: {e}"
                    raise RetryError(msg)
                await asyncio.sleep(self.retry_delay)

        msg = "Max retries exceeded"
        raise RetryError(msg)
