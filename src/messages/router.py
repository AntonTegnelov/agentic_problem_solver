"""Message routing module for directing messages between agents.

This module provides functionality for routing messages between different agents,
handling retries, and managing message flow in the system. The MessageRouter class
serves as a central hub for message distribution, error handling, and retry logic.

Key features:
- Agent registration and management
- Message routing with retry capabilities
- Streaming message support
- Route configuration between agents
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
    """Message router for directing messages between agents.

    This class manages the routing of messages between different agents in the system,
    handling retries, error conditions, and maintaining routing tables.

    Attributes:
        agents: Dictionary mapping agent IDs to agent instances.
        routes: Dictionary mapping source agent IDs to target agent IDs.
        max_retries: Maximum number of retry attempts for message processing.
        retry_delay: Delay between retry attempts in seconds.

    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> None:
        """Initialize router.

        Args:
            max_retries: Maximum number of retries for message processing.
            retry_delay: Delay between retries in seconds.

        """
        self.agents: dict[str, Agent] = {}
        self.routes: dict[str, str] = {}
        self.handlers: dict[str, Callable[[Message], None]] = {}
        self.broadcast_handlers: list[Callable[[Message], None]] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register agent with the router.

        Adds an agent to the router's registry, making it available for message routing.

        Args:
            agent_id: Unique identifier for the agent.
            agent: Agent instance to register.

        """
        self.agents[agent_id] = agent

    def register_handler(self, message_type: str, handler: Callable[[Message], None]) -> None:
        """Register message handler for a specific message type.

        Args:
            message_type: Type of message to handle.
            handler: Handler function to process messages of the specified type.

        """
        self.handlers[message_type] = handler

    def register_broadcast_handler(self, handler: Callable[[Message], None]) -> None:
        """Register broadcast message handler.

        Broadcast handlers receive all messages regardless of their type.

        Args:
            handler: Handler function to process all messages.

        """
        self.broadcast_handlers.append(handler)

    async def route_message(
        self,
        message: Message,
        target_agent_id: str,
        chain: MessageChain | None = None,
    ) -> StepResult:
        """Route message to a specific agent with retry capability.

        This method sends a message to the specified agent, handling retries
        if processing fails. If a message chain is provided, the message will
        be added to the chain after successful processing.

        Args:
            message: Message to route.
            target_agent_id: Target agent ID.
            chain: Optional message chain to record the message.

        Returns:
            Step result from the agent's processing.

        Raises:
            AgentNotFoundError: If the target agent is not found.
            RetryError: If maximum retries are exceeded.

        """
        # Check if there's a route defined for this agent
        actual_target_id = self.routes.get(target_agent_id, target_agent_id)

        if actual_target_id not in self.agents:
            msg = f"Agent not found: {actual_target_id}"
            raise AgentNotFoundError(msg)

        agent = self.agents[actual_target_id]
        retries = 0
        set_message_metadata(message, "retry_count", 0)
        last_result = None

        while retries <= self.max_retries:
            try:
                last_result = await agent.process(message)
                # Process successful, add to chain if needed
                if chain is not None:
                    chain.add_message(message)
                if hasattr(agent, "processed_messages"):
                    agent.processed_messages.append(message)
                # Exit the loop with the result
                break
            except AgentNotFoundError:
                # Re-raise immediately
                raise
            except Exception as e:
                retries += 1
                set_message_metadata(message, "retry_count", retries)
                if retries > self.max_retries:
                    msg = f"Max retries exceeded ({self.max_retries}) for message: {e}"
                    raise RetryError(msg) from e
                await asyncio.sleep(self.retry_delay)

        # If we have a result, return it
        if last_result is not None:
            return last_result

        # This should never be reached due to the exception above, but added for completeness
        msg = "Max retries exceeded"
        raise RetryError(msg)

    async def broadcast_message(self, message: Message) -> list[StepResult]:
        """Broadcast message to all registered agents.

        Sends the same message to all agents in the registry and collects their results.
        Exceptions during processing are filtered out from the results.

        Args:
            message: Message to broadcast.

        Returns:
            List of successful results from each agent.

        """
        tasks = [agent.process(message) for agent in self.agents.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    def get_handler(self, message_type: str) -> Callable[[Message], None] | None:
        """Get handler for a specific message type.

        Args:
            message_type: Type of message.

        Returns:
            Handler function if registered, None otherwise.

        """
        return self.handlers.get(message_type)

    def add_route(self, source: str, target: str) -> None:
        """Add route between agents.

        Configures a routing rule that redirects messages from the source agent
        to the target agent.

        Args:
            source: Source agent ID.
            target: Target agent ID.

        Raises:
            ConfigError: If either the source or target agent is not registered.

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
        """Route message to agent with streaming response.

        Similar to route_message, but returns a stream of response chunks
        instead of waiting for the complete response.

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
                break  # Exit the loop after successful processing
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    msg = f"Max retries exceeded: {e}"
                    raise RetryError(msg) from e
                await asyncio.sleep(self.retry_delay)

        # This should never be reached due to the break above, but added for completeness
        if retries > self.max_retries:
            msg = "Max retries exceeded"
            raise RetryError(msg)
