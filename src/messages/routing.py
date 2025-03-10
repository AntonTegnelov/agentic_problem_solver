"""Message routing and priority handling."""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agent.agent_types.agent_types import Agent, Message, StepResult
from src.exceptions import RetryError
from src.messages import (
    MessageChain,
    MessagePriority,
    get_message_metadata,
    set_message_metadata,
)


@dataclass
class MessageRouter:
    """Message routing and priority handling."""

    agents: dict[str, Agent] = field(default_factory=dict)
    routes: dict[str, list[str]] = field(default_factory=dict)
    retry_count: int = 3
    retry_delay: float = 1.0

    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register an agent for message routing.

        Args:
            agent_id: Unique agent identifier.
            agent: Agent instance.

        """
        self.agents[agent_id] = agent

    def add_route(self, from_agent: str, to_agent: str) -> None:
        """Add message route between agents.

        Args:
            from_agent: Source agent ID.
            to_agent: Target agent ID.

        """
        if from_agent not in self.routes:
            self.routes[from_agent] = []
        self.routes[from_agent].append(to_agent)

    async def route_message(
        self,
        message: Message,
        from_agent: str,
        chain: MessageChain | None = None,
    ) -> list[StepResult]:
        """Route message to target agents.

        Args:
            message: Message to route.
            from_agent: Source agent ID.
            chain: Optional message chain for tracking.

        Returns:
            List of step results from target agents.

        Raises:
            RetryError: If message routing fails after retries.

        """
        if from_agent not in self.routes:
            return []

        results = []
        for to_agent in self.routes[from_agent]:
            if to_agent not in self.agents:
                continue

            # Get message priority
            priority = get_message_metadata(
                message,
                "priority",
                MessagePriority.NORMAL.value,
            )

            # Add to chain if provided
            if chain:
                chain.add_message(message, MessagePriority(priority))

            # Process message with retries
            result = await self._process_with_retry(
                self.agents[to_agent].process,
                message,
            )
            results.append(result)

        return results

    async def _process_with_retry(
        self,
        process_fn: Callable[[Message], Coroutine[Any, Any, StepResult]],
        message: Message,
    ) -> StepResult:
        """Process message with retry on failure.

        Args:
            process_fn: Message processing function.
            message: Message to process.

        Returns:
            Step result from processing.

        Raises:
            RetryError: If processing fails after retries.

        """
        last_error = None
        for _ in range(self.retry_count):
            try:
                return await process_fn(message)
            except Exception as e:
                last_error = e
                # Add retry metadata
                retries = get_message_metadata(message, "retries", 0)
                set_message_metadata(message, "retries", retries + 1)
                set_message_metadata(
                    message,
                    "last_retry",
                    datetime.now().isoformat(),
                )
                # Wait before retry
                await asyncio.sleep(self.retry_delay)

        msg = (
            f"Message processing failed after {self.retry_count} retries: {last_error}"
        )
        raise RetryError(msg)
