"""Message routing and priority handling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from src.agent.errors import AgentError
from src.common_types.enums import MessagePriority
from src.exceptions import RetryError
from src.llm_providers.config.errors import ConfigError
from src.messages.utils import get_message_metadata, set_message_metadata

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from src.agent.agent_types.agent_types import Agent, Message, StepResult
    from src.messages.chain import MessageChain


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

    async def _try_process_once(
        self,
        process_fn: Callable[[Message], Coroutine[Any, Any, StepResult]],
        message: Message,
        attempt: int,
    ) -> StepResult:
        """Try to process message once.

        Args:
            process_fn: Message processing function.
            message: Message to process.
            attempt: Current attempt number.

        Returns:
            Step result from processing.

        Raises:
            ConfigError: If processing fails.
            ValueError: If processing fails.
            AgentError: If processing fails.

        """
        try:
            return await process_fn(message)
        except (ConfigError, ValueError, AgentError):
            # Add retry metadata
            set_message_metadata(message, "retries", attempt)
            set_message_metadata(
                message,
                "last_retry",
                datetime.now(timezone.utc).isoformat(),
            )
            raise

    async def _try_process_once_with_delay(
        self,
        process_fn: Callable[[Message], Coroutine[Any, Any, StepResult]],
        message: Message,
        attempt: int,
        *,
        is_last_attempt: bool,
    ) -> tuple[bool, StepResult | None, Exception | None]:
        """Try to process message once with delay on failure.

        Args:
            process_fn: Message processing function.
            message: Message to process.
            attempt: Current attempt number.
            is_last_attempt: Whether this is the last attempt.

        Returns:
            Tuple of (success, result, error).

        """
        try:
            result = await self._try_process_once(process_fn, message, attempt)
        except (ConfigError, ValueError, AgentError) as e:
            if not is_last_attempt:
                await asyncio.sleep(self.retry_delay)
            return False, None, e
        else:
            return True, result, None

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
        for attempt in range(self.retry_count):
            is_last_attempt = attempt == self.retry_count - 1
            success, result, error = await self._try_process_once_with_delay(
                process_fn,
                message,
                attempt + 1,
                is_last_attempt=is_last_attempt,
            )
            if success:
                return result
            last_error = error

        msg = f"Message processing failed after {self.retry_count} retries: {last_error}"
        raise RetryError(msg)
