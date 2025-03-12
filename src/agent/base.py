"""Base agent class.

MIGRATION PLAN: Transitioning from ABC to Protocol-based Agent Definition
=========================================================================

This module contains the abstract base class implementation of Agent, which is being
deprecated in favor of the Protocol-based approach in src.agent.agent_types.agent_types.

Migration Strategy:
------------------

1. Phase 1: Deprecation and Warning (Current)
   - Add deprecation warnings to all methods in the ABC
   - Document usage tracking in logs
   - Maintain backward compatibility
   - Timeline: Current release

2. Phase 2: Dual Support
   - Update SolverAgent to implement the Protocol instead of inheriting from ABC
   - Create adapter classes if needed for legacy code
   - Update tests to use Protocol-based approach
   - Timeline: Next 1-2 releases

3. Phase 3: Protocol Dominance
   - Make ABC a wrapper around Protocol (reverse the dependency)
   - Update all remaining direct usages of ABC
   - Timeline: 2-3 releases from now

4. Phase 4: ABC Removal
   - Remove the ABC implementation entirely
   - Ensure all code uses Protocol-based approach
   - Timeline: 4+ releases from now

Benefits of Protocol-based Approach:
----------------------------------
- More flexible implementation without requiring inheritance
- Better support for type checking with generics
- Aligns with architecture's emphasis on clean interfaces
- Encourages composition over inheritance
- More consistent with modern Python typing practices

Usage Tracking:
--------------
- Deprecation warnings will be logged when methods are called
- Usage metrics will be collected to track migration progress
- Regular reports will be generated to identify remaining usages

Implementation Guidelines:
------------------------
- When implementing new agents, use the Protocol from src.agent.agent_types.agent_types
- For existing agents, gradually migrate to Protocol implementation
- Use adapter pattern for complex migrations
- Ensure test coverage during migration

See src.agent.agent_types.agent_types.Agent for the Protocol definition that
should be used going forward.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.result import Result
    from src.common_types.message_types import Message

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def deprecated(func: F) -> F:
    """Mark a function as deprecated with a warning.

    Args:
        func: The function to mark as deprecated.

    Returns:
        Wrapped function that issues a deprecation warning.

    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"Call to deprecated method {func.__name__}. Use src.agent.agent_types.agent_types.Agent Protocol instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return cast(F, wrapper)


class Agent(ABC):
    """Base agent class.

    DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.
    This abstract base class will be removed in a future release.
    """

    @deprecated
    @abstractmethod
    def get_agent_id(self) -> str:
        """Get agent ID.

        DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.

        Returns:
            Agent ID.

        """
        raise NotImplementedError

    @deprecated
    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.

        Returns:
            List of capabilities.

        """
        raise NotImplementedError

    @deprecated
    @abstractmethod
    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        raise NotImplementedError

    @deprecated
    @abstractmethod
    async def process(self, message: Message) -> Result:
        """Process message.

        DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        raise NotImplementedError

    @deprecated
    @abstractmethod
    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        """
        raise NotImplementedError

    @deprecated
    @abstractmethod
    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        raise NotImplementedError

    @deprecated
    @abstractmethod
    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        DEPRECATED: Use src.agent.agent_types.agent_types.Agent Protocol instead.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        raise NotImplementedError
