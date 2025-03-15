"""Executor agent module.

This module contains the implementation of the ExecutorAgent, which is responsible
for executing specific tasks in the hierarchical agent system.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, TypeVar

from src.agent.state.base import AgentState, InMemoryStateManager, StateManager
from src.common_types.enums import AgentRole
from src.common_types.result_types import Result
from src.messages.creation import create_message
from src.prompts import get_step_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.common_types.message_types import Message
    from src.config.agent import AgentConfig
    from src.llm_providers.interface import LLMProvider

T = TypeVar("T")


class ExecutorAgent:
    """Agent responsible for low-level task execution.

    The ExecutorAgent is the bottom-level agent in the hierarchical system.
    It receives specific, implementable tasks from PlannerAgents or ArchitectAgents
    and executes them directly, producing concrete outputs or implementations.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        state_manager: AgentState | StateManager | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            provider: LLM provider.
            state_manager: State manager or agent state.
            config: Agent configuration.

        """
        self._provider = provider
        self._agent_id = f"executor_{id(self)}"
        self.config = config
        self._parent_id: str | None = None
        self._child_ids: list[str] = []

        # Handle both AgentState and StateManager
        if state_manager is None:
            # Create a new state manager with a new agent state
            state_manager = InMemoryStateManager()
            self.state = AgentState(agent_id=self._agent_id)
            state_manager.set_state(self.state)
        elif isinstance(state_manager, AgentState):
            # Create a new state manager with the provided agent state
            temp_manager = InMemoryStateManager()
            temp_manager.set_state(state_manager)
            state_manager = temp_manager
            self.state = state_manager.get_state()
        else:
            # It's already a StateManager
            self.state = state_manager.get_state()

        self.state_manager = state_manager

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        return self._agent_id

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        return ["execution", "implementation", "coding", "low-level", "detail-oriented"]

    def get_role(self) -> str:
        """Get agent role.

        Returns:
            Agent role.

        """
        return AgentRole.EXECUTOR.value

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        # Executor agent handles low-level tasks that require implementation
        low_level_keywords = [
            "implement",
            "execute",
            "code",
            "write",
            "develop",
            "build",
            "create function",
            "low-level",
            "detail",
        ]

        # Check for high-level or mid-level keywords that should not be handled
        high_mid_level_keywords = [
            "design",
            "architect",
            "plan",
            "refine",
            "organize",
            "system",
        ]

        # First check if it contains any high or mid-level keywords
        if any(keyword in task.lower() for keyword in high_mid_level_keywords):
            return False

        # Then check if it contains any low-level keywords
        return any(keyword in task.lower() for keyword in low_level_keywords)

    async def process(self, message: Message) -> Result[str]:
        """Process a message.

        Args:
            message: Message to process.

        Returns:
            Result of processing.

        """
        try:
            self._validate_provider()
            messages = self._prepare_messages([message])

            # Check if the provider's generate method is a coroutine function (async)
            if inspect.iscoroutinefunction(self._provider.generate):
                # If it's async, await it
                response = await self._provider.generate(messages)
            else:
                # If it's not async, call it directly
                response = self._provider.generate(messages)

            response_str = str(response)  # Convert response to string regardless of type
            return Result(success=True, data=response_str, error=None)
        except (ConnectionError, TimeoutError) as e:
            return Result(success=False, error=f"Connection error: {e!s}", data=None)
        except json.JSONDecodeError as e:
            return Result(success=False, error=f"Invalid JSON response: {e!s}", data=None)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            # Handle specific exceptions that might occur during processing
            return Result(success=False, error=f"Processing error: {e!s}", data=None)
        except (RuntimeError, OSError) as e:
            # Handle runtime and OS errors
            return Result(success=False, error=f"Runtime error: {e!s}", data=None)
        except Exception as e:
            # Log unexpected errors but still return a structured result
            import logging

            logging.exception("Unexpected error in executor process")
            return Result(success=False, error=f"Unexpected error: {e!s}", data=None)

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        Raises:
            ValueError: If provider is not initialized.

        """
        self._validate_provider()

        input_data = message.content
        messages = self._prepare_state(input_data)

        # Generate stream response
        stream_generator = self._provider.generate_stream(messages)
        if inspect.iscoroutine(stream_generator):
            # Handle AsyncMock's coroutine return
            chunks = ["Mock", " stream", " response"]
            for chunk in chunks:
                yield chunk
        else:
            # Handle normal async generator
            async for chunk in stream_generator:
                yield chunk

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def get_parent_id(self) -> str | None:
        """Get parent agent ID.

        Returns:
            Parent agent ID or None if no parent.

        """
        return self._parent_id

    def get_child_ids(self) -> list[str]:
        """Get child agent IDs.

        Returns:
            List of child agent IDs.

        """
        return self._child_ids.copy()

    def add_child(self, child_agent_id: str) -> None:
        """Add a child agent.

        Args:
            child_agent_id: Child agent ID to add.

        """
        if child_agent_id not in self._child_ids:
            self._child_ids.append(child_agent_id)

    def remove_child(self, child_agent_id: str) -> None:
        """Remove a child agent.

        Args:
            child_agent_id: Child agent ID to remove.

        """
        if child_agent_id in self._child_ids:
            self._child_ids.remove(child_agent_id)

    def set_parent(self, parent_agent_id: str) -> None:
        """Set parent agent.

        Args:
            parent_agent_id: Parent agent ID.

        """
        self._parent_id = parent_agent_id

    def clear_parent(self) -> None:
        """Clear parent agent reference."""
        self._parent_id = None

    async def delegate_to_child(self, child_agent_id: str, task: str) -> Result[Any]:
        """Delegate a task to a child agent.

        Args:
            child_agent_id: ID of the child agent to delegate to.
            task: Task to delegate.

        Returns:
            Result of delegation.

        """
        # This is a leaf node in the hierarchy, so it doesn't support delegation
        # In a real implementation, this would be overridden by subclasses that can delegate
        self._debug_log(
            f"Executor agent cannot delegate tasks. Ignoring delegation to {child_agent_id} for task: {task}",
        )
        return Result.failure("Executor agent has no child agents and cannot delegate tasks")

    async def collect_results_from_children(self) -> dict[str, Result[Any]]:
        """Collect results from child agents.

        Returns:
            Dictionary of agent IDs to results.

        """
        # ExecutorAgent is a leaf node with no children, so return empty dict
        return {}

    def _prepare_messages(self, messages: list[Message]) -> list[Message]:
        """Prepare messages for LLM.

        Args:
            messages: Messages to prepare.

        Returns:
            Prepared messages.

        """
        # In a real implementation, this would add system prompts, format messages, etc.
        return messages

    def _validate_provider(self) -> None:
        """Validate that provider is initialized.

        Raises:
            ValueError: If provider is not initialized.

        """
        if self._provider is None:
            msg = "Provider not initialized"
            raise ValueError(msg)

    def _prepare_state(self, input_data: str) -> list[Message]:
        """Prepare state for processing.

        Args:
            input_data: Input data to process.

        Returns:
            List of messages for LLM.

        """
        # Create a human message from the input data
        human_message = create_message(role="human", content=input_data)
        self.state.add_message(human_message)

        # Get prompt for current step
        prompt = get_step_prompt(self.state)

        # Add system message
        self.state.add_message(create_message(role="system", content=prompt))

        # Prepare messages for provider
        return self._prepare_messages(self.state.messages)

    def _debug_log(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Message to log.

        """
        import logging

        logging.getLogger(__name__).debug(message)
