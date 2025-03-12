"""Architect agent module.

This module contains the implementation of the ArchitectAgent, which is responsible
for high-level task decomposition and system design in the hierarchical agent system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from src.agent.state.base import AgentState, StateManager
from src.common_types.result_types import Result
from src.messages.creation import create_message
from src.prompts import get_step_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.common_types.message_types import Message
    from src.config.agent import AgentConfig
    from src.llm_providers.interface import LLMProvider

T = TypeVar("T")


class ArchitectAgent:
    """Agent responsible for high-level task decomposition and system design.

    The ArchitectAgent is the top-level agent in the hierarchical system.
    It breaks down complex problems into independent components and delegates
    them to PlannerAgents or directly to ExecutorAgents for simpler tasks.
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
        self._state_manager = state_manager
        self._agent_id = f"architect_{id(self)}"
        self.config = config
        self._parent_id: str | None = None
        self._child_ids: list[str] = []

        # Handle both AgentState and StateManager
        if state_manager is None:
            self.state = AgentState(agent_id=self._agent_id)
        elif isinstance(state_manager, AgentState):
            self.state = state_manager
        else:
            # It's a StateManager
            self.state = state_manager.get_state()

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
        return ["architecture", "design", "decomposition", "system", "high-level"]

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        # Architect agent handles high-level tasks that require decomposition
        high_level_keywords = [
            "design",
            "architect",
            "system",
            "decompose",
            "break down",
            "structure",
            "high-level",
            "architecture",
        ]
        return any(keyword in task.lower() for keyword in high_level_keywords)

    def process(self, message: Message) -> Result[str]:
        """Process a message.

        Args:
            message: Message to process.

        Returns:
            Result containing the processed message.

        Raises:
            ValueError: If provider is not initialized.

        """
        self._validate_provider()

        input_data = message.content
        messages = self._prepare_state(input_data)

        response = self._provider.generate(messages)
        self.state.add_message(create_message(role="ai", content=response))

        return Result(success=True, data=response, error=None)

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

        async for chunk in self._provider.generate_stream(messages):
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

    def delegate_to_child(self, child_agent_id: str, task: str) -> Result[Any]:
        """Delegate a task to a specific child agent.

        Args:
            child_agent_id: Child agent ID.
            task: Task to delegate.

        Returns:
            Result of task processing.

        Raises:
            ValueError: If child agent not found or delegation fails.

        """
        if child_agent_id not in self._child_ids:
            msg = f"Child agent not found: {child_agent_id}"
            return Result(success=False, data=None, error=msg)

        # In a real implementation, this would use a registry to get the child agent
        # and delegate the task to it. For now, we'll just return a placeholder result.
        return Result(
            success=True,
            data=f"Task '{task}' delegated to child agent {child_agent_id}",
            error=None,
        )

    def collect_results_from_children(self) -> dict[str, Result[Any]]:
        """Collect results from all child agents.

        Returns:
            Dictionary mapping child agent IDs to their results.

        """
        # In a real implementation, this would collect results from all child agents
        # For now, we'll just return a placeholder result
        return {
            child_id: Result(
                success=True,
                data=f"Result from child agent {child_id}",
                error=None,
            )
            for child_id in self._child_ids
        }

    def _prepare_messages(self, messages: list[Message]) -> list[Message]:
        """Prepare messages for processing.

        Args:
            messages: Messages to prepare.

        Returns:
            Prepared messages.

        """
        # For now, just return the messages as is
        return messages

    def _validate_provider(self) -> None:
        """Validate provider is initialized.

        Raises:
            ValueError: If provider is not initialized.

        """
        if not self._provider:
            msg = "Provider not initialized"
            raise ValueError(msg)

    def _prepare_state(self, input_data: str) -> list[Message]:
        """Prepare agent state for processing.

        Args:
            input_data: Input data to process.

        Returns:
            List of prepared messages.

        """
        # Add user message
        self.state.add_message(create_message(role="human", content=input_data))

        # Get prompt for current step
        prompt = get_step_prompt(self.state)

        # Add system message
        self.state.add_message(create_message(role="system", content=prompt))

        # Prepare messages for provider
        return self._prepare_messages(self.state.messages)
