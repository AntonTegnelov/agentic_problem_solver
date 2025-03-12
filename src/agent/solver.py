"""Solver agent module."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeVar, overload

from src.agent.state.base import AgentState, StateManager
from src.common_types.message_types import Message, SystemMessage
from src.common_types.result_types import Result
from src.messages.creation import create_message
from src.prompts import get_step_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.config.agent import AgentConfig
    from src.llm_providers.interface import LLMProvider
    from src.messages import Message

T = TypeVar("T")


# Remove inheritance from Agent ABC
class SolverAgent:
    """Agent that solves programming problems.

    This agent implements the Agent Protocol from src.agent.agent_types.agent_types.

    .. deprecated:: 0.1.0
       SolverAgent is deprecated and will be removed in a future version.
       Use the hierarchical agent system (ArchitectAgent, PlannerAgent, ExecutorAgent) instead.
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
        warnings.warn(
            "SolverAgent is deprecated and will be removed in a future version. "
            "Use the hierarchical agent system (ArchitectAgent, PlannerAgent, ExecutorAgent) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._provider = provider
        self._state_manager = state_manager
        self._agent_id = "solver_agent"
        self.config = config

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
        return ["solve", "code", "explain", "plan"]

    def can_handle(self, _task: str) -> bool:
        """Check if agent can handle task.

        Args:
            _task: Task to check.

        Returns:
            True if agent can handle task.

        """
        # This agent can handle any task
        return True

    @overload
    def process(self, message: str) -> str: ...

    @overload
    def process(self, message: Message) -> Result[str]: ...

    def process(self, message: str | Message) -> str | Result[str]:
        """Process a message.

        Args:
            message: Message to process (either a string or a Message object).

        Returns:
            If input is a string: The processed response as a string (for backward compatibility).
            If input is a Message: Result containing the processed message.

        Raises:
            ValueError: If provider is not initialized.

        """
        self._validate_provider()

        # Extract content from message
        if isinstance(message, str):
            input_data = message
            return_result = False  # Return string for backward compatibility
        else:
            input_data = message.content
            return_result = True  # Return Result object for Protocol compatibility

        messages = self._prepare_state(input_data)

        response = self._provider.generate(messages)
        self.state.add_message(create_message(role="ai", content=response))

        if return_result:
            return Result(success=True, data=response, error=None)
        return response  # Return string for backward compatibility

    async def process_message(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        result = self.process(message)
        # Ensure we return a Result object
        if isinstance(result, str):
            return Result(success=True, data=result, error=None)
        return result

    async def process_stream(self, message: str | Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process (either a string or a Message object).

        Yields:
            Chunks of processed message.

        """
        self._validate_provider()

        # Extract content from message
        input_data = message if isinstance(message, str) else message.content

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
        result = self.process(message)
        # Ensure we return a Result object
        if isinstance(result, str):
            return Result(success=True, data=result, error=None)
        return result

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        result = self.process(message)
        # Ensure we return a Result object
        if isinstance(result, str):
            return Result(success=True, data=result, error=None)
        return result

    def _prepare_messages(self, messages: list[Message]) -> list[Message]:
        """Prepare messages for processing.

        Args:
            messages: Messages to prepare.

        Returns:
            Prepared messages.

        """
        prepared_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prepared_messages.append(
                    create_message(role="human", content=msg.content),
                )
            else:
                prepared_messages.append(msg)
        return prepared_messages

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
