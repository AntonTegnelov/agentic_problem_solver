"""Solver agent module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from src.agent.base import Agent
from src.agent.result import Result
from src.common_types.enums import MessageRole
from src.messages import Message
from src.prompts import get_step_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.state.base import AgentState
    from src.config.agent import AgentConfig
    from src.llm_providers.interface import LLMProvider

T = TypeVar("T")


class SolverAgent(Agent):
    """Agent that solves programming problems."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        state_manager: AgentState | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            provider: LLM provider.
            state_manager: State manager.
            config: Agent configuration.

        """
        super().__init__(config)
        self._provider = provider
        self._state_manager = state_manager
        self._agent_id = "solver_agent"

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

    async def process_message(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        response = self.process(message.content)
        return Result(success=True, data=response, error=None)

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        """
        self._validate_provider()
        messages = self._prepare_state(message.content)

        async for chunk in self._provider.generate_stream(messages):
            yield chunk

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        response = self.process(message.content)
        return Result(success=True, data=response, error=None)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        response = self.process(message.content)
        return Result(success=True, data=response, error=None)

    def _prepare_messages(self, messages: list[Message]) -> list[Message]:
        """Prepare messages for provider.

        Some providers (like Gemini) don't support system messages.
        This method converts system messages to user messages.

        Args:
            messages: Messages to prepare.

        Returns:
            Prepared messages.

        """
        prepared_messages = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                prepared_messages.append(
                    Message(role=MessageRole.USER, content=msg.content),
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
        self.state.add_message(Message(role=MessageRole.USER, content=input_data))

        # Get prompt for current step
        prompt = get_step_prompt(self.state)

        # Add system message
        self.state.add_message(Message(role=MessageRole.SYSTEM, content=prompt))

        # Prepare messages for provider
        return self._prepare_messages(self.state.messages)

    def process(self, input_data: str) -> str:
        """Process input data.

        Args:
            input_data: Input data to process.

        Returns:
            Processed output.

        Raises:
            ValueError: If provider is not initialized.

        """
        self._validate_provider()
        messages = self._prepare_state(input_data)

        response = self._provider.generate(messages)
        self.state.add_message(Message(role=MessageRole.ASSISTANT, content=response))

        return response
