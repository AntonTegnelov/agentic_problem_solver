"""Solver agent module."""

from collections.abc import AsyncGenerator
from typing import TypeVar

from src.agent.agent_types.agent_types import Message
from src.agent.base import BaseAgent
from src.agent.state.base import AgentState
from src.common_types.enums import MessageRole
from src.config.agent import AgentConfig
from src.llm_providers.interface import LLMProvider
from src.prompts import get_step_prompt

T = TypeVar("T")


class SolverAgent(BaseAgent[str, str]):
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

    async def process_stream(self, input_data: str) -> AsyncGenerator[str, None]:
        """Process input data and stream results.

        Args:
            input_data: Input data to process.

        Yields:
            Processed output chunks.

        Raises:
            ValueError: If provider is not initialized.

        """
        self._validate_provider()
        messages = self._prepare_state(input_data)

        async for chunk in self._provider.generate_stream(messages):
            yield chunk
