"""Test utilities for the agent system."""

from collections.abc import AsyncGenerator
from typing import Any

from src.agent.base import Agent
from src.agent.errors import AgentError
from src.agent.result import Result
from src.common_types.message_types import Message


class TestProcessingError(Exception):
    """Test processing error."""


class TestGenerationError(Exception):
    """Test generation error."""


class TestAgent(Agent):
    """Test agent implementation."""

    def __init__(self, agent_id: str = "test", should_fail: bool = False) -> None:
        """Initialize test agent.

        Args:
            agent_id: Agent ID.
            should_fail: Whether agent should fail processing.

        """
        self._agent_id = agent_id
        self.should_fail = should_fail
        self.processed_messages: list[Message] = []
        self.capabilities: list[str] = []

    def get_agent_id(self) -> str:
        """Get agent ID."""
        return self._agent_id

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities."""
        return self.capabilities

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task."""
        return True

    async def process(self, message: Message) -> Result:
        """Process a message."""
        if self.should_fail:
            msg = f"Error streaming from agent {self._agent_id}: Test processing error"
            raise AgentError(msg)
        self.processed_messages.append(message)
        return Result(success=True, data=f"Processed by {self._agent_id}", error="")

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process a message in streaming mode."""
        if self.should_fail:
            msg = f"Error streaming from agent {self._agent_id}: Test processing error"
            raise AgentError(msg)
        self.processed_messages.append(message)
        yield f"Processed by {self._agent_id}"

    def send_message(self, message: Message) -> Result[Any]:
        """Send a message."""
        return Result(success=True, data=f"Sent by {self._agent_id}", error="")

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive a message."""
        return Result(success=True, data=f"Received by {self._agent_id}", error="")
