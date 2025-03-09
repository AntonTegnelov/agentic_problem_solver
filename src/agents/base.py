"""Base agent implementation."""

from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.messages import ToolMessage

from src.utils.logging import get_logger

logger = get_logger(__name__)

Message = HumanMessage | AIMessage | SystemMessage | ToolMessage
MessageMetadata = dict[str, Any]


class AgentStep(str, Enum):
    """Agent processing steps."""

    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    EXECUTE = "IMPLEMENT"
    VERIFY = "VERIFY"
    END = "END"


class AgentState:
    """Base agent state."""

    def __init__(self) -> None:
        """Initialize agent state."""
        self.messages: list[Message] = []
        self.error: str | None = None
        self.result: str | None = None

    def get_message_metadata(
        self,
        index: int,
        key: str,
        default: MessageMetadata | None = None,
    ) -> MessageMetadata | None:
        """Get metadata from a message at the specified index.

        Args:
            index: Message index.
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Message metadata value.
        """
        try:
            message = self.messages[index]
            return message.additional_kwargs.get(key, default)
        except IndexError:
            return default

    def set_message_metadata(
        self,
        index: int,
        key: str,
        value: MessageMetadata,
    ) -> None:
        """Set metadata for a message at the specified index.

        Args:
            index: Message index.
            key: Metadata key.
            value: Metadata value.
        """
        try:
            message = self.messages[index]
            message.additional_kwargs[key] = value
        except IndexError:
            pass

    def add_system_message(
        self,
        content: str,
        metadata: MessageMetadata | None = None,
    ) -> None:
        """Add a system message.

        Args:
            content: Message content.
            metadata: Optional message metadata.
        """
        message = SystemMessage(content=content)
        if metadata:
            self.set_message_metadata(len(self.messages), "metadata", metadata)
        self.messages.append(message)

    def add_human_message(
        self,
        content: str,
        metadata: MessageMetadata | None = None,
    ) -> None:
        """Add a human message.

        Args:
            content: Message content.
            metadata: Optional message metadata.
        """
        message = HumanMessage(content=content)
        if metadata:
            self.set_message_metadata(len(self.messages), "metadata", metadata)
        self.messages.append(message)

    def add_ai_message(
        self,
        content: str,
        metadata: MessageMetadata | None = None,
    ) -> None:
        """Add an AI message.

        Args:
            content: Message content.
            metadata: Optional message metadata.
        """
        message = AIMessage(content=content)
        if metadata:
            self.set_message_metadata(len(self.messages), "metadata", metadata)
        self.messages.append(message)

    def add_tool_message(
        self,
        content: str,
        metadata: MessageMetadata | None = None,
    ) -> None:
        """Add a tool message.

        Args:
            content: Message content.
            metadata: Optional message metadata.
        """
        message = ToolMessage(content=content)
        if metadata:
            self.set_message_metadata(len(self.messages), "metadata", metadata)
        self.messages.append(message)

    def clear(self) -> None:
        """Clear agent state."""
        self.messages.clear()
        self.error = None
        self.result = None


class BaseAgent:
    """Base agent implementation."""

    def __init__(
        self, name: str = "base", config: dict[str, Any] | None = None
    ) -> None:
        """Initialize agent.

        Args:
            name: Agent name.
            config: Optional configuration.
        """
        self.name = name
        self.config = config or {}
        self.state = AgentState()

    async def process_message(self, message: HumanMessage) -> AIMessage:
        """Process a message.

        Args:
            message: The message to process.

        Returns:
            Agent response.

        Raises:
            Exception: If message processing fails.
        """
        try:
            return await self._process_message(message)
        except Exception as e:
            logger.exception("Error processing message")
            self.state.error = str(e)
            raise

    async def process_message_stream(
        self,
        message: HumanMessage,
    ) -> AsyncGenerator[AIMessage, None]:
        """Process a message with streaming.

        Args:
            message: The message to process.

        Yields:
            Agent response chunks.

        Raises:
            Exception: If message processing fails.
        """
        try:
            async for response in self._process_message_stream(message):
                yield response
        except Exception as e:
            logger.exception("Error processing message")
            self.state.error = str(e)
            raise

    async def _process_message(self, message: HumanMessage) -> AIMessage:
        """Process a message (to be implemented by subclasses).

        Args:
            message: The message to process.

        Returns:
            Agent response.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    async def _process_message_stream(
        self,
        message: HumanMessage,
    ) -> AsyncGenerator[AIMessage, None]:
        """Process a message with streaming (to be implemented by subclasses).

        Args:
            message: The message to process.

        Yields:
            Agent response chunks.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    def update_state(self, **kwargs: MessageMetadata) -> None:
        """Update agent state.

        Args:
            **kwargs: State updates.
        """
        for key, value in kwargs.items():
            setattr(self.state, key, value)

    def clear_state(self) -> None:
        """Clear agent state."""
        self.state.clear()

    def update_config(self, **kwargs: MessageMetadata) -> None:
        """Update agent configuration.

        Args:
            **kwargs: Configuration updates.
        """
        self.config.update(kwargs)

    def get_message_metadata(
        self,
        index: int,
        key: str,
        default: MessageMetadata | None = None,
    ) -> MessageMetadata | None:
        """Get metadata from a message at the specified index.

        Args:
            index: Message index.
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Message metadata value.
        """
        return self.state.get_message_metadata(index, key, default)

    def set_message_metadata(
        self,
        index: int,
        key: str,
        value: MessageMetadata,
    ) -> None:
        """Set metadata for a message at the specified index.

        Args:
            index: Message index.
            key: Metadata key.
            value: Metadata value.
        """
        self.state.set_message_metadata(index, key, value)
