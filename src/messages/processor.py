"""Message processor module."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypeVar,
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.common_types import Message
from src.exceptions import AgentError, ConfigError, RetryError
from src.messages import get_message_metadata

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.common_types import Message

T = TypeVar("T")


def set_metadata_at_index(messages: list[Message], index: int, key: str, value: Any) -> None:
    """Set metadata value for message at index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.
        value: Metadata value.

    Raises:
        IndexError: If index is out of range.

    """
    messages[index].metadata[key] = value


def get_metadata_at_index(messages: list[Message], index: int, key: str) -> Any:
    """Get metadata value from message at index.

    Args:
        messages: List of messages.
        index: Message index.
        key: Metadata key.

    Returns:
        Metadata value.

    Raises:
        IndexError: If index is out of range.

    """
    return messages[index].metadata.get(key)


def parse_structured_content(message: Message) -> dict[str, Any]:
    """Parse structured message content.

    Args:
        message: Message to parse.

    Returns:
        Parsed content as dictionary.

    Raises:
        ConfigError: If content cannot be parsed.

    """
    try:
        if isinstance(message.content, dict):
            return message.content
        return json.loads(message.content)
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        msg = f"Failed to parse message content: {e}"
        raise ConfigError(msg) from e


def validate_message_content(message: Message, required_fields: list[str] | None = None) -> bool:
    """Validate message content.

    Args:
        message: Message to validate.
        required_fields: Optional list of required metadata fields.

    Returns:
        True if message is valid.

    Raises:
        ConfigError: If message validation fails.

    """
    # Check for empty content
    if not message.content:
        msg = "Message content cannot be empty"
        raise ConfigError(msg)

    # Check required metadata fields
    if required_fields:
        for field in required_fields:
            if field not in message.metadata:
                msg = f"Missing required metadata field: {field}"
                raise ConfigError(msg)

    return True


class MessageProcessor(Protocol):
    """Message processor protocol."""

    def process(self, message: Message) -> Message:
        """Process a message.

        Args:
            message: Message to process.

        Returns:
            Processed message.

        Raises:
            ConfigError: If message processing fails.

        """
        ...

    def validate(self, message: Message) -> bool:
        """Validate a message.

        Args:
            message: Message to validate.

        Returns:
            True if message is valid.

        Raises:
            ConfigError: If message validation fails.

        """
        ...


def create_message_from_dict(data: dict[str, Any]) -> Message:
    """Create message from dictionary.

    Args:
        data: Message data.

    Returns:
        Created message.

    Raises:
        ConfigError: If message creation fails.

    """
    if not isinstance(data, dict):
        msg = f"Invalid message data type: {type(data)}"
        raise ConfigError(msg)

    required_fields = ["role", "content"]
    for field in required_fields:
        if field not in data:
            msg = f"Missing required field: {field}"
            raise ConfigError(msg)

    role = data["role"]
    content = data["content"]
    metadata = data.get("metadata", {})

    if role == "human":
        return HumanMessage(content=content, metadata=metadata)
    if role == "ai":
        return AIMessage(content=content, metadata=metadata)
    if role == "system":
        return SystemMessage(content=content, metadata=metadata)
    if role == "tool":
        return ToolMessage(content=content, metadata=metadata)
    msg = f"Invalid message role: {role}"
    raise ConfigError(msg)


def get_message_metadata(message: Message, key: str, default: Any = None) -> Any:
    """Get metadata value from message.

    Args:
        message: Message to get metadata from.
        key: Metadata key.
        default: Default value if key not found.

    Returns:
        Metadata value.

    """
    return message.metadata.get(key, default)


def set_message_metadata(message: Message, key: str, value: Any) -> None:
    """Set message metadata value.

    Args:
        message: Message to set metadata on.
        key: Metadata key.
        value: Metadata value.

    """
    message.metadata[key] = value


class MessageProcessor:
    """Processes messages through registered agents."""

    def __init__(self) -> None:
        """Initialize processor."""
        self.agents: dict[str, Agent] = {}
        self.validators: list[Callable[[Message], bool]] = []
        self.transformers: list[Callable[[Message], Message]] = []

    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register agent.

        Args:
            agent_id: Agent ID.
            agent: Agent instance.

        """
        self.agents[agent_id] = agent

    def list_agents(self) -> list[str]:
        """List registered agent IDs.

        Returns:
            List of agent IDs.

        """
        return list(self.agents.keys())

    def add_validator(self, validator: Callable[[Message], bool]) -> None:
        """Add message validator.

        Args:
            validator: Validator function.

        """
        self.validators.append(validator)

    def add_transformer(self, transformer: Callable[[Message], Message]) -> None:
        """Add message transformer.

        Args:
            transformer: Transformer function.

        """
        self.transformers.append(transformer)

    async def process(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        Raises:
            ConfigError: If message validation fails.
            RetryError: If max retries exceeded.

        """
        # Validate message
        for validator in self.validators:
            if not validator(message):
                msg = "Message validation failed"
                raise ConfigError(msg)

        # Transform message
        for transformer in self.transformers:
            message = transformer(message)

        # Get target agent
        target_agent_id = get_message_metadata(message, "target_agent")
        if not target_agent_id or target_agent_id not in self.agents:
            msg = f"Invalid target agent: {target_agent_id}"
            raise ConfigError(msg)

        # Process message with retries
        agent = self.agents[target_agent_id]
        retries = 0
        max_retries = 3
        last_error = None

        while retries < max_retries:
            try:
                result = await agent.process(message)
                if result.success:
                    return result
            except Exception as e:
                last_error = str(e)
            retries += 1

        msg = f"Max retries ({max_retries}) exceeded. Last error: {last_error}"
        raise RetryError(msg)

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        Raises:
            ConfigError: If message validation fails.
            AgentError: If processing fails.

        """
        # Validate message
        for validator in self.validators:
            if not validator(message):
                msg = "Message validation failed"
                raise ConfigError(msg)

        # Transform message
        for transformer in self.transformers:
            message = transformer(message)

        # Get target agent
        target_agent_id = get_message_metadata(message, "target_agent")
        if not target_agent_id or target_agent_id not in self.agents:
            msg = f"Invalid target agent: {target_agent_id}"
            raise ConfigError(msg)

        # Process message
        agent = self.agents[target_agent_id]
        try:
            async for chunk in agent.process_stream(message):
                yield chunk
        except AgentError as e:
            msg = f"Error streaming from agent {target_agent_id}: {e}"
            raise AgentError(msg) from e
