"""Message processor module."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.common_types import Message
from src.exceptions import AgentNotFoundError, ConfigError, RetryError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.agent.agent_types import Agent
    from src.agent.result import Result
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


class DefaultMessageProcessor:
    """Default implementation of MessageProcessor protocol."""

    def process(self, message: Message) -> Message:
        """Process a message.

        Args:
            message: Message to process.

        Returns:
            Processed message.

        Raises:
            ConfigError: If message processing fails.

        """
        self.validate(message)
        return message

    def validate(self, message: Message) -> bool:
        """Validate a message.

        Args:
            message: Message to validate.

        Returns:
            True if message is valid.

        Raises:
            ConfigError: If message validation fails.

        """
        return validate_message_content(message)


def create_message_from_dict(data: dict[str, Any]) -> Message:
    """Create message from dictionary.

    Args:
        data: Message data.

    Returns:
        Message instance.

    Raises:
        ConfigError: If role is invalid.

    """
    role = data.get("role", "")
    content = data.get("content", "")
    metadata = data.get("metadata", {})

    if not role:
        msg = "Message role is required"
        raise ConfigError(msg)
    if not content:
        msg = "Message content is required"
        raise ConfigError(msg)

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


async def process_message_with_retry(
    message: Message,
    agents: dict[str, Agent],
    agent_id: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Result:
    """Process message with retry.

    Args:
        message: Message to process.
        agents: Dictionary of agents.
        agent_id: Agent ID.
        max_retries: Maximum number of retries.
        retry_delay: Delay between retries.

    Returns:
        Processing result.

    Raises:
        AgentNotFoundError: If agent is not found.
        RetryError: If max retries exceeded.

    """
    retries = 0
    last_error = ""

    while retries <= max_retries:
        try:
            if agent_id not in agents:
                msg = f"Agent not found: {agent_id}"
                raise AgentNotFoundError(msg)

            agent = agents[agent_id]
            result = await agent.process(message)
            if result.success:
                return result
        except Exception as e:
            last_error = str(e)
        retries += 1
        if retries > max_retries:
            break
        await asyncio.sleep(retry_delay)

    msg = f"Max retries exceeded: {last_error}"
    raise RetryError(msg)


async def process_stream_with_retry(
    message: Message,
    agents: dict[str, Agent],
    agent_id: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> AsyncGenerator[str, None]:
    """Process message stream with retry.

    Args:
        message: Message to process.
        agents: Dictionary of agents.
        agent_id: Agent ID.
        max_retries: Maximum number of retries.
        retry_delay: Delay between retries.

    Yields:
        Message chunks.

    Raises:
        AgentNotFoundError: If agent is not found.
        RetryError: If max retries exceeded.

    """
    retries = 0

    while retries <= max_retries:
        try:
            if agent_id not in agents:
                msg = f"Agent not found: {agent_id}"
                raise AgentNotFoundError(msg)

            agent = agents[agent_id]
            async for chunk in agent.process_stream(message):
                yield chunk
            return
        except Exception as e:
            retries += 1
            if retries > max_retries:
                msg = f"Max retries exceeded: {e}"
                raise RetryError(msg) from e
            await asyncio.sleep(retry_delay)

    msg = "Max retries exceeded"
    raise RetryError(msg)
