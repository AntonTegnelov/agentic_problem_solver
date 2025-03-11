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


def set_metadata_at_index(messages: list[Message], index: int, key: str, value: object) -> None:
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


def get_metadata_at_index(messages: list[Message], index: int, key: str) -> object:
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


def _raise_agent_not_found(agent_id: str) -> None:
    """Raise AgentNotFoundError with appropriate message.

    Args:
        agent_id: Agent ID that was not found.

    Raises:
        AgentNotFoundError: Always raised with the agent ID in the message.

    """
    msg = f"Agent not found: {agent_id}"
    raise AgentNotFoundError(msg)


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
        AgentNotFoundError: If agent not found.
        RetryError: If max retries exceeded.

    """
    retries = 0
    last_error = ""

    while retries <= max_retries:
        try:
            if agent_id not in agents:
                _raise_agent_not_found(agent_id)

            agent = agents[agent_id]
            result = await agent.process(message)
            if result.success:
                return result
            # If result was not successful, treat as a retry case
            last_error = f"Unsuccessful result: {result.error}"
        except AgentNotFoundError:
            # Re-raise agent not found errors immediately
            raise
        except ValueError:
            # Re-raise validation errors immediately
            raise
        except (OSError, RuntimeError, ConnectionError) as e:
            # Handle specific exception types
            last_error = str(e)
        except (TypeError, AttributeError, KeyError, IndexError) as e:
            # Handle other common exceptions
            last_error = f"Unexpected error: {e!s}"

        retries += 1
        if retries > max_retries:
            break
        await asyncio.sleep(retry_delay)

    msg = f"Max retries exceeded: {last_error}"
    raise RetryError(msg)


async def process_stream_with_retry(
    message: Message,
    agents: dict[str, object],
    agent_id: str,
    max_retries: int = 3,
    retry_delay: float = 0.1,
) -> AsyncGenerator[str, None]:
    """Process message with streaming and retry.

    Args:
        message: Message to process.
        agents: Dictionary of agents.
        agent_id: Agent ID.
        max_retries: Maximum number of retries.
        retry_delay: Delay between retries.

    Yields:
        Processed message chunks.

    Raises:
        RetryError: If max retries exceeded or agent not found.

    """
    retries = 0

    while retries <= max_retries:
        try:
            if agent_id not in agents:
                # Wrap AgentNotFoundError in RetryError for consistent error handling
                msg = f"Agent not found: {agent_id}"
                raise RetryError(msg) from AgentNotFoundError(msg)

            agent = agents[agent_id]
            async for chunk in agent.process_stream(message):
                yield chunk
            break  # Exit the loop after successful processing
        except ValueError:
            # Re-raise these specific exceptions immediately
            raise
        except Exception as e:
            retries += 1
            if retries > max_retries:
                msg = f"Max retries exceeded: {e}"
                raise RetryError(msg) from e
            await asyncio.sleep(retry_delay)

    # If we've exhausted retries without success
    if retries > max_retries:
        msg = "Max retries exceeded"
        raise RetryError(msg)
