"""Message handling module."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_types.agent_types import Agent, Message, Result, StepResult
from src.agent.errors import AgentError
from src.config import ConfigError
from src.exceptions import RetryError

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")
U = TypeVar("U")
MessageValue = str | int | float | bool | dict[str, "MessageValue"] | list["MessageValue"] | None
CriteriaValue = Union[str, int, bool, None]
CriteriaDict = dict[str, CriteriaValue]


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MessageChain:
    """Message chain for tracking conversation history."""

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, MessageValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_message(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Add message to chain with metadata.

        Args:
            message: Message to add.
            priority: Message priority.

        """
        set_message_metadata(message, "timestamp", datetime.now(UTC).isoformat())
        set_message_metadata(message, "priority", priority.value)
        self.messages.append(message)
        self.last_updated = datetime.now(UTC)

    def validate_chain(self) -> bool:
        """Validate message chain structure and content.

        Returns:
            True if chain is valid.

        Raises:
            ConfigError: If chain validation fails.

        """
        if not self.messages:
            return True

        # Check message sequence
        for i, current_msg in enumerate(self.messages[1:], 1):
            prev = self.messages[i - 1]

            # Validate message types follow expected pattern
            if isinstance(prev, HumanMessage) and not isinstance(
                current_msg,
                AIMessage | ToolMessage,
            ):
                error_msg = (
                    "Invalid message sequence: Human message must be followed by AI or Tool message"
                )
                raise ConfigError(error_msg)

            if isinstance(prev, AIMessage) and not isinstance(
                current_msg,
                HumanMessage | ToolMessage,
            ):
                error_msg = (
                    "Invalid message sequence: AI message must be followed by Human or Tool message"
                )
                raise ConfigError(error_msg)

            # Validate timestamps are sequential
            prev_time = get_message_metadata(prev, "timestamp")
            curr_time = get_message_metadata(current_msg, "timestamp")
            if prev_time and curr_time and prev_time > curr_time:
                error_msg = "Invalid message sequence: Messages must be in chronological order"
                raise ConfigError(error_msg)

        return True

    def get_messages_by_type(self, msg_type: type[Message]) -> Iterator[Message]:
        """Get messages of specified type.

        Args:
            msg_type: Message type to filter by.

        Yields:
            Messages of specified type.

        """
        for msg in self.messages:
            if isinstance(msg, msg_type):
                yield msg

    def get_messages_by_priority(
        self,
        min_priority: MessagePriority = MessagePriority.LOW,
    ) -> Iterator[Message]:
        """Get messages with minimum priority level.

        Args:
            min_priority: Minimum priority level.

        Yields:
            Messages meeting priority threshold.

        """
        for msg in self.messages:
            priority = get_message_metadata(
                msg,
                "priority",
                MessagePriority.NORMAL.value,
            )
            if priority >= min_priority.value:
                yield msg

    def search_messages(
        self,
        query: str,
        metadata_key: str | None = None,
    ) -> Iterator[Message]:
        """Search messages by content or metadata.

        Args:
            query: Search query string.
            metadata_key: Optional metadata key to search in.

        Yields:
            Matching messages.

        """
        query = query.lower()
        for msg in self.messages:
            if metadata_key:
                value = get_message_metadata(msg, metadata_key)
                if value and str(value).lower().find(query) != -1:
                    yield msg
            elif msg.content.lower().find(query) != -1:  # type: ignore[union-attr]
                yield msg

    def _validate_message_metadata(self, msg: Message, index: int) -> None:
        """Validate required metadata for a message.

        Args:
            msg: Message to validate.
            index: Index of message in chain.

        Raises:
            ConfigError: If message is missing required metadata.

        """
        required_metadata = ["timestamp", "priority"]
        for key in required_metadata:
            if get_message_metadata(msg, key) is None:
                error_msg = f"Message at index {index} missing required metadata: {key}"
                raise ConfigError(error_msg)

    def _validate_message_content(self, msg: Message, index: int) -> None:
        """Validate content structure for a message.

        Args:
            msg: Message to validate.
            index: Index of message in chain.

        Raises:
            ConfigError: If message content is invalid.

        """
        if isinstance(msg, (HumanMessage, AIMessage)):
            if not msg.content or not isinstance(msg.content, str):
                msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
                error_msg = f"{msg_type} message at index {index} has invalid content"
                raise ConfigError(error_msg)
        elif isinstance(msg, ToolMessage):
            if not msg.content or not isinstance(msg.content, str):
                error_msg = f"Tool message at index {index} has invalid content"
                raise ConfigError(error_msg)
            if not get_message_metadata(msg, "tool_call_id"):
                error_msg = f"Tool message at index {index} missing tool_call_id"
                raise ConfigError(error_msg)

    def validate_message_chain(self) -> bool:
        """Validate the message chain using comprehensive checks.

        Returns:
            True if chain is valid.

        Raises:
            ConfigError: If chain validation fails.

        """
        if not self.messages:
            return True

        # Validate each message
        for i, msg in enumerate(self.messages):
            self._validate_message_metadata(msg, i)
            self._validate_message_content(msg, i)

        # Validate the basic chain structure
        return self.validate_chain()

    def filter_messages(
        self,
        criteria: CriteriaDict | None = None,
        **kwargs: CriteriaValue,
    ) -> list[Message]:
        """Filter messages based on criteria.

        Args:
            criteria: Dictionary of criteria to filter by
            **kwargs: Additional keyword arguments for filtering

        Returns:
            List of messages matching criteria

        """
        if not criteria and not kwargs:
            return list(self.messages)

        # Combine dictionary and keyword criteria
        all_criteria = {}
        if criteria:
            all_criteria.update(criteria)
        if kwargs:
            all_criteria.update(kwargs)

        filtered = []
        for msg in self.messages:
            matches = True
            for key, value in all_criteria.items():
                msg_value = get_message_metadata(msg, key)  # type: ignore[type-var]
                if msg_value != value:
                    matches = False
                    break
            if matches:
                filtered.append(msg)

        return filtered

    def get_history(
        self,
        limit: int | None = None,
        *,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """Get message history with optional metadata.

        Args:
            limit: Maximum number of messages to return
            include_metadata: Whether to include metadata in the response

        Returns:
            List of message dictionaries

        """
        messages = self.messages
        if limit is not None:
            messages = messages[-limit:]

        result = []
        for msg in messages:
            # Determine role based on message type
            if isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            elif isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "unknown"

            message_dict = {
                "role": role,
                "content": msg.content,
            }

            if include_metadata:
                metadata = {}
                if hasattr(msg, "additional_kwargs") and "metadata" in msg.additional_kwargs:
                    metadata = msg.additional_kwargs["metadata"]
                message_dict["metadata"] = metadata

            result.append(message_dict)

        return result


def create_system_message(
    content: str,
    metadata: dict[str, object] | None = None,
) -> SystemMessage:
    """Create a SystemMessage with proper initialization.

    Args:
        content: The message content.
        metadata: Optional metadata to attach to the message.

    Returns:
        A SystemMessage instance.

    """
    if metadata is None:
        metadata = {}
    return SystemMessage(content=content, additional_kwargs={"metadata": metadata})


def create_human_message(
    content: str,
    metadata: dict[str, object] | None = None,
) -> HumanMessage:
    """Create a HumanMessage with proper initialization.

    Args:
        content: The message content.
        metadata: Optional metadata to attach to the message.

    Returns:
        A HumanMessage instance.

    """
    if metadata is None:
        metadata = {}
    return HumanMessage(content=content, additional_kwargs={"metadata": metadata})


def create_ai_message(
    content: str,
    metadata: dict[str, object] | None = None,
) -> AIMessage:
    """Create an AIMessage with proper initialization.

    Args:
        content: The message content.
        metadata: Optional metadata to attach to the message.

    Returns:
        An AIMessage instance.

    """
    if metadata is None:
        metadata = {}
    return AIMessage(content=content, additional_kwargs={"metadata": metadata})


def create_tool_message(
    content: str,
    tool_call_id: str,
    metadata: dict[str, object] | None = None,
) -> ToolMessage:
    """Create a ToolMessage with proper initialization.

    Args:
        content: The message content.
        tool_call_id: The ID of the tool call.
        metadata: Optional metadata to attach to the message.

    Returns:
        A ToolMessage instance.

    """
    if metadata is None:
        metadata = {}
    metadata["tool_call_id"] = tool_call_id
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        additional_kwargs={"metadata": metadata},
    )


def create_structured_message(
    role: str,
    content: MessageValue,
    metadata: dict[str, MessageValue] | None = None,
) -> Message:
    """Create a message with structured content.

    Args:
        role: The message role.
        content: The structured message content.
        metadata: Optional metadata to attach to the message.

    Returns:
        A Message instance with structured content.

    """
    if metadata is None:
        metadata = {}

    # Convert structured content to string if needed
    if not isinstance(content, str):
        import json

        content_str = json.dumps(content)
    else:
        content_str = content

    # Create appropriate message type based on role
    if role.lower() == "system":
        return create_system_message(content_str, metadata)  # type: ignore[return-value]
    if role.lower() == "human" or role.lower() == "user":
        return create_human_message(content_str, metadata)  # type: ignore[return-value]
    if role.lower() == "ai" or role.lower() == "assistant":
        return create_ai_message(content_str, metadata)  # type: ignore[return-value]
    if role.lower() == "tool" or role.lower() == "function":
        tool_id = metadata.get("tool_call_id", "default_tool_id")  # type: ignore[dict-item]
        return create_tool_message(content_str, str(tool_id), metadata)  # type: ignore[return-value]
    msg = f"Unsupported message role: {role}"
    raise ValueError(msg)


def get_message_metadata(
    message: Message | HumanMessage | AIMessage | SystemMessage | ToolMessage,
    key: str,
    default: T | None = None,
) -> T | None:
    """Get metadata value from message.

    Args:
        message: The message to get metadata from.
        key: The metadata key.
        default: Default value if key not found.

    Returns:
        The metadata value or default.

    """
    if not hasattr(message, "additional_kwargs"):
        return default

    metadata = message.additional_kwargs.get("metadata", {})
    return metadata.get(key, default)  # type: ignore[dict-item]


def set_message_metadata(
    message: Message | HumanMessage | AIMessage | SystemMessage | ToolMessage,
    key: str,
    value: MessageValue,
) -> None:
    """Set metadata value on message.

    Args:
        message: The message to set metadata on.
        key: The metadata key.
        value: The metadata value.

    """
    if not hasattr(message, "additional_kwargs"):
        if isinstance(message, Message):
            # For basic Message type, add additional_kwargs attribute
            message.additional_kwargs = {"metadata": {}}
        else:
            # Can't modify message without additional_kwargs
            return

    if "metadata" not in message.additional_kwargs:
        message.additional_kwargs["metadata"] = {}

    message.additional_kwargs["metadata"][key] = value


def get_message_at_index(messages: list[Message], index: int) -> Message:
    """Get message at specified index.

    Args:
        messages: List of messages.
        index: Index to get message from.

    Returns:
        Message at index.

    Raises:
        IndexError: If index is out of range.

    """
    if index < 0:
        index = len(messages) + index

    if not 0 <= index < len(messages):
        msg = f"Message index {index} out of range"
        raise IndexError(msg)

    return messages[index]


def get_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    default: T | None = None,
) -> T | None:
    """Get metadata from message at specified index.

    Args:
        messages: List of messages.
        index: Index to get message from.
        key: Metadata key.
        default: Default value if key not found.

    Returns:
        Metadata value or default.

    Raises:
        IndexError: If index is out of range.

    """
    message = get_message_at_index(messages, index)
    return get_message_metadata(message, key, default)


def set_metadata_at_index(
    messages: list[Message],
    index: int,
    key: str,
    value: MessageValue,
) -> None:
    """Set metadata on message at specified index.

    Args:
        messages: List of messages.
        index: Index to get message from.
        key: Metadata key.
        value: Metadata value.

    Raises:
        IndexError: If index is out of range.

    """
    message = get_message_at_index(messages, index)
    set_message_metadata(message, key, value)


def create_message_chain() -> MessageChain:
    """Create a new message chain.

    Returns:
        A new MessageChain instance.

    """
    return MessageChain()


def validate_message_content(
    message: Message,
    required_fields: list[str] | None = None,
) -> bool:
    """Validate message content structure.

    Args:
        message: Message to validate.
        required_fields: Optional list of required fields in content.

    Returns:
        True if content is valid.

    Raises:
        ConfigError: If content validation fails.

    """
    if not message.content:
        msg = "Message content cannot be empty"
        raise ConfigError(msg)

    # If content is a string but should be structured JSON
    if required_fields and isinstance(message.content, str):
        import json

        try:
            content_dict = json.loads(message.content)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON content: {e}"
            raise ConfigError(msg) from e

        for field in required_fields:
            if field not in content_dict:
                msg = f"Missing required field in content: {field}"
                raise ConfigError(msg)

    return True


def parse_structured_content(
    message: Message,
    default: T | None = None,
) -> dict[str, MessageValue] | T:
    """Parse structured content from message.

    Args:
        message: Message with potentially structured content
        default: Default value if parsing fails

    Returns:
        Parsed structured content or default value

    """
    if not message.content:
        return default if default is not None else {}

    if not isinstance(message.content, str):
        return default if default is not None else {}

    import json

    try:
        return json.loads(message.content)
    except json.JSONDecodeError:
        return default if default is not None else {}


@dataclass
class MessageRouter(Generic[T, U]):
    """Routes messages between agents."""

    agents: dict[str, Agent[T, U]] = field(default_factory=dict)
    message_chain: MessageChain = field(default_factory=create_message_chain)
    max_retries: int = 3
    retry_delay: float = 1.0

    def register_agent(self, name: str, agent: Agent[T, U]) -> None:
        """Register an agent with the router.

        Args:
            name: Agent name
            agent: Agent instance

        """
        self.agents[name] = agent

    def route_message(
        self,
        message: Message,
        target_agent: str,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> StepResult[U]:
        """Route a message to a specific agent.

        Args:
            message: Message to route
            target_agent: Name of target agent
            priority: Message priority

        Returns:
            Result of message processing

        Raises:
            ConfigError: If target agent not found

        """
        if target_agent not in self.agents:
            msg = f"Agent not found: {target_agent}"
            raise ConfigError(msg)

        # Add message to chain
        self.message_chain.add_message(message, priority)

        # Process message with retries
        return self._process_with_retry(message, target_agent)

    async def route_message_stream(
        self,
        message: Message,
        target_agent: str,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> AsyncGenerator[str, None]:
        """Route a message to a specific agent and stream the results.

        Args:
            message: Message to route
            target_agent: Name of target agent
            priority: Message priority

        Yields:
            Result chunks

        Raises:
            ConfigError: If target agent not found

        """
        if target_agent not in self.agents:
            msg = f"Agent not found: {target_agent}"
            raise ConfigError(msg)

        # Add message to chain
        self.message_chain.add_message(message, priority)

        # Get target agent
        agent = self.agents[target_agent]

        # Stream results
        try:
            async for chunk in agent.process_stream(message):  # type: ignore[attr-defined]
                yield chunk
        except (AgentError, RetryError) as e:
            error_msg = f"Error streaming from agent {target_agent}: {e!s}"
            set_message_metadata(message, "error", error_msg)
            raise AgentError(error_msg) from e

    def _process_with_retry(
        self,
        message: Message,
        target_agent: str,
    ) -> StepResult[U]:
        """Process a message with retry logic.

        Args:
            message: Message to process
            target_agent: Name of target agent

        Returns:
            Result of message processing

        Raises:
            RetryError: If max retries exceeded
            AgentError: If processing fails

        """
        agent = self.agents[target_agent]
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                result = agent.process(message)  # type: ignore[attr-defined]

                # If result is a StepResult, return it directly
                if isinstance(result, Result):
                    if not result.success:
                        # Record error in message metadata
                        set_message_metadata(message, "error", result.error)

                        # If critical priority, retry regardless of retry count
                        if (
                            get_message_metadata(message, "priority")
                            == MessagePriority.CRITICAL.value
                        ):
                            retries += 1
                            last_error = result.error
                            continue
                    return result  # type: ignore[return-value]

                # Wrap non-StepResult results
                return StepResult(success=True, data=result, error="")  # type: ignore[return-value]

            except (AgentError, ConfigError) as e:
                retries += 1
                last_error = str(e)

                # Record error in message metadata
                set_message_metadata(message, "error", last_error)
                set_message_metadata(message, "retry_count", retries)
                set_message_metadata(
                    message,
                    "last_retry",
                    datetime.now(timezone.utc).isoformat(),
                )

                # Wait before retry
                time.sleep(self.retry_delay)

                # If max retries exceeded, raise RetryError
                if retries > self.max_retries:
                    break

        # Max retries exceeded
        error_msg = f"Max retries exceeded ({self.max_retries}). Last error: {last_error}"
        raise RetryError(error_msg)

    def _process_with_retry_safely(self, message: Message, agent_name: str) -> StepResult[U]:
        """Process message with retry and error handling.

        Args:
            message: Message to process.
            agent_name: Name of agent to process with.

        Returns:
            Result of processing.

        """
        try:
            return self._process_with_retry(message, agent_name)
        except RetryError as e:
            # Create failure result
            return StepResult(success=False, data=None, error=str(e))  # type: ignore[return-value]

    def broadcast(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> dict[str, StepResult[U]]:
        """Broadcast a message to all registered agents.

        Args:
            message: Message to broadcast
            priority: Message priority

        Returns:
            Dictionary of agent names to results

        """
        results = {}

        # Add message to chain
        self.message_chain.add_message(message, priority)

        # Process message with each agent
        for agent_name in self.agents:
            results[agent_name] = self._process_with_retry_safely(message, agent_name)

        return results

    def get_agent_names(self) -> list[str]:
        """Get names of all registered agents.

        Returns:
            List of agent names

        """
        return list(self.agents.keys())

    def get_history(
        self,
        limit: int | None = None,
        *,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """Get message history with optional metadata.

        Args:
            limit: Maximum number of messages to return
            include_metadata: Whether to include metadata in the response

        Returns:
            List of message dictionaries

        """
        return self.message_chain.get_history(limit, include_metadata=include_metadata)


class MessageHandler:
    """Base message handler."""

    def __init__(self) -> None:
        """Initialize message handler."""
        self.message_chain = create_message_chain()
        self.router = MessageRouter()

    def handle_message(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Handle incoming message.

        Args:
            message: Message to handle
            priority: Message priority

        """
        # Add message to chain with tracking metadata
        self.message_chain.add_message(message, priority)

        # Track message in history
        self.track_message_history(message)

    def track_message_history(self, message: Message) -> None:
        """Track message in history with additional metadata.

        Args:
            message: Message to track

        """
        # Add timestamp if not present
        if not get_message_metadata(message, "timestamp"):
            set_message_metadata(message, "timestamp", datetime.now(UTC).isoformat())

        # Add sequence number
        seq_num = len(self.message_chain.messages)
        set_message_metadata(message, "sequence", seq_num)

    def validate_message_chain(self) -> bool:
        """Validate the message chain.

        Returns:
            True if chain is valid

        """
        return self.message_chain.validate_message_chain()

    def filter_messages(self, **criteria: CriteriaValue) -> list[Message]:
        """Filter messages based on criteria.

        Args:
            **criteria: Criteria to filter by.

        Returns:
            List of filtered messages.

        """
        return self.message_chain.filter_messages(criteria=criteria)

    def get_history(
        self,
        limit: int | None = None,
        *,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """Get message history with optional metadata.

        Args:
            limit: Maximum number of messages to return
            include_metadata: Whether to include metadata in the response

        Returns:
            List of message dictionaries

        """
        return self.message_chain.get_history(limit, include_metadata=include_metadata)

    def route_to_agent(
        self,
        message: Message,
        agent_name: str,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> StepResult[Any]:
        """Route message to a specific agent.

        Args:
            message: Message to route
            agent_name: Name of target agent
            priority: Message priority

        Returns:
            Result of message processing

        """
        return self.router.route_message(message, agent_name, priority)

    async def route_to_agent_stream(
        self,
        message: Message,
        agent_name: str,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> AsyncGenerator[str, None]:
        """Route message to a specific agent and stream results.

        Args:
            message: Message to route
            agent_name: Name of target agent
            priority: Message priority

        Yields:
            Result chunks

        """
        async for chunk in self.router.route_message_stream(
            message,
            agent_name,
            priority,
        ):
            yield chunk

    def register_agent(self, name: str, agent: Agent[Any, Any]) -> None:
        """Register an agent with the router.

        Args:
            name: Agent name
            agent: Agent instance

        """
        self.router.register_agent(name, agent)

    def handle_message_with_retry(
        self,
        message: Message,
        agent_name: str,
        max_retries: int = 3,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> StepResult[Any]:
        """Handle message with retry logic.

        Args:
            message: Message to handle
            agent_name: Name of target agent
            max_retries: Maximum number of retries
            priority: Message priority

        Returns:
            Result of message processing

        """
        # Store original max_retries
        original_max_retries = self.router.max_retries

        try:
            # Set max_retries for this operation
            self.router.max_retries = max_retries

            # Route message with retry
            return self.router.route_message(message, agent_name, priority)
        finally:
            # Restore original max_retries
            self.router.max_retries = original_max_retries

    def _process_message_safely(self, message: Message, agent_name: str) -> StepResult[Any]:
        """Process message with an agent with error handling.

        Args:
            message: Message to process.
            agent_name: Name of agent to process with.

        Returns:
            Result of processing.

        """
        try:
            return self._process_with_retry(message, agent_name)
        except RetryError as e:
            # Create failure result
            return StepResult(success=False, data=None, error=str(e))  # type: ignore[return-value]

    def process_message(
        self,
        message: Message,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> dict[str, StepResult[Any]]:
        """Process message with all agents.

        Args:
            message: Message to process.
            priority: Message priority.

        Returns:
            Dictionary of agent names to results.

        """
        results = {}

        # Add message to chain
        self.message_chain.add_message(message, priority)

        # Process message with each agent
        for agent_name in self.agents:
            results[agent_name] = self._process_message_safely(message, agent_name)

        return results


def create_message_from_dict(message_dict: dict[str, Any]) -> BaseMessage:
    """Create message from dictionary.

    Args:
        message_dict: Message dictionary

    Returns:
        Message instance

    Raises:
        ValueError: If message role is invalid

    """
    role = message_dict.get("role", "")
    content = message_dict.get("content", "")
    content_str = str(content) if content else ""
    metadata = message_dict.get("metadata", {})

    # Create appropriate message type based on role
    if role.lower() == "system":
        return cast(BaseMessage, create_system_message(content_str, metadata))
    if role.lower() == "human" or role.lower() == "user":
        return cast(BaseMessage, create_human_message(content_str, metadata))
    if role.lower() == "ai" or role.lower() == "assistant":
        return cast(BaseMessage, create_ai_message(content_str, metadata))
    if role.lower() == "tool" or role.lower() == "function":
        tool_id = cast(str, metadata.get("tool_call_id", "default_tool_id"))
        return cast(BaseMessage, create_tool_message(content_str, tool_id, metadata))

    msg = f"Unsupported message role: {role}"
    raise ValueError(msg)


class MessageProcessor:
    """Message processor."""

    def __init__(self) -> None:
        """Initialize processor."""
        self.agents: dict[str, Agent[Any, Any]] = {}
        self.message_chain = create_message_chain()
        self.max_retries = 3
        self.retry_delay = 1.0

    async def process_stream(self, message: BaseMessage) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process

        Yields:
            Chunks of processed message

        Raises:
            AgentError: If processing fails

        """
        # Get target agent
        target_agent = get_message_metadata(message, "target_agent")
        if not target_agent:
            msg = "No target agent specified"
            raise AgentError(msg)

        # Get agent instance
        agent = self.get_agent(target_agent)

        # Stream results
        try:
            async for chunk in cast(AsyncGenerator[str, None], agent.process_stream(message)):
                yield chunk
        except (AgentError, RetryError) as e:
            error_msg = f"Error streaming from agent {target_agent}: {e!s}"
            set_message_metadata(message, "error", error_msg)
            raise AgentError(error_msg) from e

    def _process_with_retry(
        self,
        message: BaseMessage,
        agent_name: str,
    ) -> StepResult[Any]:
        """Process message with retries.

        Args:
            message: Message to process
            agent_name: Name of agent to process with

        Returns:
            Processing result

        Raises:
            RetryError: If max retries exceeded
            AgentError: If processing fails

        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                # Get agent instance
                agent = self.get_agent(agent_name)

                # Process message
                result = cast(Any, agent.process(message))

                # If result is a StepResult, return it directly
                if isinstance(result, StepResult):
                    # Check for error and retry if needed
                    if not result.success:
                        retries += 1
                        last_error = result.error
                        continue
                    return cast(StepResult[Any], result)

                # Wrap non-StepResult results
                return StepResult(success=True, data=result, error="")

            except (AgentError, ConfigError) as e:
                retries += 1
                last_error = str(e)

                # Add retry metadata
                set_message_metadata(message, "retries", retries)
                set_message_metadata(
                    message,
                    "last_retry",
                    datetime.now(timezone.utc).isoformat(),
                )

                # Wait before retry
                time.sleep(self.retry_delay)

        # Max retries exceeded
        msg = (
            f"Max retries ({self.max_retries}) exceeded for agent {agent_name}. "
            f"Last error: {last_error}"
        )
        raise RetryError(msg)
