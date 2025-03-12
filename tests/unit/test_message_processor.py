"""Tests for the message processor module."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.result import Result
from src.common_types.message_types import Message
from src.exceptions import AgentNotFoundError, ConfigError, RetryError
from src.messages.processor import (
    DefaultMessageProcessor,
    MessageProcessor,
    _raise_agent_not_found_error,
    create_message_from_dict,
    get_metadata_at_index,
    parse_structured_content,
    process_message_with_retry,
    process_stream_with_retry,
    set_metadata_at_index,
    validate_message_content,
)


def test_metadata_functions() -> None:
    """Test metadata functions together to avoid patching issues."""
    # Since we can't directly access the metadata attribute, we'll test the functions
    # by mocking the implementation and verifying the behavior

    # Create a mock message with a mock metadata dictionary
    message = MagicMock()
    message.metadata = {}
    messages = [message]

    # Test set_metadata_at_index
    set_metadata_at_index(messages, 0, "test_key", "test_value")
    assert message.metadata.get("test_key") == "test_value"

    # Test get_metadata_at_index
    value = get_metadata_at_index(messages, 0, "test_key")
    assert value == "test_value"

    # Test with missing key
    value = get_metadata_at_index(messages, 0, "missing_key")
    assert value is None

    # Test with invalid index
    with pytest.raises(IndexError):
        get_metadata_at_index(messages, 1, "test_key")


def test_parse_structured_content() -> None:
    """Test parse_structured_content function."""
    # Test with valid JSON content
    message = HumanMessage(content='{"key": "value", "number": 42}')
    result = parse_structured_content(message)
    assert result == {"key": "value", "number": 42}

    # Test with dictionary content directly
    message = MagicMock()
    message.content = {"key": "value", "number": 42}
    result = parse_structured_content(message)
    assert result == {"key": "value", "number": 42}

    # Test with invalid JSON content
    message = HumanMessage(content="Not a JSON string")
    with pytest.raises(ConfigError):
        parse_structured_content(message)

    # Test with empty content
    message = HumanMessage(content="")
    with pytest.raises(ConfigError):
        parse_structured_content(message)

    # Test with None content to trigger AttributeError
    message = MagicMock()
    message.content = None
    with pytest.raises(ConfigError):
        parse_structured_content(message)


def test_validate_message_content() -> None:
    """Test validate_message_content function."""
    # Test with valid message
    message = HumanMessage(content="Valid content")

    # We need to patch the metadata access
    with patch.object(message, "metadata", {}, create=True):
        assert validate_message_content(message) is True

    # Test with empty content - should raise ConfigError
    message = HumanMessage(content="")
    with pytest.raises(ConfigError):
        validate_message_content(message)

    # Test with required fields
    message = HumanMessage(content='{"field1": "value1", "field2": "value2"}')

    # We need to patch the metadata access with required fields
    with patch.object(message, "metadata", {"field1": "value1", "field2": "value2"}, create=True):
        assert validate_message_content(message, required_fields=["field1", "field2"]) is True

    # Test with missing required fields
    message = HumanMessage(content='{"field1": "value1"}')

    # We need to patch the metadata access with missing required field - combine with statements
    with (
        patch.object(message, "metadata", {"field1": "value1"}, create=True),
        pytest.raises(ConfigError, match="Missing required metadata field"),
    ):
        validate_message_content(message, required_fields=["field1", "field2"])


def test_default_message_processor_process() -> None:
    """Test DefaultMessageProcessor.process method."""
    processor = DefaultMessageProcessor()
    message = HumanMessage(content="Test message")

    # Test with valid message
    with patch("src.messages.processor.validate_message_content", return_value=True):
        processed_message = processor.process(message)
        assert processed_message == message

    # Test with invalid message - should raise ConfigError - combine with statements
    with (
        patch("src.messages.processor.validate_message_content", side_effect=ConfigError("Invalid message")),
        pytest.raises(ConfigError),
    ):
        processor.process(message)


def test_default_message_processor_validate() -> None:
    """Test DefaultMessageProcessor.validate method."""
    processor = DefaultMessageProcessor()
    message = HumanMessage(content="Test message")

    # Test with valid message
    with patch("src.messages.processor.validate_message_content", return_value=True):
        assert processor.validate(message) is True

    # Test with invalid message - should raise ConfigError - combine with statements
    with (
        patch("src.messages.processor.validate_message_content", side_effect=ConfigError("Invalid message")),
        pytest.raises(ConfigError),
    ):
        processor.validate(message)


def test_message_processor_protocol() -> None:
    """Test MessageProcessor protocol."""

    # Create a class that implements the MessageProcessor protocol
    class TestProcessor(MessageProcessor):
        def process(self, message: Message) -> Message:
            return message

        def validate(self, _: Message) -> bool:
            return True

    # Create an instance of the class
    processor = TestProcessor()

    # Test the process method
    message = HumanMessage(content="Test message")
    assert processor.process(message) == message

    # Test the validate method
    assert processor.validate(message) is True


@patch("langchain_core.messages.HumanMessage")
@patch("langchain_core.messages.AIMessage")
@patch("langchain_core.messages.SystemMessage")
@patch("langchain_core.messages.ToolMessage")
def test_create_message_from_dict(
    mock_tool: MagicMock,
    mock_system: MagicMock,
    mock_ai: MagicMock,
    mock_human: MagicMock,
) -> None:
    """Test create_message_from_dict function."""
    # Set up mocks
    mock_human.return_value = HumanMessage(content="Human message")
    mock_ai.return_value = AIMessage(content="AI message")
    mock_system.return_value = SystemMessage(content="System message")
    mock_tool.return_value = ToolMessage(content="Tool message", tool_call_id="test_id")

    # Test with human message
    data = {"role": "human", "content": "Human message", "metadata": {"key": "value"}}
    with patch("src.messages.processor.HumanMessage", return_value=mock_human.return_value):
        message = create_message_from_dict(data)
        assert message == mock_human.return_value

    # Test with AI message
    data = {"role": "ai", "content": "AI message"}
    with patch("src.messages.processor.AIMessage", return_value=mock_ai.return_value):
        message = create_message_from_dict(data)
        assert message == mock_ai.return_value

    # Test with system message
    data = {"role": "system", "content": "System message"}
    with patch("src.messages.processor.SystemMessage", return_value=mock_system.return_value):
        message = create_message_from_dict(data)
        assert message == mock_system.return_value

    # Test with tool message
    data = {"role": "tool", "content": "Tool message", "metadata": {"tool_call_id": "test_id"}}
    with patch("src.messages.processor.ToolMessage", return_value=mock_tool.return_value):
        message = create_message_from_dict(data)
        assert message == mock_tool.return_value

    # Test with missing role
    data = {"content": "Message without role"}
    with pytest.raises(ConfigError, match="Message role is required"):
        create_message_from_dict(data)

    # Test with empty role
    data = {"role": "", "content": "Message with empty role"}
    with pytest.raises(ConfigError, match="Message role is required"):
        create_message_from_dict(data)

    # Test with missing content
    data = {"role": "human"}
    with pytest.raises(ConfigError, match="Message content is required"):
        create_message_from_dict(data)

    # Test with empty content
    data = {"role": "human", "content": ""}
    with pytest.raises(ConfigError, match="Message content is required"):
        create_message_from_dict(data)

    # Test with invalid role
    data = {"role": "invalid", "content": "Message with invalid role"}
    with pytest.raises(ConfigError, match="Invalid message role: invalid"):
        create_message_from_dict(data)


def test_raise_agent_not_found_error() -> None:
    """Test _raise_agent_not_found_error function."""
    with pytest.raises(RetryError) as exc_info:
        _raise_agent_not_found_error("test_agent")

    assert "Agent not found: test_agent" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, AgentNotFoundError)


@pytest.mark.asyncio
async def test_process_message_with_retry_success() -> None:
    """Test process_message_with_retry function with successful processing."""
    # Create mock agent and message
    agent = AsyncMock()
    agent.process.return_value = Result(success=True, data="Success", error=None)
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function
    result = await process_message_with_retry(message, agents, "test_agent")

    # Check result
    assert result.success is True
    assert result.data == "Success"
    assert result.error is None

    # Check agent was called
    agent.process.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_process_message_with_retry_agent_not_found() -> None:
    """Test process_message_with_retry function with agent not found."""
    # Create empty agents dictionary
    agents: dict[str, Any] = {}
    message = HumanMessage(content="Test message")

    # Call function and check exception
    with pytest.raises(RetryError):
        await process_message_with_retry(message, agents, "nonexistent_agent")


@pytest.mark.asyncio
async def test_process_message_with_retry_unsuccessful_result() -> None:
    """Test process_message_with_retry function with unsuccessful result."""
    # Create mock agent that returns unsuccessful result
    agent = AsyncMock()
    agent.process.return_value = Result(success=False, data=None, error="Error")
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception
    with pytest.raises(RetryError):
        await process_message_with_retry(message, agents, "test_agent", max_retries=1)

    # Check agent was called multiple times (initial + retry)
    assert agent.process.call_count == 2


@pytest.mark.asyncio
async def test_process_message_with_retry_exception() -> None:
    """Test process_message_with_retry function with exception."""
    # Create mock agent that raises an exception
    agent = AsyncMock()
    agent.process.side_effect = RuntimeError("Test error")
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception
    with pytest.raises(RetryError):
        await process_message_with_retry(message, agents, "test_agent", max_retries=1)

    # Check agent was called multiple times (initial + retry)
    assert agent.process.call_count == 2


@pytest.mark.asyncio
async def test_process_message_with_retry_value_error() -> None:
    """Test process_message_with_retry function with ValueError."""
    # Create mock agent that raises ValueError
    agent = AsyncMock()
    agent.process.side_effect = ValueError("Test error")
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception
    with pytest.raises(ValueError, match="Test error"):
        await process_message_with_retry(message, agents, "test_agent")

    # Check agent was called once
    agent.process.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_process_message_with_retry_type_error() -> None:
    """Test process_message_with_retry function with TypeError."""
    # Create mock agent that raises TypeError
    agent = AsyncMock()
    agent.process.side_effect = TypeError("Test error")
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception
    with pytest.raises(RetryError):
        await process_message_with_retry(message, agents, "test_agent", max_retries=1)

    # Check agent was called multiple times (initial + retry)
    assert agent.process.call_count == 2


@pytest.mark.asyncio
async def test_process_message_with_retry_connection_error() -> None:
    """Test process_message_with_retry function with ConnectionError."""
    # Create mock agent that raises ConnectionError
    agent = AsyncMock()
    agent.process.side_effect = ConnectionError("Test error")
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception
    with pytest.raises(RetryError):
        await process_message_with_retry(message, agents, "test_agent", max_retries=1)

    # Check agent was called multiple times (initial + retry)
    assert agent.process.call_count == 2


@pytest.mark.asyncio
async def test_process_stream_with_retry_success() -> None:
    """Test process_stream_with_retry function with successful processing."""
    # Create mock agent with streaming
    agent = MagicMock()

    async def mock_stream(_: Message) -> AsyncGenerator[str, None]:
        yield "Chunk 1"
        yield "Chunk 2"
        yield "Chunk 3"

    agent.process_stream = mock_stream
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and collect chunks
    chunks = [chunk async for chunk in process_stream_with_retry(message, agents, "test_agent")]

    # Check chunks
    assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]


@pytest.mark.asyncio
async def test_process_stream_with_retry_agent_not_found() -> None:
    """Test process_stream_with_retry function with agent not found."""
    # Create empty agents dictionary
    agents: dict[str, Any] = {}
    message = HumanMessage(content="Test message")

    # Call function and check exception - use a simple statement
    with pytest.raises(RetryError):
        _ = [chunk async for chunk in process_stream_with_retry(message, agents, "nonexistent_agent")]


@pytest.mark.asyncio
async def test_process_stream_with_retry_exception() -> None:
    """Test process_stream_with_retry function with exception."""
    # Create mock agent that raises an exception
    agent = MagicMock()

    async def mock_stream_with_error() -> AsyncGenerator[str, None]:
        yield "Chunk 1"
        msg = "Test error"
        raise RuntimeError(msg)

    agent.process_stream = mock_stream_with_error
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception - use a simple statement
    with pytest.raises(RetryError):
        _ = [chunk async for chunk in process_stream_with_retry(message, agents, "test_agent", max_retries=1)]


@pytest.mark.asyncio
async def test_process_stream_with_retry_value_error() -> None:
    """Test process_stream_with_retry function with ValueError."""
    # Create mock agent that raises ValueError
    agent = MagicMock()

    async def mock_stream_with_error(_: Message) -> AsyncGenerator[str, None]:
        yield "Chunk 1"
        msg = "Test error"
        raise ValueError(msg)

    agent.process_stream = mock_stream_with_error
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception - use a simple statement with match
    with pytest.raises(ValueError, match="Test error"):
        _ = [chunk async for chunk in process_stream_with_retry(message, agents, "test_agent")]


@pytest.mark.asyncio
async def test_process_stream_with_retry_max_retries_exceeded() -> None:
    """Test process_stream_with_retry function with max retries exceeded."""
    # Create mock agent that raises an exception
    agent = MagicMock()
    agent.process_stream = AsyncMock(side_effect=Exception("Test error"))
    agents = {"test_agent": agent}
    message = HumanMessage(content="Test message")

    # Call function and check exception - use a simple statement
    with pytest.raises(RetryError, match="Max retries exceeded"):
        _ = [
            chunk
            async for chunk in process_stream_with_retry(message, agents, "test_agent", max_retries=1, retry_delay=0.01)
        ]

    # Check agent was called multiple times (initial + retry)
    assert agent.process_stream.call_count == 2
