"""Test enhanced message system functionality."""

from datetime import UTC, datetime

import pytest

from src.agent.errors import AgentNotFoundError
from src.common_types.message_types import (
    HumanMessage,
    SystemMessage,
)
from src.exceptions import ConfigError, RetryError
from src.messages import (
    MessageHandler,
    MessagePriority,
    MessageRouter,
    create_ai_message,
    create_human_message,
    create_message_chain,
    create_structured_message,
    create_system_message,
    create_tool_message,
    get_message_metadata,
    parse_structured_content,
    set_message_metadata,
)
from tests.unit.test_utils import MockAgent


def test_create_structured_message() -> None:
    """Test creating structured messages."""
    # Test with string content
    msg = create_structured_message("human", "Hello")
    assert isinstance(msg, HumanMessage)
    assert msg.content == "Hello"

    # Test with dict content
    structured_content = {"action": "search", "query": "test"}
    msg = create_structured_message("system", structured_content)
    assert isinstance(msg, SystemMessage)
    assert "action" in msg.content
    assert "query" in msg.content

    # Test with metadata
    metadata = {"source": "test", "timestamp": "2023-01-01"}
    msg = create_structured_message("ai", "Hello", metadata)
    assert get_message_metadata(msg, "source") == "test"
    assert get_message_metadata(msg, "timestamp") == "2023-01-01"


def test_parse_structured_content() -> None:
    """Test parsing structured content from messages."""
    # Test with JSON string content
    structured_content = {"action": "search", "query": "test"}
    msg = create_structured_message("system", structured_content)
    parsed = parse_structured_content(msg)
    assert parsed["action"] == "search"
    assert parsed["query"] == "test"

    # Test with plain string content
    msg = create_human_message("Hello")
    parsed = parse_structured_content(msg, default={"type": "plain"})
    assert parsed["type"] == "plain"

    # Test with empty content
    msg = create_human_message("")
    parsed = parse_structured_content(msg, default={"type": "empty"})
    assert parsed["type"] == "empty"


def test_message_chain_validation() -> None:
    """Test enhanced message chain validation."""
    chain = create_message_chain()

    # Test empty chain
    assert chain.validate_message_chain()

    # Add messages with required metadata
    human_msg = create_human_message("Hello")
    set_message_metadata(human_msg, "timestamp", datetime.now(UTC).isoformat())
    set_message_metadata(human_msg, "priority", MessagePriority.NORMAL.value)

    ai_msg = create_ai_message("Hi there")
    set_message_metadata(ai_msg, "timestamp", datetime.now(UTC).isoformat())
    set_message_metadata(ai_msg, "priority", MessagePriority.NORMAL.value)

    tool_msg = create_tool_message("Tool result", "tool1")
    set_message_metadata(tool_msg, "timestamp", datetime.now(UTC).isoformat())
    set_message_metadata(tool_msg, "priority", MessagePriority.NORMAL.value)

    chain.messages.append(human_msg)
    chain.messages.append(ai_msg)
    chain.messages.append(tool_msg)

    # Test validation with complete metadata
    assert chain.validate_message_chain()

    # Test validation with missing metadata
    chain = create_message_chain()
    bad_msg = create_human_message("Missing metadata")
    chain.messages.append(bad_msg)
    with pytest.raises(ConfigError):
        chain.validate_message_chain()


def test_message_filtering() -> None:
    """Test enhanced message filtering."""
    chain = create_message_chain()

    # Add messages with different metadata
    msg1 = create_human_message("Test 1")
    set_message_metadata(msg1, "category", "test")
    set_message_metadata(msg1, "priority", MessagePriority.LOW.value)
    set_message_metadata(msg1, "timestamp", datetime.now(UTC).isoformat())

    msg2 = create_ai_message("Test 2")
    set_message_metadata(msg2, "category", "production")
    set_message_metadata(msg2, "priority", MessagePriority.HIGH.value)
    set_message_metadata(msg2, "timestamp", datetime.now(UTC).isoformat())

    msg3 = create_tool_message("Test 3", "tool1")
    set_message_metadata(msg3, "category", "test")
    set_message_metadata(msg3, "priority", MessagePriority.CRITICAL.value)
    set_message_metadata(msg3, "timestamp", datetime.now(UTC).isoformat())

    chain.messages.append(msg1)
    chain.messages.append(msg2)
    chain.messages.append(msg3)

    # Test filtering by criteria
    results = chain.filter_messages(criteria={"category": "test"})
    assert len(results) == 2
    assert msg1 in results
    assert msg3 in results

    # Test filtering by type
    results = chain.filter_messages(criteria={"type": type(msg2)})
    assert len(results) == 1
    assert msg2 in results

    # Test filtering with custom function
    results = chain.filter_messages(
        filter_fn=lambda msg: get_message_metadata(msg, "priority") >= MessagePriority.HIGH.value,
    )
    assert len(results) == 2
    assert msg2 in results
    assert msg3 in results

    # Test combined filtering
    results = chain.filter_messages(
        criteria={"category": "test"},
        filter_fn=lambda msg: get_message_metadata(msg, "priority") >= MessagePriority.HIGH.value,
    )
    assert len(results) == 1
    assert msg3 in results


def test_message_history() -> None:
    """Test message history tracking."""
    chain = create_message_chain()

    # Add messages with timestamps
    for i in range(5):
        msg = create_human_message(f"Message {i}")
        set_message_metadata(msg, "timestamp", datetime.now(UTC).isoformat())
        set_message_metadata(msg, "priority", MessagePriority.NORMAL.value)
        chain.messages.append(msg)

    # Test getting history
    history = chain.get_message_history(limit=3)
    assert len(history) == 3
    assert history[0].content == "Message 2"
    assert history[2].content == "Message 4"

    # Test with metadata
    history = chain.get_message_history(limit=2)
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert get_message_metadata(history[0], "timestamp") is not None
    assert get_message_metadata(history[0], "priority") is not None


@pytest.mark.asyncio
async def test_message_router() -> None:
    """Test message router."""
    # Create agents
    agent1 = MockAgent(agent_id="agent1")
    agent2 = MockAgent(agent_id="agent2")

    router = MessageRouter()

    # Register agents
    router.register_agent("agent1", agent1)
    router.register_agent("agent2", agent2)

    # Test routing message
    message = create_human_message("Test routing")
    result = await router.route_message(message, "agent1")
    assert result.success
    assert result.data == "Processed by agent1"
    assert message in agent1.processed_messages

    # Test streaming
    message = create_human_message("Test streaming")
    chunks = [chunk async for chunk in router.route_message_stream(message, "agent2")]
    assert len(chunks) == 1
    assert chunks[0] == "Processed by agent2"
    assert message in agent2.processed_messages

    # Test routing to non-existent agent
    with pytest.raises(AgentNotFoundError, match="Agent not found: non_existent"):
        await router.route_message(message, "non_existent")


@pytest.mark.asyncio
async def test_message_retry() -> None:
    """Test message retry mechanism."""
    # Create agents
    failing_agent = MockAgent(agent_id="failing", should_fail=True)

    router = MessageRouter(max_retries=2)

    # Register agent
    router.register_agent("failing", failing_agent)

    # Test retry logic
    message = create_human_message("Test retry")
    with pytest.raises(RetryError):
        await router.route_message(message, "failing")

    # Verify retry metadata
    assert get_message_metadata(message, "retry_count") == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_message_broadcast() -> None:
    """Test message broadcasting to multiple agents."""
    # Create agents
    agent1 = MockAgent(agent_id="agent1")
    agent2 = MockAgent(agent_id="agent2")
    agent3 = MockAgent(agent_id="agent3", should_fail=True)

    router = MessageRouter()

    # Register agents
    router.register_agent("agent1", agent1)
    router.register_agent("agent2", agent2)
    router.register_agent("agent3", agent3)

    # Test broadcasting
    message = create_human_message("Broadcast test")
    results = await router.broadcast_message(message)

    # Check results - only successful results are returned
    assert len(results) == 2  # agent3 fails, so only 2 successful results
    assert message in agent1.processed_messages
    assert message in agent2.processed_messages


@pytest.mark.asyncio
async def test_message_handler() -> None:
    """Test message handler."""
    # Create agents
    agent1 = MockAgent(agent_id="agent1")
    agent2 = MockAgent(agent_id="agent2")

    handler = MessageHandler()

    # Register agents
    handler.register_agent("agent1", agent1)
    handler.register_agent("agent2", agent2)

    # Test routing
    message = create_human_message("Test routing")
    result = await handler.route_to_agent(message, "agent1")
    assert result.success
    assert result.data == "Processed by agent1"
    assert message in agent1.processed_messages

    # Test retry handling
    message = create_human_message("Test retry handling")
    result = await handler.handle_message_with_retry(message, "agent2", max_retries=1)
    assert result.success
    assert result.data == "Processed by agent2"
    assert message in agent2.processed_messages


def test_message_creation() -> None:
    """Test message creation."""
    # Test system message
    system_msg = create_system_message("System message")
    assert system_msg.content == "System message"
    assert system_msg.type == "system"

    # Test human message
    human_msg = create_human_message("Human message")
    assert human_msg.content == "Human message"
    assert human_msg.type == "human"

    # Test AI message
    ai_msg = create_ai_message("AI message")
    assert ai_msg.content == "AI message"
    assert ai_msg.type == "ai"

    # Test tool message
    tool_msg = create_tool_message("Tool message", "tool-id")
    assert tool_msg.content == "Tool message"
    assert tool_msg.type == "tool"
    assert get_message_metadata(tool_msg, "tool_call_id") == "tool-id"


def test_message_metadata() -> None:
    """Test message metadata handling."""
    message = create_human_message("Test message")

    # Test setting metadata
    set_message_metadata(message, "key1", "value1")
    assert get_message_metadata(message, "key1") == "value1"

    # Test default value
    assert get_message_metadata(message, "nonexistent", "default") == "default"


@pytest.mark.asyncio
async def test_message_processor() -> None:
    """Test message processor."""
    from src.messages.processor import DefaultMessageProcessor

    processor = DefaultMessageProcessor()

    # Test message processing
    message = create_human_message("Test message")
    result = processor.process(message)
    assert result == message

    # Test validation
    assert processor.validate(message) is True

    # Test validation failure
    empty_message = create_human_message("")
    with pytest.raises(ConfigError, match="Message content cannot be empty"):
        processor.validate(empty_message)


@pytest.mark.asyncio
async def test_message_processor_streaming() -> None:
    """Test message processor streaming."""
    from src.messages.processor import DefaultMessageProcessor, process_stream_with_retry

    processor = DefaultMessageProcessor()

    # Test message processing
    message = create_human_message("Test message")
    result = processor.process(message)
    assert result == message

    # Test with agents
    # Since we can't actually run the stream in a test without mocking agents,
    # we'll just verify that the function exists and has the right signature
    assert callable(process_stream_with_retry)

    # Mock the agent dictionary and agent_id
    agents = {}
    agent_id = "test_agent"

    # This should raise RetryError since the agent doesn't exist and retries will be exhausted
    async def _test_process_stream() -> None:
        async for _ in process_stream_with_retry(message, agents, agent_id):
            pass

    with pytest.raises(RetryError):
        await _test_process_stream()
