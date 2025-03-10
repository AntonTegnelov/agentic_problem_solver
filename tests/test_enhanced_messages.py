"""Test enhanced message system functionality."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.agent_types.agent_types import Agent, Message, Result
from src.agent.errors import AgentError
from src.exceptions import ConfigError, RetryError
from src.messages import (
    MessageHandler,
    MessagePriority,
    MessageProcessor,
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


class TestAgent(Agent):
    """Test agent implementation."""

    def __init__(self, agent_id: str, should_fail: bool = False) -> None:
        """Initialize agent.

        Args:
            agent_id: Agent ID
            should_fail: Whether agent should fail processing

        """
        self.agent_id = agent_id
        self.should_fail = should_fail
        self.processed_messages: list[Message] = []

    def process(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process

        Returns:
            Processing result

        Raises:
            AgentError: If processing fails

        """
        if self.should_fail:
            msg = "Processing failed"
            raise AgentError(msg)
        self.processed_messages.append(message)
        return Result(success=True, data=f"Processed by {self.agent_id}", error="")

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process

        Yields:
            Chunks of processed message

        Raises:
            AgentError: If processing fails

        """
        if self.should_fail:
            msg = "Processing failed"
            raise AgentError(msg)
        self.processed_messages.append(message)
        yield f"Processed by {self.agent_id}"

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        return self.agent_id

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        return ["test", "mock"]

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        return "test" in task.lower()

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        return self.process(message)


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
    assert history[0]["content"] == "Message 2"
    assert history[2]["content"] == "Message 4"

    # Test with metadata
    history = chain.get_message_history(limit=2, include_metadata=True)
    assert len(history) == 2
    assert "metadata" in history[0]
    assert "timestamp" in history[0]["metadata"]
    assert "priority" in history[0]["metadata"]


@pytest.mark.asyncio
async def test_message_router() -> None:
    """Test message router functionality."""
    router = MessageRouter()
    agent1 = TestAgent(agent_id="agent1")
    agent2 = TestAgent(agent_id="agent2")

    # Register agents
    router.register_agent("agent1", agent1)
    router.register_agent("agent2", agent2)

    # Test routing message
    message = create_human_message("Test routing")
    result = router.route_message(message, "agent1")
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
    with pytest.raises(ConfigError):
        router.route_message(message, "non_existent")


def test_message_retry() -> None:
    """Test message retry functionality."""
    router = MessageRouter(max_retries=2)
    failing_agent = TestAgent(agent_id="failing", should_fail=True)

    # Register agent
    router.register_agent("failing", failing_agent)

    # Test retry logic
    message = create_human_message("Test retry")
    with pytest.raises(RetryError):
        router.route_message(message, "failing")

    # Verify retry metadata
    assert get_message_metadata(message, "retry_count") == 3  # Initial + 2 retries


def test_message_broadcast() -> None:
    """Test message broadcasting."""
    router = MessageRouter()
    agent1 = TestAgent(agent_id="agent1")
    agent2 = TestAgent(agent_id="agent2")
    agent3 = TestAgent(agent_id="agent3", should_fail=True)

    # Register agents
    router.register_agent("agent1", agent1)
    router.register_agent("agent2", agent2)
    router.register_agent("agent3", agent3)

    # Test broadcasting
    message = create_human_message("Broadcast test")
    results = router.broadcast_message(message)

    # Check results
    assert len(results) == 3
    assert results["agent1"].success
    assert results["agent2"].success
    assert not results["agent3"].success
    assert message in agent1.processed_messages
    assert message in agent2.processed_messages


def test_message_handler() -> None:
    """Test message handler functionality."""
    handler = MessageHandler()
    agent1 = TestAgent(agent_id="agent1")
    agent2 = TestAgent(agent_id="agent2")

    # Register agents
    handler.register_agent("agent1", agent1)
    handler.register_agent("agent2", agent2)

    # Test handling message
    message = create_human_message("Test handling")
    handler.handle_message(message)

    # Verify message tracking
    assert len(handler.message_chain.messages) == 1
    assert get_message_metadata(message, "sequence") == 1
    assert get_message_metadata(message, "timestamp") is not None

    # Test routing
    message = create_human_message("Test routing")
    result = handler.route_to_agent(message, "agent1")
    assert result.success
    assert message in agent1.processed_messages

    # Test retry handling
    message = create_human_message("Test retry handling")
    result = handler.handle_message_with_retry(message, "agent2", max_retries=1)
    assert result.success
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


def test_message_processor() -> None:
    """Test message processor."""
    processor = MessageProcessor()

    # Test agent registration
    agent1 = TestAgent("agent1")
    processor.register_agent("agent1", agent1)
    assert "agent1" in processor.list_agents()

    # Test message processing
    message = create_human_message("Test message")
    set_message_metadata(message, "target_agent", "agent1")
    result = processor.process(message)
    assert result.success
    assert result.data == "Processed by agent1"
    assert message in agent1.processed_messages

    # Test processing failure
    failing_agent = TestAgent("failing", should_fail=True)
    processor.register_agent("failing", failing_agent)
    message = create_human_message("Test message")
    set_message_metadata(message, "target_agent", "failing")
    with pytest.raises(RetryError, match="Max retries.*exceeded"):
        processor.process(message)


@pytest.mark.asyncio
async def test_message_processor_streaming() -> None:
    """Test message processor streaming."""
    processor = MessageProcessor()

    # Test successful streaming
    agent1 = TestAgent("agent1")
    processor.register_agent("agent1", agent1)
    message = create_human_message("Test message")
    set_message_metadata(message, "target_agent", "agent1")

    chunks = [chunk async for chunk in processor.process_stream(message)]
    assert chunks == ["Processed by agent1"]
    assert message in agent1.processed_messages

    # Test streaming failure
    failing_agent = TestAgent("failing", should_fail=True)
    processor.register_agent("failing", failing_agent)
    message = create_human_message("Test message")
    set_message_metadata(message, "target_agent", "failing")

    async def process_stream() -> None:
        async for _ in processor.process_stream(message):
            pass

    with pytest.raises(AgentError, match="Error streaming from agent failing:.*"):
        await process_stream()
