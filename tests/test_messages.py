"""Test message system functionality."""

from datetime import UTC, datetime

import pytest

from src.agent.agent_types.agent_types import Message, StepResult
from src.exceptions import ConfigError, RetryError
from src.messages import (
    MessagePriority,
    create_ai_message,
    create_human_message,
    create_message_chain,
    create_tool_message,
    get_message_metadata,
    set_message_metadata,
    validate_message_content,
)
from src.messages.routing import MessageRouter


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize mock agent.

        Args:
            should_fail: Whether agent should fail processing.

        """
        self.should_fail = should_fail
        self.processed_messages: list[Message] = []

    async def process(self, message: Message) -> StepResult:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Step result.

        Raises:
            Exception: If should_fail is True.

        """
        if self.should_fail:
            msg = "Processing failed"
            raise Exception(msg)
        self.processed_messages.append(message)
        return StepResult(success=True, message="Success")


def test_message_chain_validation() -> None:
    """Test message chain validation."""
    chain = create_message_chain()

    # Test empty chain
    assert chain.validate_chain()

    # Test valid sequence
    human_msg = create_human_message("Hello")
    ai_msg = create_ai_message("Hi there")
    tool_msg = create_tool_message("Tool result", "tool1")

    chain.add_message(human_msg)
    chain.add_message(ai_msg)
    chain.add_message(tool_msg)
    assert chain.validate_chain()

    # Test invalid sequence
    chain = create_message_chain()
    chain.add_message(human_msg)
    chain.add_message(human_msg)  # Invalid: Human after Human
    with pytest.raises(ConfigError):
        chain.validate_chain()


def test_message_priority_filtering() -> None:
    """Test message priority filtering."""
    chain = create_message_chain()

    # Add messages with different priorities
    msg1 = create_human_message("Low priority")
    msg2 = create_human_message("High priority")
    msg3 = create_human_message("Critical")

    chain.add_message(msg1, MessagePriority.LOW)
    chain.add_message(msg2, MessagePriority.HIGH)
    chain.add_message(msg3, MessagePriority.CRITICAL)

    # Test filtering
    high_priority = list(chain.get_messages_by_priority(MessagePriority.HIGH))
    assert len(high_priority) == 2  # HIGH and CRITICAL
    assert msg2 in high_priority
    assert msg3 in high_priority


def test_message_search() -> None:
    """Test message search functionality."""
    chain = create_message_chain()

    # Add messages with content and metadata
    msg1 = create_human_message("Test message")
    set_message_metadata(msg1, "category", "test")

    msg2 = create_ai_message("Another message")
    set_message_metadata(msg2, "category", "production")

    chain.add_message(msg1)
    chain.add_message(msg2)

    # Test content search
    results = list(chain.search_messages("test"))
    assert len(results) == 1
    assert results[0] == msg1

    # Test metadata search
    results = list(chain.search_messages("production", "category"))
    assert len(results) == 1
    assert results[0] == msg2


@pytest.mark.asyncio
async def test_message_routing() -> None:
    """Test message routing functionality."""
    router = MessageRouter()
    agent1 = MockAgent()
    agent2 = MockAgent()

    # Register agents and routes
    router.register_agent("agent1", agent1)
    router.register_agent("agent2", agent2)
    router.add_route("agent1", "agent2")

    # Test message routing
    message = create_human_message("Test routing")
    chain = create_message_chain()

    results = await router.route_message(message, "agent1", chain)
    assert len(results) == 1
    assert results[0].success
    assert message in agent2.processed_messages


@pytest.mark.asyncio
async def test_message_retry() -> None:
    """Test message retry functionality."""
    router = MessageRouter(retry_count=2, retry_delay=0.1)
    failing_agent = MockAgent(should_fail=True)

    router.register_agent("failing", failing_agent)
    router.add_route("source", "failing")

    message = create_human_message("Test retry")
    with pytest.raises(RetryError):
        await router.route_message(message, "source")

    # Verify retry metadata
    retries = get_message_metadata(message, "retries")
    assert retries == 2  # Should have attempted twice


def test_message_content_validation() -> None:
    """Test message content validation."""
    # Test empty content
    empty_msg = create_human_message("")
    with pytest.raises(ConfigError):
        validate_message_content(empty_msg)

    # Test missing required fields
    msg = create_human_message("Test")
    with pytest.raises(ConfigError):
        validate_message_content(msg, required_fields=["timestamp"])

    # Test valid message
    valid_msg = create_human_message("Test")
    set_message_metadata(valid_msg, "timestamp", datetime.now(UTC).isoformat())
    assert validate_message_content(valid_msg, required_fields=["timestamp"])
