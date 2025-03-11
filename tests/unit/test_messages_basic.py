"""Tests for basic message functionality."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.common_types.enums import MessagePriority
from src.config import ConfigError
from src.messages.chain import MessageChain, create_message_chain
from src.messages.creation import (
    create_ai_message,
    create_human_message,
    create_system_message,
    create_tool_message,
)
from src.messages.utils import get_message_metadata, set_message_metadata


def test_message_creation() -> None:
    """Test message creation."""
    # Test human message creation
    human_msg = create_human_message("Hello")
    assert isinstance(human_msg, HumanMessage)
    assert human_msg.content == "Hello"

    # Test AI message creation
    ai_msg = create_ai_message("Hi there")
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Hi there"

    # Test system message creation
    system_msg = create_system_message("System instruction")
    assert isinstance(system_msg, SystemMessage)
    assert system_msg.content == "System instruction"

    # Test tool message creation
    tool_msg = create_tool_message("Tool output", "tool_123")
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.content == "Tool output"
    assert tool_msg.tool_call_id == "tool_123"


def test_message_metadata() -> None:
    """Test message metadata functions."""
    # Create message with metadata
    msg = create_human_message("Test", {"key1": "value1"})

    # Test metadata was set
    assert get_message_metadata(msg, "key1") == "value1"

    # Test setting new metadata
    set_message_metadata(msg, "key2", "value2")
    assert get_message_metadata(msg, "key2") == "value2"

    # Test default value for missing key
    assert get_message_metadata(msg, "missing", "default") == "default"
    assert get_message_metadata(msg, "missing") is None


def test_message_chain_creation() -> None:
    """Test message chain creation."""
    # Test empty chain creation
    chain = create_message_chain()
    assert isinstance(chain, MessageChain)
    assert len(chain.messages) == 0

    # Test adding messages
    msg1 = create_human_message("Hello")
    chain.add_message(msg1)
    assert len(chain.messages) == 1
    assert chain.messages[0] == msg1

    # Test metadata was added to message
    assert get_message_metadata(msg1, "timestamp") is not None
    assert get_message_metadata(msg1, "priority") == MessagePriority.NORMAL.value


def test_message_chain_validation() -> None:
    """Test message chain validation."""
    chain = create_message_chain()

    # Empty chain should be valid
    assert chain.validate_chain() is True

    # Add valid sequence
    human_msg = create_human_message("Hello")
    ai_msg = create_ai_message("Hi there")
    chain.add_message(human_msg)
    chain.add_message(ai_msg)

    # Valid sequence should pass validation
    assert chain.validate_chain() is True

    # Test invalid sequence
    chain2 = create_message_chain()
    chain2.add_message(create_human_message("Hello"))
    chain2.add_message(create_human_message("Invalid sequence"))

    # Invalid sequence should raise error
    with pytest.raises(ConfigError):
        chain2.validate_chain()


def test_message_filtering() -> None:
    """Test message filtering."""
    chain = create_message_chain()

    # Add messages with different priorities
    msg1 = create_human_message("Low priority")
    msg2 = create_human_message("High priority")

    chain.add_message(msg1, MessagePriority.LOW)
    chain.add_message(msg2, MessagePriority.HIGH)

    # Test filtering by priority
    high_priority = list(chain.get_messages_by_priority(MessagePriority.HIGH))
    assert len(high_priority) == 1
    assert high_priority[0] == msg2

    # Test filtering by criteria
    set_message_metadata(msg1, "category", "question")
    set_message_metadata(msg2, "category", "command")

    questions = chain.filter_messages(category="question")
    assert len(questions) == 1
    assert questions[0] == msg1


def test_message_search() -> None:
    """Test message search functionality."""
    chain = create_message_chain()

    # Add messages with searchable content
    chain.add_message(create_human_message("How do I install Python?"))
    chain.add_message(create_ai_message("You can download Python from python.org"))
    chain.add_message(create_human_message("What about dependencies?"))

    # Test content search
    python_msgs = list(chain.search_messages("python"))
    assert len(python_msgs) == 2

    # Test metadata search
    msg = create_human_message("Test message")
    set_message_metadata(msg, "tags", "important, urgent")
    chain.add_message(msg)

    important_msgs = list(chain.search_messages("important", "tags"))
    assert len(important_msgs) == 1
    assert important_msgs[0] == msg
