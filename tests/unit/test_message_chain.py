"""Tests for the MessageChain class in src/messages/chain.py."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.common_types.enums import MessagePriority
from src.exceptions import ConfigError
from src.messages.chain import MessageChain, create_message_chain
from src.messages.utils import get_message_metadata, set_message_metadata


def test_message_chain_initialization() -> None:
    """Test MessageChain initialization."""
    # Test empty initialization
    chain = MessageChain()
    assert len(chain) == 0
    assert chain.messages == []

    # Test initialization with messages
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
    chain = MessageChain(messages)
    assert len(chain) == 2
    assert chain.messages == messages


def test_message_chain_add_message() -> None:
    """Test adding messages to a chain."""
    chain = MessageChain()
    msg1 = HumanMessage(content="Hello")
    msg2 = AIMessage(content="Hi there")

    # Add messages with different priorities
    chain.add_message(msg1, MessagePriority.LOW)
    chain.add_message(msg2, MessagePriority.HIGH)

    # Check messages were added
    assert len(chain) == 2
    assert chain[0] == msg1
    assert chain[1] == msg2

    # Check metadata was set correctly
    assert get_message_metadata(msg1, "priority") == MessagePriority.LOW.value
    assert get_message_metadata(msg2, "priority") == MessagePriority.HIGH.value
    assert get_message_metadata(msg1, "sequence") == 1
    assert get_message_metadata(msg2, "sequence") == 2


def test_message_chain_get_messages_by_priority() -> None:
    """Test filtering messages by priority."""
    chain = MessageChain()
    msg1 = HumanMessage(content="Low priority")
    msg2 = AIMessage(content="Normal priority")
    msg3 = HumanMessage(content="High priority")

    chain.add_message(msg1, MessagePriority.LOW)
    chain.add_message(msg2, MessagePriority.NORMAL)
    chain.add_message(msg3, MessagePriority.HIGH)

    # Test filtering by priority
    high_priority = chain.get_messages_by_priority(MessagePriority.HIGH)
    assert len(high_priority) == 1
    assert high_priority[0] == msg3

    normal_and_above = chain.get_messages_by_priority(MessagePriority.NORMAL)
    assert len(normal_and_above) == 2
    assert msg2 in normal_and_above
    assert msg3 in normal_and_above


def test_message_chain_filter_messages() -> None:
    """Test filtering messages by criteria."""
    chain = MessageChain()
    msg1 = HumanMessage(content="Question about Python")
    msg2 = AIMessage(content="Answer about Python")
    msg3 = HumanMessage(content="Command to run")

    chain.add_message(msg1)
    chain.add_message(msg2)
    chain.add_message(msg3)

    # Add metadata for filtering
    set_message_metadata(msg1, "category", "question")
    set_message_metadata(msg2, "category", "answer")
    set_message_metadata(msg3, "category", "command")

    set_message_metadata(msg1, "topic", "python")
    set_message_metadata(msg2, "topic", "python")
    set_message_metadata(msg3, "topic", "system")

    # Test filtering by single criterion using kwargs
    questions = chain.filter_messages(category="question")
    assert len(questions) == 1
    assert questions[0] == msg1

    # Test filtering by single criterion using dictionary
    commands = chain.filter_messages(criteria={"category": "command"})
    assert len(commands) == 1
    assert commands[0] == msg3

    # Test filtering by multiple criteria
    python_questions = chain.filter_messages(category="question", topic="python")
    assert len(python_questions) == 1
    assert python_questions[0] == msg1

    # Test filtering with no matches
    no_matches = chain.filter_messages(category="unknown")
    assert len(no_matches) == 0


def test_message_chain_search_messages() -> None:
    """Test searching messages by content or metadata."""
    chain = MessageChain()
    msg1 = HumanMessage(content="How do I use Python?")
    msg2 = AIMessage(content="Python is a programming language")

    chain.add_message(msg1)
    chain.add_message(msg2)

    set_message_metadata(msg1, "language", "english")
    set_message_metadata(msg2, "language", "english")

    # Test searching by content
    python_msgs = chain.search_messages("python")
    assert len(python_msgs) == 2
    assert msg1 in python_msgs
    assert msg2 in python_msgs

    # Test searching by metadata field
    english_msgs = chain.search_messages("english", field="language")
    assert len(english_msgs) == 2
    assert msg1 in english_msgs
    assert msg2 in english_msgs


def test_message_chain_validation() -> None:
    """Test message chain validation."""
    chain = MessageChain()

    # Valid sequence: human -> ai -> human
    chain.add_message(HumanMessage(content="Hello"))
    chain.add_message(AIMessage(content="Hi there"))
    chain.add_message(HumanMessage(content="How are you?"))

    # This should pass
    assert chain.validate_chain() is True

    # Invalid sequence: human -> human
    chain2 = MessageChain()
    chain2.add_message(HumanMessage(content="Hello"))
    chain2.add_message(HumanMessage(content="How are you?"))

    # This should raise an error
    with pytest.raises(ConfigError):
        chain2.validate_chain()


def test_create_message_chain_factory() -> None:
    """Test the create_message_chain factory function."""
    chain = create_message_chain()
    assert isinstance(chain, MessageChain)
    assert len(chain) == 0
