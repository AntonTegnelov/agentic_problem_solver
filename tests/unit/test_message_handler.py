"""Tests for the MessageHandler class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.common_types.error_types import AgentNotFoundError, RetryError
from src.messages.handler import MessageHandler


def test_message_handler_initialization() -> None:
    """Test MessageHandler initialization."""
    handler = MessageHandler()
    assert handler.handlers == {}
    assert handler.agents == {}
    assert handler.sequence == 0
    assert handler.message_chain is not None
    assert handler.router is not None


def test_register_handler() -> None:
    """Test register_handler method."""
    handler = MessageHandler()
    mock_handler_func = MagicMock()

    # Register a handler
    handler.register_handler("test_type", mock_handler_func)

    # Check if handler was registered
    assert "test_type" in handler.handlers
    assert handler.handlers["test_type"] == mock_handler_func


def test_handle_message() -> None:
    """Test handle_message method."""
    handler = MessageHandler()
    mock_handler_func = MagicMock()

    # Register a handler
    handler.register_handler("test_type", mock_handler_func)

    # Create a message
    message = MagicMock()
    message.type = "test_type"
    message.metadata = {}

    # Handle the message
    with patch("src.messages.handler.set_message_metadata") as mock_set_metadata:
        handler.handle_message(message)

        # Check if handler was called
        mock_handler_func.assert_called_once_with(message)

        # Check if sequence was incremented
        assert handler.sequence == 1

        # Check if metadata was set
        assert mock_set_metadata.call_count == 2

        # Check if message was added to chain
        assert message in handler.message_chain.messages


def test_handle_message_no_handler() -> None:
    """Test handle_message method with no handler."""
    handler = MessageHandler()

    # Create a message
    message = MagicMock()
    message.type = "unknown_type"
    message.metadata = {}

    # Handle the message
    with patch("src.messages.handler.set_message_metadata") as mock_set_metadata:
        handler.handle_message(message)

        # Check if sequence was incremented
        assert handler.sequence == 1

        # Check if metadata was set
        assert mock_set_metadata.call_count == 2

        # Check if message was added to chain
        assert message in handler.message_chain.messages


def test_register_agent() -> None:
    """Test register_agent method."""
    handler = MessageHandler()
    mock_agent = MagicMock()

    # Mock the router
    handler.router = MagicMock()

    # Register an agent
    handler.register_agent("test_agent", mock_agent)

    # Check if agent was registered
    assert "test_agent" in handler.agents
    assert handler.agents["test_agent"] == mock_agent

    # Check if agent was registered with router
    handler.router.register_agent.assert_called_once_with("test_agent", mock_agent)


@pytest.mark.asyncio
async def test_route_to_agent_success() -> None:
    """Test route_to_agent method with success."""
    handler = MessageHandler()
    mock_agent = MagicMock()
    mock_result = MagicMock()

    # Register an agent
    handler.register_agent("test_agent", mock_agent)

    # Mock router.route_message
    handler.router.route_message = AsyncMock(return_value=mock_result)

    # Create a message
    message = HumanMessage(content="Test message")

    # Route the message
    result = await handler.route_to_agent(message, "test_agent")

    # Check if router.route_message was called
    handler.router.route_message.assert_called_once_with(message, "test_agent")

    # Check if result is correct
    assert result == mock_result


@pytest.mark.asyncio
async def test_route_to_agent_not_found() -> None:
    """Test route_to_agent method with agent not found."""
    handler = MessageHandler()

    # Create a message
    message = HumanMessage(content="Test message")

    # Route the message
    with pytest.raises(AgentNotFoundError, match="Agent not found: unknown_agent"):
        await handler.route_to_agent(message, "unknown_agent")


@pytest.mark.asyncio
async def test_handle_message_with_retry_success() -> None:
    """Test handle_message_with_retry method with success."""
    handler = MessageHandler()
    mock_agent = MagicMock()
    mock_result = MagicMock()

    # Register an agent
    handler.register_agent("test_agent", mock_agent)

    # Mock router.route_message
    handler.router.route_message = AsyncMock(return_value=mock_result)

    # Create a message
    message = HumanMessage(content="Test message")

    # Handle the message with retry
    result = await handler.handle_message_with_retry(message, "test_agent")

    # Check if router.route_message was called
    handler.router.route_message.assert_called_once_with(message, "test_agent")

    # Check if result is correct
    assert result == mock_result


@pytest.mark.asyncio
async def test_handle_message_with_retry_agent_not_found() -> None:
    """Test handle_message_with_retry method with agent not found."""
    handler = MessageHandler()

    # Create a message
    message = HumanMessage(content="Test message")

    # Handle the message with retry
    with pytest.raises(AgentNotFoundError, match="Agent not found: unknown_agent"):
        await handler.handle_message_with_retry(message, "unknown_agent")


@pytest.mark.asyncio
async def test_handle_message_with_retry_exception() -> None:
    """Test handle_message_with_retry method with exception."""
    handler = MessageHandler()
    mock_agent = MagicMock()

    # Register an agent
    handler.register_agent("test_agent", mock_agent)

    # Mock router.route_message to raise an exception
    handler.router.route_message = AsyncMock(side_effect=RuntimeError("Test error"))

    # Create a message
    message = HumanMessage(content="Test message")

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", AsyncMock()):
        # Handle the message with retry
        with pytest.raises(RetryError, match="Max retries exceeded"):
            await handler.handle_message_with_retry(message, "test_agent", max_retries=2)

        # Check if router.route_message was called multiple times
        assert handler.router.route_message.call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_handle_message_with_retry_success_after_retry() -> None:
    """Test handle_message_with_retry method with success after retry."""
    handler = MessageHandler()
    mock_agent = MagicMock()
    mock_result = MagicMock()

    # Register an agent
    handler.register_agent("test_agent", mock_agent)

    # Mock router.route_message to fail once then succeed
    side_effects = [RuntimeError("Test error"), mock_result]
    handler.router.route_message = AsyncMock(side_effect=side_effects)

    # Create a message
    message = HumanMessage(content="Test message")

    # Mock asyncio.sleep to avoid waiting
    with patch("asyncio.sleep", AsyncMock()):
        # Handle the message with retry
        result = await handler.handle_message_with_retry(message, "test_agent")

        # Check if router.route_message was called twice
        assert handler.router.route_message.call_count == 2

        # Check if result is correct
        assert result == mock_result


@pytest.mark.asyncio
async def test_handle_message_with_retry_different_exceptions() -> None:
    """Test handle_message_with_retry method with different exceptions."""
    handler = MessageHandler()
    mock_agent = MagicMock()

    # Register an agent
    handler.register_agent("test_agent", mock_agent)

    # Test with different exception types
    exception_types = [
        ValueError("Test error"),
        TypeError("Test error"),
        AttributeError("Test error"),
        KeyError("Test error"),
        IndexError("Test error"),
        OSError("Test error"),
        RuntimeError("Test error"),
        ConnectionError("Test error"),
    ]

    for exception in exception_types:
        # Reset mock
        handler.router.route_message = AsyncMock(side_effect=exception)

        # Create a message
        message = HumanMessage(content="Test message")

        # Mock asyncio.sleep to avoid waiting
        with patch("asyncio.sleep", AsyncMock()):
            # Handle the message with retry
            with pytest.raises(RetryError, match="Max retries exceeded"):
                await handler.handle_message_with_retry(message, "test_agent", max_retries=1)

            # Check if router.route_message was called multiple times
            assert handler.router.route_message.call_count == 2  # Initial + 1 retry
