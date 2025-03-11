"""Tests for agent enums."""

import pytest

from src.agent.agent_types.enums import AgentStatus, MessageRole


def test_agent_status_enum() -> None:
    """Test AgentStatus enum values."""
    # Test enum values
    assert AgentStatus.IDLE == "idle"
    assert AgentStatus.PROCESSING == "processing"
    assert AgentStatus.ERROR == "error"
    assert AgentStatus.DONE == "done"

    # Test enum creation from string
    assert AgentStatus("idle") == AgentStatus.IDLE
    assert AgentStatus("processing") == AgentStatus.PROCESSING
    assert AgentStatus("error") == AgentStatus.ERROR
    assert AgentStatus("done") == AgentStatus.DONE

    # Test invalid enum value
    with pytest.raises(ValueError, match="'invalid_status' is not a valid AgentStatus"):
        AgentStatus("invalid_status")


def test_message_role_enum() -> None:
    """Test MessageRole enum values."""
    # Test enum values
    assert MessageRole.SYSTEM == "system"
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.TOOL == "tool"

    # Test enum creation from string
    assert MessageRole("system") == MessageRole.SYSTEM
    assert MessageRole("user") == MessageRole.USER
    assert MessageRole("assistant") == MessageRole.ASSISTANT
    assert MessageRole("tool") == MessageRole.TOOL

    # Test invalid enum value
    with pytest.raises(ValueError, match="'invalid_role' is not a valid MessageRole"):
        MessageRole("invalid_role")
