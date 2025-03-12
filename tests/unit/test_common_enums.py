"""Tests for common enumerations."""

import pytest

from src.common_types.enums import (
    AgentStatus,
    AgentStep,
    LogLevel,
    MessagePriority,
    MessageRole,
)


def test_agent_step_enum() -> None:
    """Test AgentStep enum values."""
    # Test enum values
    assert AgentStep.UNDERSTAND == "understand"
    assert AgentStep.PLAN == "plan"
    assert AgentStep.EXECUTE == "execute"
    assert AgentStep.VERIFY == "verify"

    # Test enum creation from string
    assert AgentStep("understand") == AgentStep.UNDERSTAND
    assert AgentStep("plan") == AgentStep.PLAN
    assert AgentStep("execute") == AgentStep.EXECUTE
    assert AgentStep("verify") == AgentStep.VERIFY

    # Test invalid enum value
    with pytest.raises(ValueError, match="'invalid_step' is not a valid AgentStep"):
        AgentStep("invalid_step")


def test_agent_status_enum() -> None:
    """Test AgentStatus enum values."""
    # Test enum values
    assert AgentStatus.IDLE == "idle"
    assert AgentStatus.BUSY == "busy"
    assert AgentStatus.PROCESSING == "processing"  # Alias for BUSY
    assert AgentStatus.ERROR == "error"
    assert AgentStatus.COMPLETED == "completed"
    assert AgentStatus.DONE == "done"  # Alias for COMPLETED

    # Test enum creation from string
    assert AgentStatus("idle") == AgentStatus.IDLE
    assert AgentStatus("busy") == AgentStatus.BUSY
    assert AgentStatus("processing") == AgentStatus.PROCESSING
    assert AgentStatus("error") == AgentStatus.ERROR
    assert AgentStatus("completed") == AgentStatus.COMPLETED
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


def test_log_level_enum() -> None:
    """Test LogLevel enum values."""
    # Test enum values
    assert LogLevel.DEBUG == "debug"
    assert LogLevel.INFO == "info"
    assert LogLevel.WARNING == "warning"
    assert LogLevel.ERROR == "error"
    assert LogLevel.CRITICAL == "critical"

    # Test enum creation from string
    assert LogLevel("debug") == LogLevel.DEBUG
    assert LogLevel("info") == LogLevel.INFO
    assert LogLevel("warning") == LogLevel.WARNING
    assert LogLevel("error") == LogLevel.ERROR
    assert LogLevel("critical") == LogLevel.CRITICAL

    # Test invalid enum value
    with pytest.raises(ValueError, match="'invalid_level' is not a valid LogLevel"):
        LogLevel("invalid_level")


def test_message_priority_enum() -> None:
    """Test MessagePriority enum values and comparison methods."""
    # Test enum values
    assert MessagePriority.LOW.value == 1
    assert MessagePriority.NORMAL.value == 2
    assert MessagePriority.HIGH.value == 3
    assert MessagePriority.CRITICAL.value == 4

    # Test comparison methods
    # Less than
    assert MessagePriority.LOW < MessagePriority.NORMAL
    assert MessagePriority.NORMAL < MessagePriority.HIGH
    assert MessagePriority.HIGH < MessagePriority.CRITICAL
    assert not (MessagePriority.CRITICAL < MessagePriority.HIGH)

    # Less than or equal
    assert MessagePriority.LOW <= MessagePriority.LOW
    assert MessagePriority.LOW <= MessagePriority.NORMAL
    assert not (MessagePriority.HIGH <= MessagePriority.NORMAL)

    # Greater than
    assert MessagePriority.CRITICAL > MessagePriority.HIGH
    assert MessagePriority.HIGH > MessagePriority.NORMAL
    assert MessagePriority.NORMAL > MessagePriority.LOW
    assert not (MessagePriority.LOW > MessagePriority.NORMAL)

    # Greater than or equal
    assert MessagePriority.HIGH >= MessagePriority.HIGH
    assert MessagePriority.HIGH >= MessagePriority.NORMAL
    assert not (MessagePriority.NORMAL >= MessagePriority.HIGH)
