"""Unit tests for validation models."""

import pytest

from src.agents.base import AgentState
from src.config import AgentStep
from src.validation import AgentConfig


def test_agent_state_validation():
    """Test AgentState validation."""
    # Test valid state with initial message
    state = AgentState()
    state.add_system_message("Initial message")

    assert len(state.messages) == 1
    assert state.current_step == 0
    assert state.error is None
    assert state.result is None

    # Test metadata operations
    state.set_message_metadata(state.messages[0], "test_key", "test_value")
    assert state.get_message_metadata(state.messages[0], "test_key") == "test_value"
    assert (
        state.get_message_metadata(state.messages[0], "non_existent", "default")
        == "default"
    )
    assert state.get_message_metadata(state.messages[0], "content") == "Initial message"


def test_agent_config_validation():
    """Test AgentConfig validation."""
    # Test default values
    config = AgentConfig()
    assert config.temperature == 0.7
    assert config.max_tokens is None  # max_tokens can be None
    assert config.model == "gemini-2.0-flash-lite"
    assert config.task_timeout > 0
    assert config.max_retries >= 0
    assert config.max_steps > 0

    # Test invalid values
    with pytest.raises(ValueError):
        AgentConfig(temperature=1.5)
    with pytest.raises(ValueError):
        AgentConfig(max_tokens=0)
    with pytest.raises(ValueError):
        AgentConfig(task_timeout=0)
    with pytest.raises(ValueError):
        AgentConfig(max_retries=-1)
    with pytest.raises(ValueError):
        AgentConfig(max_steps=0)


def test_agent_step_enum():
    """Test AgentStep enum."""
    assert AgentStep.UNDERSTAND.value == "understand"
    assert AgentStep.PLAN.value == "plan"
    assert AgentStep.EXECUTE.value == "execute"
    assert AgentStep.VERIFY.value == "verify"
    assert AgentStep.END.value == "end"

    # Test all steps are unique
    steps = [step.value for step in AgentStep]
    assert len(steps) == len(set(steps))


def test_message_types():
    """Test different message types and metadata handling."""
    state = AgentState()
    state.add_system_message("System message")
    state.add_human_message("Human message")
    state.add_ai_message("AI message")

    # Test content access
    assert state.get_message_metadata(state.messages[0], "content") == "System message"
    assert state.get_message_metadata(state.messages[1], "content") == "Human message"
    assert state.get_message_metadata(state.messages[2], "content") == "AI message"

    # Test type access
    assert state.get_message_metadata(state.messages[0], "type") == "system"
    assert state.get_message_metadata(state.messages[1], "type") == "human"
    assert state.get_message_metadata(state.messages[2], "type") == "ai"

    # Test metadata operations
    state.set_message_metadata(state.messages[0], "test_key", "test_value")
    assert state.get_message_metadata(state.messages[0], "test_key") == "test_value"

    # Test out of range index
    try:
        message = state.messages[99]
    except IndexError:
        message = None
    assert state.get_message_metadata(message, "content", "default") == "default"
