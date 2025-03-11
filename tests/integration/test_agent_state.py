"""Test agent state management functionality."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.agent.agent_types.agent_types import StepResult
from src.agent.state.base import (
    AgentState,
    Context,
    FileStateManager,
    InMemoryStateManager,
)
from src.common_types.enums import AgentStep
from src.config import ConfigError
from src.messages import create_human_message, create_system_message


def test_context_validation() -> None:
    """Test context validation."""
    # Create context with required metadata
    context = Context()
    context.metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "change_count": 0,
    }

    # Test valid context
    assert context.validate()

    # Test invalid context (missing required metadata)
    invalid_context = Context()
    with pytest.raises(ConfigError):
        invalid_context.validate()

    # Test tracking changes
    context.track_changes()
    assert context.metadata["change_count"] == 1


def test_agent_state_initialization() -> None:
    """Test agent state initialization."""
    # Test default initialization
    state = AgentState()
    assert state.current_step == AgentStep.UNDERSTAND
    assert state.step_count == 0
    assert state.task_completed is False
    assert state.error is None
    assert state.agent_id != ""  # Should generate UUID

    # Test context metadata initialization
    assert "created_at" in state.context.metadata
    assert "updated_at" in state.context.metadata
    assert "change_count" in state.context.metadata

    # Test custom initialization
    custom_state = AgentState(
        current_step=AgentStep.PLAN,
        step_count=2,
        task_completed=True,
        agent_id="custom-id",
    )
    assert custom_state.current_step == AgentStep.PLAN
    assert custom_state.step_count == 2
    assert custom_state.task_completed is True
    assert custom_state.agent_id == "custom-id"


def test_agent_state_message_handling() -> None:
    """Test agent state message handling."""
    state = AgentState()

    # Test adding messages
    msg1 = create_human_message("Test message 1")
    msg2 = create_system_message("Test message 2")

    state.add_message(msg1)
    state.add_message(msg2)

    assert len(state.messages) == 2
    assert state.messages[0] == msg1
    assert state.messages[1] == msg2

    # Test getting message
    assert state.get_message(0) == msg1
    assert state.get_message(1) == msg2

    # Test setting and getting message metadata
    state.set_message_metadata(0, "test_key", {"value": "test_value"})
    metadata = state.get_message_metadata(0, "test_key")
    assert metadata == {"value": "test_value"}


def test_agent_state_context_handling() -> None:
    """Test agent state context handling."""
    state = AgentState()

    # Test setting and getting context values
    state.set_context("test_key", "test_value")
    assert state.get_context("test_key") == "test_value"

    # Test default value for non-existent key
    assert state.get_context("non_existent", "default") == "default"

    # Test context change tracking
    initial_count = state.context.metadata["change_count"]
    state.set_context("another_key", "another_value")
    assert state.context.metadata["change_count"] > initial_count


def test_agent_state_validation() -> None:
    """Test agent state validation."""
    # Test valid state
    state = AgentState()
    assert state.validate()

    # Test invalid step
    state.current_step = "invalid_step"  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="Invalid step: invalid_step"):
        state.validate()

    # Test invalid agent_id
    invalid_state = AgentState()
    invalid_state.agent_id = ""
    with pytest.raises(ConfigError):
        invalid_state.validate()


def test_agent_state_step_results() -> None:
    """Test agent state step results handling."""
    state = AgentState()

    # Test recording step results
    result1 = StepResult(success=True, data="Test result 1", error="")
    state.record_step_result(AgentStep.UNDERSTAND, result1)

    # Test getting step results
    retrieved_result = state.get_step_result(AgentStep.UNDERSTAND)
    assert retrieved_result is not None
    assert retrieved_result.success
    assert retrieved_result.data == "Test result 1"

    # Test non-existent step result
    assert state.get_step_result(AgentStep.PLAN) is None


def test_agent_state_serialization() -> None:
    """Test agent state serialization and deserialization."""
    # Create state with data
    state = AgentState()
    state.add_message(create_human_message("Test message"))
    state.set_context("test_key", "test_value")
    state.record_step_result(
        AgentStep.UNDERSTAND,
        StepResult(success=True, data="Test result", error=""),
    )

    # Serialize to dict
    state_dict = state.to_dict()

    # Check serialized data
    assert len(state_dict["messages"]) == 1
    assert state_dict["messages"][0]["content"] == "Test message"
    assert state_dict["context"]["data"]["test_key"] == "test_value"
    assert state_dict["step_results"][AgentStep.UNDERSTAND.value]["success"] is True

    # Deserialize from dict
    new_state = AgentState.from_dict(state_dict)

    # Check deserialized data
    assert len(new_state.messages) == 1
    assert new_state.messages[0].content == "Test message"
    assert new_state.get_context("test_key") == "test_value"
    result = new_state.get_step_result(AgentStep.UNDERSTAND)
    assert result is not None
    assert result.success is True


def test_in_memory_state_manager() -> None:
    """Test in-memory state manager."""
    manager = InMemoryStateManager()

    # Test initial state
    state = manager.get_state()
    assert state is not None

    # Test setting state
    new_state = AgentState(agent_id="test-id")
    manager.set_state(new_state)
    retrieved_state = manager.get_state()
    assert retrieved_state.agent_id == "test-id"

    # Test clearing state
    manager.clear_state()
    cleared_state = manager.get_state()
    assert cleared_state.agent_id != "test-id"  # Should generate new UUID

    # Test saving and loading state
    with tempfile.TemporaryDirectory() as temp_dir:
        state_path = Path(temp_dir) / "test_state.json"
        test_state = AgentState(agent_id="save-test-id")
        manager.set_state(test_state)
        saved_path = manager.save_state(str(state_path))

        # Check saved file
        assert Path(saved_path).exists()

        # Load state from file
        loaded_state = AgentState.from_dict(
            json.loads(Path(saved_path).read_text(encoding="utf-8")),
        )
        assert loaded_state.agent_id == "save-test-id"


def test_file_state_manager() -> None:
    """Test file state manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = FileStateManager(base_dir=temp_dir)

        # Test initial state
        state = manager.get_state()
        assert state is not None

        # Test setting state
        new_state = AgentState(agent_id="test-id")
        manager.set_state(new_state)
        retrieved_state = manager.get_state()
        assert retrieved_state.agent_id == "test-id"

        # Test saving state
        saved_path = manager.save_state()
        assert Path(saved_path).exists()
        assert Path(saved_path).name == "test-id.json"

        # Test listing states
        states = manager.list_states()
        assert len(states) == 1
        assert states[0] == "test-id"  # Should return just the agent ID, not the full path

        # Test getting state by ID
        retrieved_state = manager.get_state_by_id("test-id")
        assert retrieved_state.agent_id == "test-id"

        # Test non-existent state
        with pytest.raises(ConfigError):
            manager.get_state_by_id("non-existent-id")


def test_state_manager() -> None:
    """Test state manager functionality."""
    manager = InMemoryStateManager()
    state = AgentState(agent_id="test-id")

    # Test saving state
    manager.set_state(state)
    saved_path = manager.save_state()
    assert Path(saved_path).exists()

    # Test loading state from file
    loaded_state = AgentState.from_dict(
        json.loads(Path(saved_path).read_text(encoding="utf-8")),
    )
    assert loaded_state.agent_id == state.agent_id

    # Test listing states
    states = manager.list_states()
    assert len(states) == 1
    assert states[0] == "test-id"  # Should return just the agent ID, not the full path

    # Test deleting state
    manager.delete_state(state.agent_id)
    with pytest.raises(ConfigError):
        manager.load_state(state.agent_id)


def test_state_manager_auto_path() -> None:
    """Test state manager with auto path generation."""
    manager = InMemoryStateManager()
    state = AgentState(agent_id="test-id")

    # Test saving state with auto path
    manager.set_state(state)
    saved_path = manager.save_state()
    assert Path(saved_path).exists()
    assert Path(saved_path).name == "test-id.json"

    # Test loading state from file
    loaded_state = AgentState.from_dict(
        json.loads(Path(saved_path).read_text(encoding="utf-8")),
    )
    assert loaded_state.agent_id == state.agent_id


def test_state_manager_errors() -> None:
    """Test state manager error handling."""
    manager = InMemoryStateManager()

    # Test loading non-existent state
    with pytest.raises(ConfigError):
        manager.load_state("non-existent")

    # Test deleting non-existent state
    manager.delete_state("non-existent")  # Should not raise
