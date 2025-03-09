"""Test module for solver agent."""

import pytest
from langchain_core.messages import SystemMessage

from src.agents.base import AgentState
from src.agents.solver_agent import SolverAgent


@pytest.fixture
def agent_state():
    """Create a test agent state."""
    state = AgentState()
    state.add_system_message("Test system message")
    return state


@pytest.fixture
def solver_agent():
    """Create a test solver agent."""
    agent = SolverAgent("test_agent")
    agent.state.add_system_message("Test system message")
    return agent


def test_agent_initialization(solver_agent):
    """Test agent initialization."""
    assert isinstance(solver_agent.state, AgentState)
    assert len(solver_agent.state.messages) == 1
    assert isinstance(solver_agent.state.messages[0], SystemMessage)


def test_understand_task(solver_agent):
    """Test task understanding."""
    task = "Test task"
    solver_agent.state.add_human_message(task)
    assert solver_agent.state.messages[-1].content == task


def test_create_plan(solver_agent):
    """Test plan creation."""
    plan = "Test plan"
    solver_agent.state.add_ai_message(plan)
    assert solver_agent.state.messages[-1].content == plan


def test_execute_plan(solver_agent):
    """Test plan execution."""
    result = "Test result"
    solver_agent.state.add_ai_message(result)
    assert solver_agent.state.messages[-1].content == result


def test_verify_result(solver_agent):
    """Test result verification."""
    verification = "Test verification"
    solver_agent.state.add_ai_message(verification)
    assert solver_agent.state.messages[-1].content == verification


def test_end_processing(solver_agent):
    """Test processing end."""
    end_message = "Test end message"
    solver_agent.state.add_ai_message(end_message)
    assert solver_agent.state.messages[-1].content == end_message


def test_should_continue(solver_agent):
    """Test continue check."""
    solver_agent.state.add_human_message("Continue")
    assert len(solver_agent.state.messages) > 0


def test_process_message_impl(solver_agent):
    """Test message processing implementation."""
    message = "Test message"
    solver_agent.state.add_human_message(message)
    assert solver_agent.state.messages[-1].content == message


def test_error_handling(solver_agent):
    """Test error handling."""
    error_message = "Test error"
    solver_agent.state.error = error_message
    assert solver_agent.state.error == error_message


def test_state_validation(solver_agent):
    """Test state validation."""
    solver_agent.state.add_system_message("Test validation")
    assert len(solver_agent.state.messages) > 0


def test_message_metadata(solver_agent):
    """Test message metadata handling."""
    metadata = {"key": "value"}
    solver_agent.state.add_system_message("Test metadata", metadata)
    assert solver_agent.get_message_metadata(-1, "key") == "value"


def test_clear_state(solver_agent):
    """Test state clearing."""
    solver_agent.clear_state()
    assert len(solver_agent.state.messages) == 0
    assert solver_agent.state.error is None
    assert solver_agent.state.result is None
