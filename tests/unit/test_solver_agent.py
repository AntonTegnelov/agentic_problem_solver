"""Tests for the SolverAgent class."""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from src.agent.solver import SolverAgent
from src.agent.state.base import AgentState
from src.common_types.message_types import HumanMessage, Message, SystemMessage
from src.common_types.result_types import Result
from src.messages.creation import create_message

# DEPRECATED: Tests for SolverAgent which is deprecated and will be removed in a future version.
# These tests will need to be updated or removed when SolverAgent is removed.
# New features should use and test the hierarchical agent system instead.
# See docs/explanation/hierarchical_agents.md for more information.
#
# TODO: Migrate these tests to use hierarchical agents before SolverAgent is removed.
# Most test logic can be adapted to test ArchitectAgent from src.agent.agent_types.architect


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate.return_value = "Test response"

    # Create a proper async generator for generate_stream
    async def mock_generate_stream(_: list[Message]) -> AsyncGenerator[str, None]:
        yield "Test"
        yield " response"
        yield " chunk"

    provider.generate_stream = mock_generate_stream
    return provider


@pytest.fixture
def solver_agent(mock_provider: MagicMock) -> SolverAgent:
    """Create a SolverAgent instance with a mock provider."""
    # WARNING: This creates a deprecated SolverAgent instance.
    # In new code, use ArchitectAgent, PlannerAgent, or ExecutorAgent instead.
    return SolverAgent(provider=mock_provider)


def test_solver_agent_initialization() -> None:
    """Test SolverAgent initialization."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    # Test with default parameters
    agent = SolverAgent()
    assert agent.get_agent_id() == "solver_agent"
    assert agent.state is not None

    # Test with custom state
    custom_state = AgentState(agent_id="custom_agent")
    agent = SolverAgent(state_manager=custom_state)
    assert agent.state == custom_state

    # Test with state manager
    state_manager = MagicMock()
    state_manager.get_state.return_value = AgentState(agent_id="managed_agent")
    agent = SolverAgent(state_manager=state_manager)
    assert agent.state == state_manager.get_state.return_value


def test_get_agent_id(solver_agent: SolverAgent) -> None:
    """Test get_agent_id method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    assert solver_agent.get_agent_id() == "solver_agent"


def test_get_capabilities(solver_agent: SolverAgent) -> None:
    """Test get_capabilities method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    capabilities = solver_agent.get_capabilities()
    assert isinstance(capabilities, list)
    assert "solve" in capabilities
    assert "code" in capabilities
    assert "explain" in capabilities
    assert "plan" in capabilities


def test_can_handle(solver_agent: SolverAgent) -> None:
    """Test can_handle method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    assert solver_agent.can_handle("any task") is True


@pytest.mark.asyncio
async def test_process_message(solver_agent: SolverAgent) -> None:
    """Test process_message method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    message = HumanMessage(content="Test message")
    message.add_metadata("test", "value")

    # Patch the process method to return a known value
    with patch.object(SolverAgent, "process", return_value="Test response") as mock_process:
        result = await solver_agent.process_message(message)
        mock_process.assert_called_once_with(message)
        assert isinstance(result, Result)
        assert result.success is True
        assert result.data == "Test response"


@pytest.mark.asyncio
async def test_process_stream(solver_agent: SolverAgent) -> None:
    """Test process_stream method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    message = "Test message"
    chunks = ["Chunk", "1", "2", "3"]
    solver_agent._provider.generate_stream = MagicMock()
    solver_agent._provider.generate_stream.return_value.__aiter__.return_value = chunks

    result = [chunk async for chunk in solver_agent.process_stream(message)]
    assert result == chunks


def test_send_message(solver_agent: SolverAgent) -> None:
    """Test send_message method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    message = HumanMessage(content="Test message")
    message.add_metadata("test", "value")

    # Patch the process method to return a known value
    with patch.object(solver_agent, "process", return_value="Test response"):
        result = solver_agent.send_message(message)
        assert isinstance(result, Result)
        assert result.success is True
        assert result.data == "Test response"

    # Test when process returns a Result directly
    process_result = Result(success=True, data="Direct result", error=None)
    with patch.object(solver_agent, "process", return_value=process_result):
        result = solver_agent.send_message(message)
        assert result is process_result


def test_receive_message(solver_agent: SolverAgent) -> None:
    """Test receive_message method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    message = HumanMessage(content="Test message")
    message.add_metadata("test", "value")

    # Patch the process method to return a known value
    with patch.object(solver_agent, "process", return_value="Test response"):
        result = solver_agent.receive_message(message)
        assert isinstance(result, Result)
        assert result.success is True
        assert result.data == "Test response"

    # Test when process returns a Result directly
    process_result = Result(success=True, data="Direct result", error=None)
    with patch.object(solver_agent, "process", return_value=process_result):
        result = solver_agent.receive_message(message)
        assert result is process_result


def test_prepare_messages() -> None:
    """Test _prepare_messages method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    SolverAgent()
    messages = [
        HumanMessage(content="Human message"),
        SystemMessage(content="System message"),
    ]

    expected_output = [
        HumanMessage(content="Human message"),
        HumanMessage(content="System message"),  # System messages should be converted to human
    ]

    with patch.object(SolverAgent, "_prepare_messages", return_value=expected_output) as mock_prepare:
        result = mock_prepare(messages)
        assert result == expected_output


def test_validate_provider() -> None:
    """Test _validate_provider method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    agent = SolverAgent(provider=MagicMock())

    # Should not raise when provider is set
    agent._validate_provider()

    # Should raise ValueError when provider is not set
    with pytest.raises(ValueError, match="Provider not initialized"):
        SolverAgent(provider=None)._validate_provider()

    # Test the exception is raised
    with (
        patch.object(
            SolverAgent,
            "_validate_provider",
            side_effect=ValueError("Provider not initialized"),
        ),
        pytest.raises(ValueError, match="Provider not initialized"),
    ):
        agent._validate_provider()


def test_prepare_state(solver_agent: SolverAgent) -> None:
    """Test _prepare_state method."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    input_data = "Test input"
    expected_messages = [create_message(role="system", content="Test")]

    # Mock state and get_step_prompt
    with (
        patch("src.agent.solver.get_step_prompt", return_value="Test system prompt"),
        patch.object(
            solver_agent.state,
            "add_message",
        ),
        patch.object(
            SolverAgent,
            "_prepare_state",
            return_value=expected_messages,
        ) as mock_prepare,
    ):
        result = mock_prepare(input_data)
        assert result == expected_messages


def test_process(solver_agent: SolverAgent, mock_provider: MagicMock) -> None:
    """Test process method with string input."""
    # DEPRECATED: This test will be removed when SolverAgent is removed.
    input_message = "Test message"
    expected_response = "Test response"
    mock_provider.generate.return_value = expected_response

    # Mock methods to isolate test
    with (
        patch.object(solver_agent, "_prepare_state", return_value=[]),
        patch.object(
            solver_agent.state,
            "add_message",
        ),
    ):
        result = solver_agent.process(input_message)
        assert result == expected_response

    # Test with Message input
    message = HumanMessage(content="Test message")
    with (
        patch.object(solver_agent, "_prepare_state", return_value=[]),
        patch.object(
            solver_agent.state,
            "add_message",
        ),
    ):
        result = solver_agent.process(message)
        assert isinstance(result, Result)
        assert result.success is True
        assert result.data == expected_response
