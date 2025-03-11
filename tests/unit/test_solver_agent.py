"""Tests for the SolverAgent class."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.result import Result
from src.agent.solver import SolverAgent
from src.agent.state.base import AgentState
from src.common_types.message_types import AIMessage, HumanMessage, SystemMessage
from src.messages.creation import create_message


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate.return_value = "Test response"

    # Create a proper async generator for generate_stream
    async def mock_generate_stream(*args, **kwargs):
        yield "Test"
        yield " response"
        yield " chunk"

    provider.generate_stream = mock_generate_stream
    return provider


@pytest.fixture
def solver_agent(mock_provider):
    """Create a SolverAgent instance with a mock provider."""
    return SolverAgent(provider=mock_provider)


def test_solver_agent_initialization() -> None:
    """Test SolverAgent initialization."""
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


def test_get_agent_id(solver_agent) -> None:
    """Test get_agent_id method."""
    assert solver_agent.get_agent_id() == "solver_agent"


def test_get_capabilities(solver_agent) -> None:
    """Test get_capabilities method."""
    capabilities = solver_agent.get_capabilities()
    assert isinstance(capabilities, list)
    assert "solve" in capabilities
    assert "code" in capabilities
    assert "explain" in capabilities
    assert "plan" in capabilities


def test_can_handle(solver_agent) -> None:
    """Test can_handle method."""
    assert solver_agent.can_handle("any_task") is True


@pytest.mark.asyncio
async def test_process_message(solver_agent) -> None:
    """Test process_message method."""
    message = create_message(role="human", content="Test message")
    result = await solver_agent.process_message(message)

    assert isinstance(result, Result)
    assert result.success is True
    assert result.data == "Test response"
    assert result.error is None


@pytest.mark.asyncio
async def test_process_stream(solver_agent) -> None:
    """Test process_stream method."""
    message = create_message(role="human", content="Test message")
    chunks = []

    async for chunk in solver_agent.process_stream(message):
        chunks.append(chunk)

    assert chunks == ["Test", " response", " chunk"]


def test_send_message(solver_agent) -> None:
    """Test send_message method."""
    message = create_message(role="human", content="Test message")
    result = solver_agent.send_message(message)

    assert isinstance(result, Result)
    assert result.success is True
    assert result.data == "Test response"
    assert result.error is None


def test_receive_message(solver_agent) -> None:
    """Test receive_message method."""
    message = create_message(role="human", content="Test message")
    result = solver_agent.receive_message(message)

    assert isinstance(result, Result)
    assert result.success is True
    assert result.data == "Test response"
    assert result.error is None


def test_prepare_messages(solver_agent) -> None:
    """Test _prepare_messages method."""
    system_message = SystemMessage(content="System message")
    human_message = HumanMessage(content="Human message")

    prepared_messages = solver_agent._prepare_messages([system_message, human_message])

    assert len(prepared_messages) == 2
    assert isinstance(prepared_messages[0], HumanMessage)  # System message converted to human
    assert prepared_messages[0].content == "System message"
    assert prepared_messages[1] == human_message  # Human message unchanged


def test_validate_provider(solver_agent, mock_provider) -> None:
    """Test _validate_provider method."""
    # Should not raise an error with a valid provider
    solver_agent._validate_provider()

    # Should raise an error without a provider
    agent_without_provider = SolverAgent(provider=None)
    with pytest.raises(ValueError, match="Provider not initialized"):
        agent_without_provider._validate_provider()


def test_prepare_state(solver_agent) -> None:
    """Test _prepare_state method."""
    with patch("src.agent.solver.get_step_prompt", return_value="Test prompt"):
        messages = solver_agent._prepare_state("Test input")

        # Check that messages were added to state
        assert len(solver_agent.state.messages) == 2
        assert isinstance(solver_agent.state.messages[0], HumanMessage)
        assert solver_agent.state.messages[0].content == "Test input"
        assert isinstance(solver_agent.state.messages[1], SystemMessage)
        assert solver_agent.state.messages[1].content == "Test prompt"

        # Check returned messages
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Test input"
        assert isinstance(messages[1], HumanMessage)  # System message converted to human
        assert messages[1].content == "Test prompt"


def test_process(solver_agent, mock_provider) -> None:
    """Test process method."""
    with patch("src.agent.solver.get_step_prompt", return_value="Test prompt"):
        response = solver_agent.process("Test input")

        # Check that provider was called
        mock_provider.generate.assert_called_once()

        # Check that response was added to state
        assert len(solver_agent.state.messages) == 3
        assert isinstance(solver_agent.state.messages[2], AIMessage)
        assert solver_agent.state.messages[2].content == "Test response"

        # Check returned response
        assert response == "Test response"
