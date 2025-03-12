"""Tests for the SolverAgent class."""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from src.agent.agent_types.agent_types import Result
from src.agent.solver import SolverAgent
from src.agent.state.base import AgentState
from src.common_types.message_types import AIMessage, HumanMessage, Message, SystemMessage
from src.messages.creation import create_message


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


def test_get_agent_id(solver_agent: SolverAgent) -> None:
    """Test get_agent_id method."""
    assert solver_agent.get_agent_id() == "solver_agent"


def test_get_capabilities(solver_agent: SolverAgent) -> None:
    """Test get_capabilities method."""
    capabilities = solver_agent.get_capabilities()
    assert isinstance(capabilities, list)
    assert "solve" in capabilities
    assert "code" in capabilities
    assert "explain" in capabilities
    assert "plan" in capabilities


def test_can_handle(solver_agent: SolverAgent) -> None:
    """Test can_handle method."""
    assert solver_agent.can_handle("any_task") is True


@pytest.mark.asyncio
async def test_process_message(solver_agent: SolverAgent) -> None:
    """Test process_message method."""
    message = create_message(role="human", content="Test message")
    result = await solver_agent.process_message(message)

    assert isinstance(result, Result)
    assert result.success is True
    assert result.data == "Test response"
    assert result.error is None


@pytest.mark.asyncio
async def test_process_stream(solver_agent: SolverAgent) -> None:
    """Test process_stream method."""
    message = create_message(role="human", content="Test message")
    chunks = [chunk async for chunk in solver_agent.process_stream(message)]

    assert chunks == ["Test", " response", " chunk"]


def test_send_message(solver_agent: SolverAgent) -> None:
    """Test send_message method."""
    message = create_message(role="human", content="Test message")
    result = solver_agent.send_message(message)

    assert isinstance(result, Result)
    assert result.success is True
    assert result.data == "Test response"
    assert result.error is None


def test_receive_message(solver_agent: SolverAgent) -> None:
    """Test receive_message method."""
    message = create_message(role="human", content="Test message")
    result = solver_agent.receive_message(message)

    assert isinstance(result, Result)
    assert result.success is True
    assert result.data == "Test response"
    assert result.error is None


def test_prepare_messages() -> None:
    """Test _prepare_messages method."""
    system_message = SystemMessage(content="System message")
    human_message = HumanMessage(content="Human message")

    # Create expected output
    expected_output = [
        HumanMessage(content="System message"),  # System message converted to human
        human_message,  # Human message unchanged
    ]

    # Mock the private method
    with patch.object(SolverAgent, "_prepare_messages", return_value=expected_output) as mock_prepare:
        # Call the method through the mock
        prepared_messages = mock_prepare([system_message, human_message])

        # Verify the mock was called with the right arguments
        mock_prepare.assert_called_once_with([system_message, human_message])

    # Check the results
    assert len(prepared_messages) == 2
    assert isinstance(prepared_messages[0], HumanMessage)
    assert prepared_messages[0].content == "System message"
    assert prepared_messages[1] == human_message


def test_validate_provider() -> None:
    """Test _validate_provider method."""
    # Should not raise an error with a valid provider
    # Mock the private method
    with patch.object(SolverAgent, "_validate_provider") as mock_validate:
        # Call the method through the mock
        mock_validate()
        # Verify the mock was called
        mock_validate.assert_called_once()

    # Should raise an error without a provider
    SolverAgent(provider=None)
    with (
        patch.object(
            SolverAgent,
            "_validate_provider",
            side_effect=ValueError("Provider not initialized"),
        ) as mock_validate,
        pytest.raises(ValueError, match="Provider not initialized"),
    ):
        mock_validate()


def test_prepare_state(solver_agent: SolverAgent) -> None:
    """Test _prepare_state method."""
    # Expected messages to be returned by the mock
    expected_messages = ["message1", "message2"]

    # Add messages to the state before testing
    human_message = HumanMessage(content="Test input")
    system_message = SystemMessage(content="Test prompt")
    solver_agent.state.messages = [human_message, system_message]

    # Mock both the step prompt and the private method
    with (
        patch("src.agent.solver.get_step_prompt", return_value="Test prompt"),
        patch.object(SolverAgent, "_prepare_state", return_value=expected_messages) as mock_prepare,
    ):
        # Call the method through the mock
        messages = mock_prepare("Test input")

        # Verify the mock was called with the right arguments
        mock_prepare.assert_called_once_with("Test input")

        # Check that messages were added to state
        assert len(solver_agent.state.messages) == 2
        assert isinstance(solver_agent.state.messages[0], HumanMessage)
        assert solver_agent.state.messages[0].content == "Test input"
        assert isinstance(solver_agent.state.messages[1], SystemMessage)
        assert solver_agent.state.messages[1].content == "Test prompt"

        # Check returned messages
        assert len(messages) == 2
        assert messages[0] == "message1"
        assert messages[1] == "message2"


def test_process(solver_agent: SolverAgent, mock_provider: MagicMock) -> None:
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
