"""Unit tests for specialized agent types."""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.planner import PlannerAgent
from src.agent.state.base import AgentState
from src.common_types.result_types import Result


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate.return_value = "Mock response"
    provider.generate_stream = MagicMock()

    async def mock_stream(*_: object) -> AsyncGenerator[str, None]:
        yield "Mock"
        yield " stream"
        yield " response"

    provider.generate_stream.side_effect = mock_stream
    return provider


@pytest.fixture
def architect_agent(mock_provider: MagicMock) -> ArchitectAgent:
    """Create an ArchitectAgent instance with a mock provider."""
    return ArchitectAgent(provider=mock_provider)


@pytest.fixture
def planner_agent(mock_provider: MagicMock) -> PlannerAgent:
    """Create a PlannerAgent instance with a mock provider."""
    return PlannerAgent(provider=mock_provider)


class TestArchitectAgent:
    """Tests for the ArchitectAgent class."""

    def test_initialization(self) -> None:
        """Test ArchitectAgent initialization."""
        # Test with default parameters
        agent = ArchitectAgent()
        assert agent.get_agent_id().startswith("architect_")
        assert agent.state is not None
        assert agent.get_parent_id() is None
        assert agent.get_child_ids() == []

        # Test with custom state
        custom_state = AgentState(agent_id="custom_agent")
        agent = ArchitectAgent(state_manager=custom_state)
        assert agent.state == custom_state

        # Test with state manager
        state_manager = MagicMock()
        state_manager.get_state.return_value = AgentState(agent_id="managed_agent")
        agent = ArchitectAgent(state_manager=state_manager)
        assert agent.state == state_manager.get_state.return_value

    def test_get_agent_id(self, architect_agent: ArchitectAgent) -> None:
        """Test get_agent_id method."""
        assert architect_agent.get_agent_id().startswith("architect_")

    def test_get_capabilities(self, architect_agent: ArchitectAgent) -> None:
        """Test get_capabilities method."""
        capabilities = architect_agent.get_capabilities()
        assert isinstance(capabilities, list)
        assert "architecture" in capabilities
        assert "design" in capabilities
        assert "decomposition" in capabilities
        assert "system" in capabilities
        assert "high-level" in capabilities

    def test_can_handle(self, architect_agent: ArchitectAgent) -> None:
        """Test can_handle method."""
        # Should handle high-level tasks
        assert architect_agent.can_handle("Design a system architecture")
        assert architect_agent.can_handle("Break down this problem")
        assert architect_agent.can_handle("Create a high-level design")

        # Should not handle low-level tasks
        assert not architect_agent.can_handle("Implement this function")
        assert not architect_agent.can_handle("Fix this bug")
        assert not architect_agent.can_handle("Write a test case")

    def test_process(self, architect_agent: ArchitectAgent, mock_provider: MagicMock) -> None:
        """Test process method."""
        message = HumanMessage(content="Design a system")
        result = architect_agent.process(message)

        # Check that the provider was called
        mock_provider.generate.assert_called_once()

        # Check the result
        assert isinstance(result, Result)
        assert result.success is True
        assert result.data == "Mock response"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_stream(self, architect_agent: ArchitectAgent, mock_provider: MagicMock) -> None:
        """Test process_stream method."""
        message = HumanMessage(content="Design a system")

        # Use async list comprehension instead of for loop
        chunks = [chunk async for chunk in architect_agent.process_stream(message)]

        # Check that the provider was called
        mock_provider.generate_stream.assert_called_once()

        # Check the result
        assert chunks == ["Mock", " stream", " response"]

    def test_parent_child_relationship(self, architect_agent: ArchitectAgent) -> None:
        """Test parent-child relationship methods."""
        # Initially no parent or children
        assert architect_agent.get_parent_id() is None
        assert architect_agent.get_child_ids() == []

        # Set parent
        architect_agent.set_parent("parent_agent")
        assert architect_agent.get_parent_id() == "parent_agent"

        # Clear parent
        architect_agent.clear_parent()
        assert architect_agent.get_parent_id() is None

        # Add children
        architect_agent.add_child("child1")
        architect_agent.add_child("child2")
        assert set(architect_agent.get_child_ids()) == {"child1", "child2"}

        # Add duplicate child (should not add)
        architect_agent.add_child("child1")
        assert len(architect_agent.get_child_ids()) == 2

        # Remove child
        architect_agent.remove_child("child1")
        assert architect_agent.get_child_ids() == ["child2"]

        # Remove non-existent child (should not error)
        architect_agent.remove_child("non_existent")
        assert architect_agent.get_child_ids() == ["child2"]

    def test_delegate_to_child(self, architect_agent: ArchitectAgent) -> None:
        """Test delegate_to_child method."""
        # Add a child
        architect_agent.add_child("child1")

        # Delegate to existing child
        result = architect_agent.delegate_to_child("child1", "Do this task")
        assert result.success is True
        assert "delegated to child agent child1" in result.data

        # Delegate to non-existent child
        result = architect_agent.delegate_to_child("non_existent", "Do this task")
        assert result.success is False
        assert result.error == "Child agent not found: non_existent"

    def test_collect_results_from_children(self, architect_agent: ArchitectAgent) -> None:
        """Test collect_results_from_children method."""
        # Add children
        architect_agent.add_child("child1")
        architect_agent.add_child("child2")

        # Collect results
        results = architect_agent.collect_results_from_children()
        assert len(results) == 2
        assert "child1" in results
        assert "child2" in results
        assert results["child1"].success is True
        assert "Result from child agent child1" in results["child1"].data

    def test_validate_provider(self) -> None:
        """Test _validate_provider method."""
        # Create a new agent with no provider for testing
        agent = ArchitectAgent()

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provider not initialized"):
            # We need to call a public method that uses _validate_provider
            agent.process(HumanMessage(content="Test"))


class TestPlannerAgent:
    """Tests for the PlannerAgent class."""

    def test_initialization(self) -> None:
        """Test PlannerAgent initialization."""
        # Test with default parameters
        agent = PlannerAgent()
        assert agent.get_agent_id().startswith("planner_")
        assert agent.state is not None
        assert agent.get_parent_id() is None
        assert agent.get_child_ids() == []

        # Test with custom state
        custom_state = AgentState(agent_id="custom_agent")
        agent = PlannerAgent(state_manager=custom_state)
        assert agent.state == custom_state

        # Test with state manager
        state_manager = MagicMock()
        state_manager.get_state.return_value = AgentState(agent_id="managed_agent")
        agent = PlannerAgent(state_manager=state_manager)
        assert agent.state == state_manager.get_state.return_value

    def test_get_agent_id(self, planner_agent: PlannerAgent) -> None:
        """Test get_agent_id method."""
        assert planner_agent.get_agent_id().startswith("planner_")

    def test_get_capabilities(self, planner_agent: PlannerAgent) -> None:
        """Test get_capabilities method."""
        capabilities = planner_agent.get_capabilities()
        assert isinstance(capabilities, list)
        assert "planning" in capabilities
        assert "refinement" in capabilities
        assert "task-breakdown" in capabilities
        assert "mid-level" in capabilities
        assert "organization" in capabilities

    def test_can_handle(self, planner_agent: PlannerAgent) -> None:
        """Test can_handle method."""
        # Should handle mid-level tasks
        assert planner_agent.can_handle("Plan the implementation steps")
        assert planner_agent.can_handle("Refine this task")
        assert planner_agent.can_handle("Organize these requirements")

        # Should not handle high-level or low-level tasks
        assert not planner_agent.can_handle("Design a system architecture")
        assert not planner_agent.can_handle("Implement this function")
        assert not planner_agent.can_handle("Fix this bug")

    def test_process(self, planner_agent: PlannerAgent, mock_provider: MagicMock) -> None:
        """Test process method."""
        message = HumanMessage(content="Plan the implementation")
        result = planner_agent.process(message)

        # Check that the provider was called
        mock_provider.generate.assert_called_once()

        # Check the result
        assert isinstance(result, Result)
        assert result.success is True
        assert result.data == "Mock response"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_stream(self, planner_agent: PlannerAgent, mock_provider: MagicMock) -> None:
        """Test process_stream method."""
        message = HumanMessage(content="Plan the implementation")

        # Use async list comprehension instead of for loop
        chunks = [chunk async for chunk in planner_agent.process_stream(message)]

        # Check that the provider was called
        mock_provider.generate_stream.assert_called_once()

        # Check the result
        assert chunks == ["Mock", " stream", " response"]

    def test_parent_child_relationship(self, planner_agent: PlannerAgent) -> None:
        """Test parent-child relationship methods."""
        # Initially no parent or children
        assert planner_agent.get_parent_id() is None
        assert planner_agent.get_child_ids() == []

        # Set parent
        planner_agent.set_parent("architect_agent")
        assert planner_agent.get_parent_id() == "architect_agent"

        # Clear parent
        planner_agent.clear_parent()
        assert planner_agent.get_parent_id() is None

        # Add children
        planner_agent.add_child("executor1")
        planner_agent.add_child("executor2")
        assert set(planner_agent.get_child_ids()) == {"executor1", "executor2"}

        # Add duplicate child (should not add)
        planner_agent.add_child("executor1")
        assert len(planner_agent.get_child_ids()) == 2

        # Remove child
        planner_agent.remove_child("executor1")
        assert planner_agent.get_child_ids() == ["executor2"]

        # Remove non-existent child (should not error)
        planner_agent.remove_child("non_existent")
        assert planner_agent.get_child_ids() == ["executor2"]

    def test_delegate_to_child(self, planner_agent: PlannerAgent) -> None:
        """Test delegate_to_child method."""
        # Add a child
        planner_agent.add_child("executor1")

        # Delegate to existing child
        result = planner_agent.delegate_to_child("executor1", "Implement this function")
        assert result.success is True
        assert "delegated to child agent executor1" in result.data

        # Delegate to non-existent child
        result = planner_agent.delegate_to_child("non_existent", "Implement this function")
        assert result.success is False
        assert result.error == "Child agent not found: non_existent"

    def test_collect_results_from_children(self, planner_agent: PlannerAgent) -> None:
        """Test collect_results_from_children method."""
        # Add children
        planner_agent.add_child("executor1")
        planner_agent.add_child("executor2")

        # Collect results
        results = planner_agent.collect_results_from_children()
        assert len(results) == 2
        assert "executor1" in results
        assert "executor2" in results
        assert results["executor1"].success is True
        assert "Result from child agent executor1" in results["executor1"].data

    def test_validate_provider(self) -> None:
        """Test _validate_provider method."""
        # Create a new agent with no provider for testing
        agent = PlannerAgent()

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provider not initialized"):
            # We need to call a public method that uses _validate_provider
            agent.process(HumanMessage(content="Test"))
