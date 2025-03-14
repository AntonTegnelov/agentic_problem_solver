"""Unit tests for specialized agent types."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages.base import BaseMessage

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.executor import ExecutorAgent
from src.agent.agent_types.planner import PlannerAgent
from src.agent.state.base import AgentState
from src.common_types.message_types import HumanMessage
from src.common_types.result_types import Result
from src.common_types.task_types import TaskComplexity


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider."""
    provider = MagicMock()

    async def mock_generate(_messages: list[BaseMessage]) -> str:
        return "Test response"

    # Create a proper async generator for streaming
    async def mock_stream(_messages: list[BaseMessage]) -> AsyncGenerator[str, None]:
        chunks = ["Mock", " stream", " response"]
        for chunk in chunks:
            yield chunk

    # Set up the generate method
    provider.generate = AsyncMock(side_effect=mock_generate)

    # Instead of using AsyncMock with side_effect for the generator,
    # we'll return the generator function directly
    provider.generate_stream = mock_stream
    provider.__bool__.return_value = True
    return provider


@pytest.fixture
def architect_agent(mock_provider: MagicMock) -> ArchitectAgent:
    """Create an architect agent."""
    agent = ArchitectAgent(provider=mock_provider)
    agent.state.register_agent(agent.get_agent_id(), agent)
    return agent


@pytest.fixture
def planner_agent(mock_provider: MagicMock) -> PlannerAgent:
    """Create a planner agent."""
    agent = PlannerAgent(provider=mock_provider)
    agent.state.register_agent(agent.get_agent_id(), agent)
    return agent


@pytest.fixture
def executor_agent(mock_provider: MagicMock) -> ExecutorAgent:
    """Create an executor agent."""
    agent = ExecutorAgent(provider=mock_provider)
    agent.state.register_agent(agent.get_agent_id(), agent)
    return agent


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

    @pytest.mark.asyncio
    async def test_process(self, architect_agent: ArchitectAgent, mock_provider: MagicMock) -> None:
        """Test process method."""
        message = HumanMessage(content="Design a system")
        result = await architect_agent.process(message)

        # Check that the provider was called
        mock_provider.generate.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_stream(self, architect_agent: ArchitectAgent) -> None:
        """Test process_stream method."""
        message = HumanMessage(content="Design a system")

        # Use async list comprehension instead of for loop
        chunks = [chunk async for chunk in architect_agent.process_stream(message)]

        # Since we're now using a direct function for generate_stream, we can't check .called
        # Instead, check that the result matches what we expect from our mock
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

    @pytest.mark.asyncio
    async def test_delegate_to_child(self, architect_agent: ArchitectAgent) -> None:
        """Test delegate_to_child method."""
        # Add a child
        architect_agent.add_child("child1")

        # Delegate to existing child
        result = await architect_agent.delegate_to_child("child1", "Do this task")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_collect_results_from_children(self, architect_agent: ArchitectAgent) -> None:
        """Test collect_results_from_children method."""
        # Add children
        architect_agent.add_child("child1")
        architect_agent.add_child("child2")

        # Collect results
        results = await architect_agent.collect_results_from_children()
        assert len(results) == 2

    def test_analyze_task_complexity(self, architect_agent: ArchitectAgent) -> None:
        """Test analyze_task_complexity method."""
        # Test that the method returns a valid TaskComplexity enum value
        simple_task = "Create a simple function to add two numbers."
        simple_result = architect_agent.analyze_task_complexity(simple_task)
        assert isinstance(simple_result, TaskComplexity)

        # Test the rule-based approach directly
        simple_task = "Create a simple function to add two numbers."
        complex_task = (
            "Create a module with multiple files for user management. "
            "This component should include several classes and interfaces."
        )
        very_complex_task = (
            "Create a full architecture for a distributed microservices system with highly complex, "
            "scalable components. This enterprise-level system requires complete redesign of the "
            "entire system with microservices for authentication, authorization, data processing, "
            "and user management."
        )

        # Test rule-based approach directly
        simple_complexity = architect_agent._analyze_task_complexity_rule_based(simple_task)
        complex_complexity = architect_agent._analyze_task_complexity_rule_based(complex_task)
        very_complex_complexity = architect_agent._analyze_task_complexity_rule_based(very_complex_task)

        # Test that complexity increases with task complexity
        # We don't assert exact values, just the relative ordering
        complexity_values = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2,
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.VERY_COMPLEX: 4,
        }

        assert complexity_values[simple_complexity] <= complexity_values[complex_complexity]
        assert complexity_values[complex_complexity] <= complexity_values[very_complex_complexity]

    def test_analyze_task_complexity_with_llm(self, architect_agent: ArchitectAgent, mock_provider: MagicMock) -> None:
        """Test analyze_task_complexity method with LLM provider."""

        # Create a mock for the _get_llm_response method
        async def mock_get_llm_response_simple(*args, **kwargs) -> str:
            return "simple"

        async def mock_get_llm_response_moderate(*args, **kwargs) -> str:
            return "moderate"

        async def mock_get_llm_response_complex(*args, **kwargs) -> str:
            return "complex"

        async def mock_get_llm_response_very_complex(*args, **kwargs) -> str:
            return "very complex"

        async def mock_get_llm_response_invalid(*args, **kwargs) -> str:
            return "Invalid response"

        # Test simple task
        with patch.object(architect_agent, "_get_llm_response", side_effect=mock_get_llm_response_simple):
            simple_task = "Create a simple function to add two numbers."
            simple_result = architect_agent._analyze_task_complexity_with_llm(simple_task)
            assert simple_result == TaskComplexity.SIMPLE

        # Test moderate task
        with patch.object(architect_agent, "_get_llm_response", side_effect=mock_get_llm_response_moderate):
            moderate_task = "Create a module with a few classes."
            moderate_result = architect_agent._analyze_task_complexity_with_llm(moderate_task)
            assert moderate_result == TaskComplexity.MODERATE

        # Test complex task
        with patch.object(architect_agent, "_get_llm_response", side_effect=mock_get_llm_response_complex):
            complex_task = "Design a system for user authentication."
            complex_result = architect_agent._analyze_task_complexity_with_llm(complex_task)
            assert complex_result == TaskComplexity.COMPLEX

        # Test very complex task
        with patch.object(architect_agent, "_get_llm_response", side_effect=mock_get_llm_response_very_complex):
            very_complex_task = "Create a distributed microservices architecture."
            very_complex_result = architect_agent._analyze_task_complexity_with_llm(very_complex_task)
            assert very_complex_result == TaskComplexity.VERY_COMPLEX

        # Test invalid response (should default to MODERATE)
        with patch.object(architect_agent, "_get_llm_response", side_effect=mock_get_llm_response_invalid):
            invalid_task = "This will return an invalid response."
            invalid_result = architect_agent._analyze_task_complexity_with_llm(invalid_task)
            assert invalid_result == TaskComplexity.MODERATE

    def test_analyze_task_complexity_fallback(self, architect_agent: ArchitectAgent, mock_provider: MagicMock) -> None:
        """Test analyze_task_complexity method with fallback to rule-based approach."""
        # Mock the _analyze_task_complexity_with_llm method to raise an exception
        with patch.object(architect_agent, "_analyze_task_complexity_with_llm", side_effect=ValueError("Test error")):
            task = "Design a system for user authentication."
            complexity = architect_agent.analyze_task_complexity(task)
            assert complexity in [
                TaskComplexity.SIMPLE,
                TaskComplexity.MODERATE,
                TaskComplexity.COMPLEX,
                TaskComplexity.VERY_COMPLEX,
            ]

    def test_validate_provider(self, architect_agent: ArchitectAgent) -> None:
        """Test validate_provider method."""
        # Set provider to None
        architect_agent._provider = None
        with pytest.raises(ValueError):
            architect_agent._validate_provider()

    @pytest.mark.asyncio
    async def test_delegate_to_executor(self, architect_agent: ArchitectAgent, mock_provider: MagicMock) -> None:
        """Test delegate_to_executor method."""
        # Mock the ExecutorAgent.process method
        with patch("src.agent.agent_types.executor.ExecutorAgent.process") as mock_process:
            # Set up the mock to return a successful result
            mock_result = Result(success=True, data="Execution result", error=None)
            mock_process.return_value = mock_result

            # Call the delegate_to_executor method
            task = "Implement a simple function to add two numbers."
            result = await architect_agent.delegate_to_executor(task)

            # Verify the result
            assert result.success is True
            assert result.data == "Execution result"
            assert result.error is None

            # Verify that the ExecutorAgent was created and process was called
            mock_process.assert_called_once()

            # Verify that the parent-child relationship was established
            assert len(architect_agent.get_child_ids()) > 0


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

    @pytest.mark.asyncio
    async def test_process(self, planner_agent: PlannerAgent, mock_provider: MagicMock) -> None:
        """Test process method."""
        message = HumanMessage(content="Plan the implementation")
        result = await planner_agent.process(message)

        # Check that the provider was called
        mock_provider.generate.assert_called_once()

        # Check the result
        assert isinstance(result, Result)
        assert result.success is True
        assert isinstance(result.data, str)
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_stream(self, planner_agent: PlannerAgent) -> None:
        """Test process_stream method."""
        message = HumanMessage(content="Plan the implementation")

        # Use async list comprehension instead of for loop
        chunks = [chunk async for chunk in planner_agent.process_stream(message)]

        # Since we're now using a direct function for generate_stream, we can't check .called
        # Instead, check that the result matches what we expect from our mock
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

    @pytest.mark.asyncio
    async def test_delegate_to_child(self, planner_agent: PlannerAgent) -> None:
        """Test delegate_to_child method."""
        # Add a child
        planner_agent.add_child("executor1")

        # Delegate to existing child
        result = await planner_agent.delegate_to_child("executor1", "Implement this function")
        assert result.success is True
        assert "delegated to child agent executor1" in result.data

        # Delegate to non-existent child
        result = await planner_agent.delegate_to_child("non_existent", "Implement this function")
        assert result.success is False
        assert "Child agent not found: non_existent" in result.error

    @pytest.mark.asyncio
    async def test_collect_results_from_children(self, planner_agent: PlannerAgent) -> None:
        """Test collect_results_from_children method."""
        # Add children
        planner_agent.add_child("executor1")
        planner_agent.add_child("executor2")

        # Collect results
        results = await planner_agent.collect_results_from_children()
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
            agent._validate_provider()  # noqa: SLF001


class TestExecutorAgent:
    """Tests for the ExecutorAgent class."""

    def test_initialization(self) -> None:
        """Test ExecutorAgent initialization."""
        # Test with default parameters
        agent = ExecutorAgent()
        assert agent.get_agent_id().startswith("executor_")
        assert agent.state is not None
        assert agent.get_parent_id() is None
        assert agent.get_child_ids() == []

        # Test with custom state
        custom_state = AgentState(agent_id="custom_agent")
        agent = ExecutorAgent(state_manager=custom_state)
        assert agent.state == custom_state

        # Test with state manager
        state_manager = MagicMock()
        state_manager.get_state.return_value = AgentState(agent_id="managed_agent")
        agent = ExecutorAgent(state_manager=state_manager)
        assert agent.state == state_manager.get_state.return_value

    def test_get_agent_id(self, executor_agent: ExecutorAgent) -> None:
        """Test get_agent_id method."""
        assert executor_agent.get_agent_id().startswith("executor_")

    def test_get_capabilities(self, executor_agent: ExecutorAgent) -> None:
        """Test get_capabilities method."""
        capabilities = executor_agent.get_capabilities()
        assert isinstance(capabilities, list)
        assert "execution" in capabilities
        assert "implementation" in capabilities
        assert "coding" in capabilities
        assert "low-level" in capabilities
        assert "detail-oriented" in capabilities

    def test_can_handle(self, executor_agent: ExecutorAgent) -> None:
        """Test can_handle method."""
        # Should handle low-level tasks
        assert executor_agent.can_handle("Implement this function")
        assert executor_agent.can_handle("Write code for this feature")
        assert executor_agent.can_handle("Develop a low-level component")

        # Should not handle high-level or mid-level tasks
        assert not executor_agent.can_handle("Design a system architecture")
        assert not executor_agent.can_handle("Plan the implementation steps")
        assert not executor_agent.can_handle("Create a high-level design")

    @pytest.mark.asyncio
    async def test_process(self, executor_agent: ExecutorAgent, mock_provider: MagicMock) -> None:
        """Test process method."""
        message = HumanMessage(content="Implement this function")
        result = await executor_agent.process(message)

        # Check that the provider was called
        mock_provider.generate.assert_called_once()

        # Check the result
        assert isinstance(result, Result)
        assert result.success is True
        assert isinstance(result.data, str)
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_stream(self, executor_agent: ExecutorAgent) -> None:
        """Test process_stream method."""
        message = HumanMessage(content="Implement this function")

        # Use async list comprehension instead of for loop
        chunks = [chunk async for chunk in executor_agent.process_stream(message)]

        # Since we're now using a direct function for generate_stream, we can't check .called
        # Instead, check that the result matches what we expect from our mock
        assert chunks == ["Mock", " stream", " response"]

    def test_parent_child_relationship(self, executor_agent: ExecutorAgent) -> None:
        """Test parent-child relationship methods."""
        # Initially no parent or children
        assert executor_agent.get_parent_id() is None
        assert executor_agent.get_child_ids() == []

        # Set parent
        executor_agent.set_parent("parent_agent")
        assert executor_agent.get_parent_id() == "parent_agent"

        # Clear parent
        executor_agent.clear_parent()
        assert executor_agent.get_parent_id() is None

        # Add children
        executor_agent.add_child("child1")
        executor_agent.add_child("child2")
        assert set(executor_agent.get_child_ids()) == {"child1", "child2"}

        # Add duplicate child (should not add)
        executor_agent.add_child("child1")
        assert len(executor_agent.get_child_ids()) == 2

        # Remove child
        executor_agent.remove_child("child1")
        assert executor_agent.get_child_ids() == ["child2"]

        # Remove non-existent child (should not error)
        executor_agent.remove_child("non_existent")
        assert executor_agent.get_child_ids() == ["child2"]

    @pytest.mark.asyncio
    async def test_delegate_to_child(self, executor_agent: ExecutorAgent) -> None:
        """Test delegate_to_child method."""
        # ExecutorAgent is a leaf node, so delegation should return an error
        result = await executor_agent.delegate_to_child("child1", "Implement this function")
        assert result.success is False
        assert "no child agents" in result.error.lower()

    @pytest.mark.asyncio
    async def test_collect_results_from_children(self, executor_agent: ExecutorAgent) -> None:
        """Test collect_results_from_children method."""
        # ExecutorAgent is a leaf node, so should return empty results
        results = await executor_agent.collect_results_from_children()
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_validate_provider(self) -> None:
        """Test _validate_provider method."""
        # Create a new agent with no provider for testing
        agent = ExecutorAgent()

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provider not initialized"):
            agent._validate_provider()  # noqa: SLF001
