"""Unit tests for specialized agent types."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from langchain_core.messages.base import BaseMessage

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.executor import ExecutorAgent
from src.agent.agent_types.planner import PlannerAgent
from src.agent.state.base import AgentState
from src.common_types.enums import ExecutionStage, VerificationStatus
from src.common_types.message_types import HumanMessage
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity, TaskStatus


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

    def test_analyze_task_complexity_with_llm(self, architect_agent: ArchitectAgent) -> None:
        """Test analyze_task_complexity method with LLM provider."""

        # Create a mock for the _get_llm_response method
        async def mock_get_llm_response_simple(*_: object) -> str:
            return "simple"

        async def mock_get_llm_response_moderate(*_: object) -> str:
            return "moderate"

        async def mock_get_llm_response_complex(*_: object) -> str:
            return "complex"

        async def mock_get_llm_response_very_complex(*_: object) -> str:
            return "very complex"

        async def mock_get_llm_response_invalid(*_: object) -> str:
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

    def test_analyze_task_complexity_fallback(self, architect_agent: ArchitectAgent) -> None:
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
        with pytest.raises(ValueError, match="Provider not initialized"):
            architect_agent._validate_provider()

    @pytest.mark.asyncio
    async def test_delegate_to_executor(self, architect_agent: ArchitectAgent) -> None:
        """Test delegate_to_executor method."""
        # Mock the analyze_task_complexity method to return SIMPLE
        with patch.object(
            architect_agent,
            "analyze_task_complexity",
            return_value=TaskComplexity.SIMPLE,
        ):
            # Call the delegate_to_executor method
            task = "Implement a simple function to add two numbers."
            result = await architect_agent.delegate_to_executor(task)

            # Verify the result
            assert result.success is True
            # The method now returns the executor's result directly
            assert result.data == "Test response"


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
        child_id = "executor1"
        planner_agent.add_child(child_id)

        # Create a mock for the child agent
        mock_child_agent = MagicMock()
        mock_child_agent.process = AsyncMock(return_value=Result.success("Task completed by executor1"))

        # Register the mock child agent in the state
        planner_agent.state.register_agent(child_id, mock_child_agent)

        # Delegate to existing child
        result = await planner_agent.delegate_to_child(child_id, "Implement this function")
        assert result.success is True
        assert "Task completed by executor1" in result.data

    @pytest.mark.asyncio
    async def test_collect_results_from_children(self, planner_agent: PlannerAgent) -> None:
        """Test collecting results from children."""
        # Add some child agents
        planner_agent.add_child("child1")
        planner_agent.add_child("child2")

        # Collect results
        results = await planner_agent.collect_results_from_children()

        # Verify results
        assert len(results) == 2
        assert "child1" in results
        assert "child2" in results
        assert results["child1"].success
        assert results["child2"].success
        assert "Result from child agent child1" in results["child1"].data
        assert "Result from child agent child2" in results["child2"].data

    @pytest.mark.asyncio
    async def test_delegate_to_planner(self, planner_agent: PlannerAgent) -> None:
        """Test delegating to another planner agent for complex sub-components."""
        # Mock the evaluate_subtask_complexity method to return COMPLEX
        with patch.object(
            planner_agent,
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.COMPLEX,
        ):
            # Test delegation
            result = await planner_agent.delegate_to_planner("Complex sub-component task")

            # Verify the result
            assert result.success is True
            assert "Task delegated to sub-planner" in result.data

    @pytest.mark.asyncio
    async def test_process_tasks_parallel(self, planner_agent: PlannerAgent) -> None:
        """Test processing tasks in parallel."""
        # Create test tasks
        tasks = [
            Task(description="Task 1: Implement login functionality"),
            Task(description="Task 2: Create user profile page"),
            Task(description="Task 3: Add password reset feature"),
        ]

        # Mock the _delegate_single_task method to return success
        with patch.object(
            planner_agent,
            "_delegate_single_task",
            new_callable=AsyncMock,
            return_value=("Task delegated successfully", False, ""),
        ):
            # Test parallel processing
            result = await planner_agent.delegate_tasks_parallel(tasks)

            # Verify the result
            assert result.success is True
            # Task data should be in the result
            assert "Task delegated successfully" in str(result.data)

    def test_validate_provider(self) -> None:
        """Test provider validation."""
        # Create a new agent with no provider for testing
        agent = PlannerAgent()

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provider not initialized"):
            agent._validate_provider()


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
        assert "cannot delegate to child agents" in result.error.lower()

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
            agent._validate_provider()

    @pytest.mark.asyncio
    async def test_iterate_task(self, executor_agent: ExecutorAgent) -> None:
        """Test the iterate_task method."""
        # Create a task

        from src.common_types.enums import ExecutionStage
        from src.common_types.message_types import HumanMessage
        from src.common_types.result_types import Result
        from src.common_types.task_types import Task, TaskStatus

        task = Task(description="Test task")

        # Mock the create_message function and process method
        with (
            patch("src.agent.agent_types.executor.create_message") as mock_create_message,
            patch.object(executor_agent, "process") as mock_process,
        ):
            # Set up the mock to return a HumanMessage
            mock_create_message.return_value = HumanMessage(content="Test task execution")

            # Set up the mock process to return a successful response
            mock_process.return_value = Result(success=True, data="Task execution result", error=None)

            # Call the iterate_task method
            result = await executor_agent.iterate_task(task)

            # Verify the result
            assert result.success is True
            assert isinstance(result.data, Task)

            # Verify task was updated correctly
            updated_task = result.data
            assert updated_task.execution_attempts == 1
            assert updated_task.created_at is not None
            assert updated_task.updated_at is not None
            assert updated_task.status == TaskStatus.IN_PROGRESS
            assert updated_task.execution_stage == ExecutionStage.IMPLEMENTING  # Should have advanced from PLANNING
            assert updated_task.result == "Task execution result"
            assert "planning_result" in updated_task.execution_metadata

            # Test a second iteration
            mock_process.return_value = Result(success=True, data="Second task execution result", error=None)
            second_result = await executor_agent.iterate_task(updated_task)
            second_task = second_result.data

            assert second_task.execution_attempts == 2
            assert second_task.execution_stage == ExecutionStage.TESTING  # Should have advanced from IMPLEMENTING
            assert "implementation_result" in second_task.execution_metadata

            # Test failure case
            mock_process.return_value = Result(success=False, data=None, error="Test error")

            failure_result = await executor_agent.iterate_task(task)
            assert failure_result.success is False
            assert isinstance(failure_result.data, Task)
            assert "Test error" in str(failure_result.error)

            # Test completion case
            complete_task = Task(description="Complete task")
            complete_task.execution_stage = ExecutionStage.FINALIZING
            complete_task.verification_status = VerificationStatus.PASSED

            mock_process.return_value = Result(success=True, data="Final result", error=None)

            completion_result = await executor_agent.iterate_task(complete_task)
            completed_task = completion_result.data

            assert completed_task.status == TaskStatus.COMPLETED
            assert completed_task.completed_at is not None
            assert "final_result" in completed_task.execution_metadata

    def test_evaluate_completion_criteria_basic_cases(self, executor_agent: ExecutorAgent) -> None:
        """Test the _evaluate_completion_criteria method for basic cases."""
        from src.common_types.enums import ExecutionStage
        from src.common_types.task_types import Task

        # Test case 1: Task not in final stage
        task1 = Task(description="Test task")
        task1.execution_stage = ExecutionStage.IMPLEMENTING
        is_complete, message = executor_agent._evaluate_completion_criteria(task1)
        assert is_complete is False
        assert "not in final stage" in message

        # Test case 2: Task in final stage but verification not passed
        task2 = Task(description="Test task")
        task2.execution_stage = ExecutionStage.FINALIZING
        task2.verification_status = VerificationStatus.FAILED
        is_complete, message = executor_agent._evaluate_completion_criteria(task2)
        assert is_complete is False
        assert "Verification not passed" in message

        # Test case 3: Task in final stage with verification passed but no result
        task3 = Task(description="Test task")
        task3.execution_stage = ExecutionStage.FINALIZING
        task3.verification_status = VerificationStatus.PASSED
        # Mock the _extract_required_outputs method to return a non-empty list
        # This ensures the test doesn't skip the result check due to backward compatibility
        with patch.object(executor_agent, "_extract_required_outputs", return_value=["required output"]):
            is_complete, message = executor_agent._evaluate_completion_criteria(task3)
            assert is_complete is False
            assert "Task has no result" in message

    def test_evaluate_completion_criteria_output_checks(self, executor_agent: ExecutorAgent) -> None:
        """Test the _evaluate_completion_criteria method for output-related checks."""
        from src.common_types.enums import ExecutionStage
        from src.common_types.task_types import Task

        # Test case 4: Task with missing required outputs
        task4 = Task(description="Test task")
        task4.execution_stage = ExecutionStage.FINALIZING
        task4.verification_status = VerificationStatus.PASSED
        task4.result = "This is a result without the required output"
        task4.execution_logs = ["Log entry"]
        task4.execution_attempts = 1
        # Mock the _check_required_outputs method to return a list of missing outputs
        with patch.object(executor_agent, "_check_required_outputs", return_value=["required output"]):
            is_complete, message = executor_agent._evaluate_completion_criteria(task4)
            assert is_complete is False
            assert "Missing required outputs" in message

        # Test case 5: Task with error in result
        task5 = Task(description="Test task")
        task5.execution_stage = ExecutionStage.FINALIZING
        task5.verification_status = VerificationStatus.PASSED
        task5.result = "This result has an error in it"
        task5.execution_logs = ["Log entry"]
        task5.execution_attempts = 1
        # Mock the _check_required_outputs method to return an empty list (no missing outputs)
        # and _check_for_errors to return an error context
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value="error context"),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task5)
            assert is_complete is False
            assert "Error detected in result" in message

    def test_evaluate_completion_criteria_metadata_checks(self, executor_agent: ExecutorAgent) -> None:
        """Test the _evaluate_completion_criteria method for metadata-related checks."""
        from src.common_types.enums import ExecutionStage
        from src.common_types.task_types import Task

        # Test case 6: Task with missing execution metadata
        task6 = Task(description="Test task")
        task6.execution_stage = ExecutionStage.FINALIZING
        task6.verification_status = VerificationStatus.PASSED
        task6.result = "This is a valid result"
        task6.execution_logs = ["Log entry"]
        task6.execution_attempts = 1
        # Mock the check methods to return appropriate values
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=["planning_result"]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task6)
            assert is_complete is False
            assert "Missing execution metadata" in message

        # Test case 7: Task with incomplete subtasks
        task7 = Task(description="Test task")
        task7.execution_stage = ExecutionStage.FINALIZING
        task7.verification_status = VerificationStatus.PASSED
        task7.result = "This is a valid result"
        task7.execution_logs = ["Log entry"]
        task7.execution_attempts = 1
        # Mock the check methods to return appropriate values
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=["subtask-1"]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task7)
            assert is_complete is False
            assert "Incomplete subtasks" in message

    def test_evaluate_completion_criteria_execution_checks(self, executor_agent: ExecutorAgent) -> None:
        """Test the _evaluate_completion_criteria method for execution-related checks."""
        from src.common_types.enums import ExecutionStage
        from src.common_types.task_types import Task

        # Test case 8: Task with no execution logs
        task8 = Task(description="Test task")
        task8.execution_stage = ExecutionStage.FINALIZING
        task8.verification_status = VerificationStatus.PASSED
        task8.result = "This is a valid result"
        task8.execution_logs = []  # Empty execution logs
        task8.execution_attempts = 1
        # Mock the check methods to return empty results
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=[]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task8)
            assert is_complete is False
            assert "No execution logs recorded" in message

        # Test case 9: Task with no execution attempts
        task9 = Task(description="Test task")
        task9.execution_stage = ExecutionStage.FINALIZING
        task9.verification_status = VerificationStatus.PASSED
        task9.result = "This is a valid result"
        task9.execution_logs = ["Log entry"]
        task9.execution_attempts = 0  # No execution attempts
        # Mock the check methods to return empty results
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=[]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task9)
            assert is_complete is False
            assert "Task has not been attempted" in message

    def test_evaluate_completion_criteria_status_checks(self, executor_agent: ExecutorAgent) -> None:
        """Test the _evaluate_completion_criteria method for status-related checks."""
        from src.common_types.enums import ExecutionStage
        from src.common_types.task_types import Task, TaskStatus

        # Test case 10: Task marked as failed
        task10 = Task(description="Test task")
        task10.execution_stage = ExecutionStage.FINALIZING
        task10.verification_status = VerificationStatus.PASSED
        task10.result = "This is a valid result"
        task10.execution_logs = ["Log entry"]
        task10.execution_attempts = 1
        task10.status = TaskStatus.FAILED
        # Mock the check methods to return empty results
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=[]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task10)
            assert is_complete is False
            assert "Task is marked as failed" in message

        # Test case 11: Task marked as blocked
        task11 = Task(description="Test task")
        task11.execution_stage = ExecutionStage.FINALIZING
        task11.verification_status = VerificationStatus.PASSED
        task11.result = "This is a valid result"
        task11.execution_logs = ["Log entry"]
        task11.execution_attempts = 1
        task11.status = TaskStatus.BLOCKED
        # Mock the check methods to return empty results
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=[]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task11)
            assert is_complete is False
            assert "Task is blocked" in message

        # Test case 12: Task with result that's too short
        task12 = Task(description="Test task")
        task12.execution_stage = ExecutionStage.FINALIZING
        task12.verification_status = VerificationStatus.PASSED
        task12.result = "Short"
        task12.execution_logs = ["Log entry"]
        task12.execution_attempts = 1
        task12.status = TaskStatus.IN_PROGRESS
        # Mock the check methods to return empty results
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=[]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task12)
            assert is_complete is False
            assert "Task result is too short" in message

        # Test case 13: Task that meets all criteria
        task13 = Task(description="Test task")
        task13.execution_stage = ExecutionStage.FINALIZING
        task13.verification_status = VerificationStatus.PASSED
        task13.result = "This is a complete and valid result that is long enough"
        task13.execution_logs = ["Log entry"]
        task13.execution_attempts = 1
        task13.status = TaskStatus.IN_PROGRESS
        # Mock all the check methods to return empty results
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=[]),
        ):
            is_complete, message = executor_agent._evaluate_completion_criteria(task13)
            assert is_complete is True
            assert "Task meets all completion criteria" in message

    def test_check_basic_requirements(self, executor_agent: ExecutorAgent) -> None:
        """Test the _check_basic_requirements method."""
        from src.common_types.enums import ExecutionStage
        from src.common_types.task_types import Task

        # Test with task not in FINALIZING stage
        task = Task(description="Test task")
        task.execution_stage = ExecutionStage.IMPLEMENTING
        assert executor_agent._check_basic_requirements(task) is False

        # Test with task in FINALIZING stage
        task.execution_stage = ExecutionStage.FINALIZING
        assert executor_agent._check_basic_requirements(task) is True

    def test_check_required_outputs(self, executor_agent: ExecutorAgent) -> None:
        """Test the _check_required_outputs method."""
        from src.common_types.task_types import Task

        # Mock _extract_required_outputs to return a list of required outputs
        with patch.object(executor_agent, "_extract_required_outputs", return_value=["output1", "output2"]):
            # Test with task result containing all required outputs
            task = Task(description="Test task")
            task.result = "This result contains output1 and output2"
            missing_outputs = executor_agent._check_required_outputs(task)
            assert missing_outputs == []

            # Test with task result missing one required output
            task.result = "This result contains only output1"
            missing_outputs = executor_agent._check_required_outputs(task)
            assert missing_outputs == ["output2"]

            # Test with task result missing all required outputs
            task.result = "This result contains no required outputs"
            missing_outputs = executor_agent._check_required_outputs(task)
            assert missing_outputs == ["output1", "output2"]

        # Test with no required outputs
        with patch.object(executor_agent, "_extract_required_outputs", return_value=[]):
            task = Task(description="Test task")
            task.result = "Any result"
            missing_outputs = executor_agent._check_required_outputs(task)
            assert missing_outputs == []

    def test_check_for_errors(self, executor_agent: ExecutorAgent) -> None:
        """Test the _check_for_errors method."""
        from src.common_types.task_types import Task

        # Mock _get_error_context and _is_actual_error
        with (
            patch.object(executor_agent, "_get_error_context", return_value="error context"),
            patch.object(executor_agent, "_is_actual_error", return_value=True),
        ):
            # Test with result containing an error indicator
            task = Task(description="Test task")
            task.result = "This result contains an error"
            error_context = executor_agent._check_for_errors(task)
            assert error_context == "error context"

        # Test with result not containing any error indicator
        with (
            patch.object(executor_agent, "_get_error_context", return_value="error context"),
            patch.object(executor_agent, "_is_actual_error", return_value=False),
        ):
            task = Task(description="Test task")
            task.result = "This result is fine"
            error_context = executor_agent._check_for_errors(task)
            assert error_context == ""

    def test_check_execution_metadata(self, executor_agent: ExecutorAgent) -> None:
        """Test the _check_execution_metadata method."""
        from src.common_types.task_types import Task

        # Test with no execution metadata
        task = Task(description="Test task")
        task.execution_metadata = {}
        missing_metadata = executor_agent._check_execution_metadata(task)
        assert missing_metadata == []

        # Test with complete execution metadata
        task.execution_metadata = {
            "planning_result": "Planning result",
            "implementation_result": "Implementation result",
            "testing_result": "Testing result",
            "refined_implementation": "Refined implementation",
            "final_result": "Final result",
        }
        missing_metadata = executor_agent._check_execution_metadata(task)
        assert missing_metadata == []

        # Test with missing execution metadata
        task.execution_metadata = {
            "planning_result": "Planning result",
            "implementation_result": "",  # Empty value
            "testing_result": "Testing result",
            # Missing refined_implementation
            "final_result": "Final result",
        }
        missing_metadata = executor_agent._check_execution_metadata(task)
        assert set(missing_metadata) == {"implementation_result", "refined_implementation"}

    def test_check_subtasks(self, executor_agent: ExecutorAgent) -> None:
        """Test the _check_subtasks method."""
        from src.common_types.task_types import Task

        # Create a task with subtasks
        task = Task(description="Test task")
        subtask_id1 = UUID("00000000-0000-0000-0000-000000000001")
        subtask_id2 = UUID("00000000-0000-0000-0000-000000000002")
        task.subtasks = [subtask_id1, subtask_id2]

        # Mock the state manager's get_task_by_id method
        def mock_get_task_by_id(task_id: UUID) -> Task | None:
            if task_id == subtask_id1:
                subtask1 = Task(description="Subtask 1")
                subtask1.status = "completed"
                return subtask1
            if task_id == subtask_id2:
                subtask2 = Task(description="Subtask 2")
                subtask2.status = "in_progress"
                return subtask2
            return None

        # Apply the mock
        executor_agent.state_manager.get_task_by_id = mock_get_task_by_id

        # Test the method
        incomplete_subtasks = executor_agent._check_subtasks(task)
        assert len(incomplete_subtasks) == 1
        assert str(subtask_id2) in incomplete_subtasks[0]

    def test_check_result_quality(self, executor_agent: ExecutorAgent) -> None:
        """Test the _check_result_quality method."""
        from src.common_types.task_types import Task

        # Test case 1: Task with no result
        task1 = Task(description="Test task")
        task1.result = None
        result1 = executor_agent._check_result_quality(task1)
        assert "No result to evaluate" in result1

        # Test case 2: Task with placeholder in result
        task2 = Task(description="Test task")
        task2.result = "This is a result with a TODO item"
        result2 = executor_agent._check_result_quality(task2)
        assert "Result contains placeholder" in result2

        # Test case 3: Task with complete code block
        task3 = Task(description="Test task")
        task3.result = (
            "Here is a function:\n```python\ndef calculate_sum(a, b):\n    # Some code\n    return a + b\n```"
        )
        result3 = executor_agent._check_result_quality(task3)
        assert result3 == ""  # No quality issues

        # Test case 4: Task with incomplete code block (missing end)
        task4 = Task(description="Test task")
        task4.result = "Here is a function:\n```python\ndef calculate_sum(a, b):\n    # Some code"
        result4 = executor_agent._check_result_quality(task4)
        assert "Result contains incomplete code block" in result4

        # Test case 5: Task with missing key terms
        task5 = Task(description="Implement a user authentication system with password hashing")
        task5.result = "Here is a simple login function"
        result5 = executor_agent._check_result_quality(task5)
        assert "Result doesn't address key terms" in result5

        # Test case 6: Task with error indicator
        task6 = Task(description="Test task")
        task6.result = "Implementation failed: could not connect to database"
        # Mock the _is_actual_error method to return True
        with patch.object(executor_agent, "_is_actual_error", return_value=True):
            result6 = executor_agent._check_result_quality(task6)
            assert "Result contains error indicator" in result6

        # Test case 7: Task with missing expected output
        task7 = Task(description="Test task")
        task7.result = "This is a complete result"
        task7.execution_metadata = {"expected_outputs": ["user interface", "database schema"]}
        result7 = executor_agent._check_result_quality(task7)
        assert "Result missing expected output" in result7

        # Test case 8: Task with good quality result
        task8 = Task(description="Implement a simple calculator function")
        task8.result = """
        def calculator(a, b, operation):
            if operation == 'add':
                return a + b
            elif operation == 'subtract':
                return a - b
            elif operation == 'multiply':
                return a * b
            elif operation == 'divide':
                if b == 0:
                    raise ValueError("Cannot divide by zero")
                return a / b
            else:
                raise ValueError("Unknown operation")
        """
        result8 = executor_agent._check_result_quality(task8)
        assert result8 == ""  # No quality issues

    def test_extract_key_terms(self, executor_agent: ExecutorAgent) -> None:
        """Test the _extract_key_terms method."""
        # Test with a simple description
        terms1 = executor_agent._extract_key_terms("Implement a user authentication system")
        assert "user" in terms1
        assert "authentication" in terms1
        assert "system" in terms1

        # Test with a more complex description
        terms2 = executor_agent._extract_key_terms(
            "Create a database schema for storing customer information including name, address, and purchase history",
        )
        assert "database" in terms2
        assert "schema" in terms2
        assert "customer" in terms2
        assert "information" in terms2
        assert "name" in terms2
        assert "address" in terms2
        assert "purchase" in terms2
        assert "history" in terms2

        # Test with common words that should be filtered out
        terms3 = executor_agent._extract_key_terms(
            "The function should be implemented with proper error handling",
        )
        assert "the" not in terms3
        assert "should" not in terms3
        assert "be" not in terms3
        assert "with" not in terms3
        assert "proper" in terms3
        assert "error" in terms3
        assert "handling" in terms3

    def test_extract_required_outputs(self, executor_agent: ExecutorAgent) -> None:
        """Test the _extract_required_outputs method."""
        # Test with no required outputs
        outputs1 = executor_agent._extract_required_outputs("Implement a function")
        assert not outputs1  # Should be empty

        # Test with explicit required outputs
        outputs2 = executor_agent._extract_required_outputs(
            "Implement a function that returns: 1) A user object, 2) An authentication token",
        )
        assert len(outputs2) == 2
        assert "user object" in outputs2
        assert "authentication token" in outputs2

        # Test with outputs in different format
        outputs3 = executor_agent._extract_required_outputs(
            "Create a module with the following outputs:\n- User interface\n- API endpoints\n- Database schema",
        )
        assert len(outputs3) == 3
        assert "user interface" in outputs3
        assert "api endpoints" in outputs3
        assert "database schema" in outputs3

    def test_evaluate_task_completion(self, executor_agent: ExecutorAgent) -> None:
        """Test the evaluate_task_completion method."""
        from src.common_types.enums import ExecutionStage
        from src.common_types.task_types import Task, TaskStatus

        # Create a task that meets all completion criteria
        task = Task(description="Test task")
        task.execution_stage = ExecutionStage.FINALIZING
        task.verification_status = VerificationStatus.PASSED
        task.result = "This is a complete and valid result that is long enough"
        task.execution_logs = ["Log entry"]
        task.execution_attempts = 1
        task.status = TaskStatus.IN_PROGRESS

        # Mock all the check methods to return empty results
        with (
            patch.object(executor_agent, "_check_required_outputs", return_value=[]),
            patch.object(executor_agent, "_check_for_errors", return_value=""),
            patch.object(executor_agent, "_check_execution_metadata", return_value=[]),
            patch.object(executor_agent, "_check_subtasks", return_value=[]),
        ):
            # Test the public method
            is_complete, message = executor_agent.evaluate_task_completion(task)

            # Verify the result
            assert is_complete is True
            assert "Task meets all completion criteria" in message

        # Create a task that doesn't meet completion criteria
        incomplete_task = Task(description="Incomplete task")
        incomplete_task.execution_stage = ExecutionStage.IMPLEMENTING  # Not in FINALIZING stage

        # Test the public method with the incomplete task
        is_complete, message = executor_agent.evaluate_task_completion(incomplete_task)

        # Verify the result
        assert is_complete is False
        assert "not in final stage" in message

    def test_adjust_strategy_code_error(self, executor_agent: ExecutorAgent) -> None:
        """Test _adjust_strategy method for code_error failure type."""
        # Create a basic task
        task = Task(
            task_id=uuid4(),
            description="Test task description",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
        )

        # Test code_error strategy adjustment
        executor_agent._adjust_strategy(task, "code_error", "Syntax error in implementation")
        assert "strategy_adjustments" in task.execution_metadata
        adjustment = task.execution_metadata["strategy_adjustments"][0]
        assert adjustment["failure_type"] == "code_error"
        assert adjustment["adjustment_type"] == "enhanced_instructions"
        assert "enhanced_instructions" in task.metadata
        assert "Syntax error in implementation" in task.metadata["enhanced_instructions"][0]

    def test_adjust_strategy_empty_result(self, executor_agent: ExecutorAgent) -> None:
        """Test _adjust_strategy method for empty_result failure type."""
        # Create a basic task
        task = Task(
            task_id=uuid4(),
            description="Test task description",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
        )

        # Test empty_result strategy adjustment
        executor_agent._adjust_strategy(task, "empty_result", "No output generated")
        assert "strategy_adjustments" in task.execution_metadata
        adjustment = task.execution_metadata["strategy_adjustments"][0]
        assert adjustment["failure_type"] == "empty_result"
        assert adjustment["adjustment_type"] == "approach_change"
        assert task.metadata.get("try_different_approach") is True

    def test_adjust_strategy_verification_failed(self, executor_agent: ExecutorAgent) -> None:
        """Test _adjust_strategy method for verification_failed failure type."""
        # Create a basic task
        task = Task(
            task_id=uuid4(),
            description="Test task description",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
        )

        # Test verification_failed strategy adjustment
        task.verification_details = {"failure_reason": "Output format incorrect"}
        executor_agent._adjust_strategy(task, "verification_failed", "Verification failed")
        assert "strategy_adjustments" in task.execution_metadata
        adjustment = task.execution_metadata["strategy_adjustments"][0]
        assert adjustment["failure_type"] == "verification_failed"
        assert adjustment["adjustment_type"] == "verification_focus"
        assert task.metadata.get("verification_focus") == "Output format incorrect"

    def test_adjust_strategy_stage_stagnation(self, executor_agent: ExecutorAgent) -> None:
        """Test _adjust_strategy method for stage_stagnation failure type."""
        # Create a basic task
        task = Task(
            task_id=uuid4(),
            description="Test task description",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.PLANNING,
        )

        # Test stage_stagnation strategy adjustment
        executor_agent._adjust_strategy(task, "stage_stagnation", "Stuck in planning stage")
        assert "strategy_adjustments" in task.execution_metadata
        adjustment = task.execution_metadata["strategy_adjustments"][0]
        assert adjustment["failure_type"] == "stage_stagnation"
        assert adjustment["adjustment_type"] == "stage_advancement"
        assert task.execution_stage == ExecutionStage.IMPLEMENTING
        assert "planning_result" in task.execution_metadata

    def test_adjust_strategy_dependency_failure(self, executor_agent: ExecutorAgent) -> None:
        """Test _adjust_strategy method for dependency_failure failure type."""
        # Create a basic task
        task = Task(
            task_id=uuid4(),
            description="Test task description",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
        )

        # Test dependency_failure strategy adjustment
        executor_agent._adjust_strategy(task, "dependency_failure", "Required dependency not available")
        assert "strategy_adjustments" in task.execution_metadata
        adjustment = task.execution_metadata["strategy_adjustments"][0]
        assert adjustment["failure_type"] == "dependency_failure"
        assert adjustment["adjustment_type"] == "dependency_workaround"
        assert task.metadata.get("ignore_dependencies") is True
        assert task.status == TaskStatus.IN_PROGRESS

    def test_adjust_strategy_unknown_error(self, executor_agent: ExecutorAgent) -> None:
        """Test _adjust_strategy method for unknown_error failure type."""
        # Create a basic task
        task = Task(
            task_id=uuid4(),
            description="Test task description",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
        )

        # Test general failure strategy adjustment
        executor_agent._adjust_strategy(task, "unknown_error", "Unknown error occurred")
        assert "strategy_adjustments" in task.execution_metadata
        adjustment = task.execution_metadata["strategy_adjustments"][0]
        assert adjustment["failure_type"] == "unknown_error"
        assert adjustment["adjustment_type"] == "general_enhancement"
        assert task.metadata.get("enhanced_context") is True
