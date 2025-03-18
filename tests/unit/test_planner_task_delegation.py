"""Unit tests for planner agent's task delegation methods."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.agent_types.planner import PlannerAgent
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity


class TestPlannerTaskDelegation:
    """Tests for the PlannerAgent's task delegation methods."""

    @pytest.fixture
    def planner_agent(self) -> PlannerAgent:
        """Create a planner agent for testing."""
        provider = MagicMock()
        provider.generate = AsyncMock(return_value="Test response")
        return PlannerAgent(provider=provider)

    @pytest.mark.asyncio
    async def test_delegate_task_simple(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a simple task to an executor."""
        # Mock evaluate_subtask_complexity to return SIMPLE
        with (
            patch.object(
                planner_agent,
                "evaluate_subtask_complexity",
                return_value=TaskComplexity.SIMPLE,
            ),
            patch.object(
                planner_agent,
                "delegate_to_executor",
                new_callable=AsyncMock,
                return_value=Result.success("Task delegated to executor"),
            ),
        ):
            # Delegate a simple task
            result = await planner_agent.delegate_task("Implement a simple function")

            # Verify the result
            assert result.success is True
            assert "Task delegated to executor" in result.data

            # Verify that delegate_to_executor was called
            planner_agent.delegate_to_executor.assert_called_once_with("Implement a simple function")

    @pytest.mark.asyncio
    async def test_delegate_task_moderate(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a moderate task to an executor."""
        # Mock evaluate_subtask_complexity to return MODERATE
        with (
            patch.object(
                planner_agent,
                "evaluate_subtask_complexity",
                return_value=TaskComplexity.MODERATE,
            ),
            patch.object(
                planner_agent,
                "delegate_to_executor",
                new_callable=AsyncMock,
                return_value=Result.success("Task delegated to executor"),
            ),
        ):
            # Delegate a moderate task
            result = await planner_agent.delegate_task("Implement a moderate complexity feature")

            # Verify the result
            assert result.success is True
            assert "Task delegated to executor" in result.data

            # Verify that delegate_to_executor was called
            planner_agent.delegate_to_executor.assert_called_once_with("Implement a moderate complexity feature")

    @pytest.mark.asyncio
    async def test_delegate_task_complex(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a complex task to another planner."""
        # Mock evaluate_subtask_complexity to return COMPLEX
        with (
            patch.object(
                planner_agent,
                "evaluate_subtask_complexity",
                return_value=TaskComplexity.COMPLEX,
            ),
            patch.object(
                planner_agent,
                "delegate_to_planner",
                new_callable=AsyncMock,
                return_value=Result.success("Task delegated to planner"),
            ),
        ):
            # Delegate a complex task
            result = await planner_agent.delegate_task("Implement a complex system")

            # Verify the result
            assert result.success is True
            assert "Task delegated to planner" in result.data

            # Verify that delegate_to_planner was called
            planner_agent.delegate_to_planner.assert_called_once_with("Implement a complex system")

    @pytest.mark.asyncio
    async def test_delegate_task_very_complex(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a very complex task to another planner."""
        # Mock evaluate_subtask_complexity to return VERY_COMPLEX
        with (
            patch.object(
                planner_agent,
                "evaluate_subtask_complexity",
                return_value=TaskComplexity.VERY_COMPLEX,
            ),
            patch.object(
                planner_agent,
                "delegate_to_planner",
                new_callable=AsyncMock,
                return_value=Result.success("Task delegated to planner"),
            ),
        ):
            # Delegate a very complex task
            result = await planner_agent.delegate_task("Implement a very complex architecture")

            # Verify the result
            assert result.success is True
            assert "Task delegated to planner" in result.data

            # Verify that delegate_to_planner was called
            planner_agent.delegate_to_planner.assert_called_once_with("Implement a very complex architecture")

    @pytest.mark.asyncio
    async def test_delegate_to_executor(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a task to an executor agent."""
        # Mock the evaluate_subtask_complexity method
        with patch.object(
            planner_agent,
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.SIMPLE,
        ):
            # Delegate to executor
            result = await planner_agent.delegate_to_executor("Implement a function")

            # Verify the result
            assert result.success is True
            assert "Task delegated directly to executor" in result.data
            assert "Implement a function" in result.data

    @pytest.mark.asyncio
    async def test_delegate_to_planner(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a task to another planner agent."""
        # The planner_agent fixture already sets up _provider as a MagicMock
        # So we can just test the expected behavior without further mocking

        # Delegate to planner
        result = await planner_agent.delegate_to_planner("Plan a complex system")

        # Verify the result
        assert result.success is True
        assert "Task delegated to sub-planner" in result.data

    @pytest.mark.asyncio
    async def test_delegate_to_child(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a task to a child agent."""
        # Add a child agent
        child_id = "child_123"
        planner_agent.add_child(child_id)

        # Mock the state's get_agent method
        mock_child_agent = MagicMock()
        mock_child_agent.process = AsyncMock(return_value=Result.success("Task processed"))

        with patch.object(
            planner_agent.state,
            "get_agent",
            return_value=mock_child_agent,
        ):
            # Delegate to child
            result = await planner_agent.delegate_to_child(child_id, "Process this task")

            # Verify the result
            assert result.success is True
            assert "Task processed" in result.data

            # Verify that the child agent's process method was called
            mock_child_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegate_to_child_not_found(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a task to a non-existent child agent."""
        # Try to delegate to a non-existent child
        with patch.object(
            planner_agent.state,
            "get_agent",
            return_value=None,
        ):
            # Set agent ID for predictable error message
            planner_agent._agent_id = "test_planner"

            # Delegate to non-existent child
            result = await planner_agent.delegate_to_child("non_existent_child", "Process this task")

            # Verify the result
            assert result.success is False
            assert "is not a child of" in result.error

    @pytest.mark.asyncio
    async def test_delegate_single_task(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a single task."""
        # Create a task
        task = Task(description="Implement a function")

        # Mock the delegate_to_executor method since that's what's being called for simple tasks
        with patch.object(
            planner_agent,
            "delegate_to_executor",
            new_callable=AsyncMock,
            return_value=Result.success("Task delegated"),
        ):
            # Delegate the task
            result = await planner_agent._delegate_single_task(task)

            # Verify the result is a tuple with the expected values
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0] == "Task delegated"  # Task result data
            assert result[1] is False  # Should retry flag is False
            assert result[2] == ""  # No error message

            # Verify that delegate_to_executor was called with the task description
            planner_agent.delegate_to_executor.assert_called_once_with(task.description)

    @pytest.mark.asyncio
    async def test_delegate_single_task_with_exception(self, planner_agent: PlannerAgent) -> None:
        """Test delegating a single task with an exception."""
        # Create a task
        task = Task(description="Implement a function")

        # Mock the delegate_to_executor method to raise an exception
        with patch.object(
            planner_agent,
            "delegate_to_executor",
            new_callable=AsyncMock,
            side_effect=Exception("Test exception"),
        ):
            # Delegate the task
            result = await planner_agent._delegate_single_task(task)

            # Verify the result is a tuple with error information
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0] is None  # No task result data
            assert result[1] is False  # Should retry flag is False
            assert "Error delegating task" in result[2]  # Error message
            assert "Test exception" in result[2]
