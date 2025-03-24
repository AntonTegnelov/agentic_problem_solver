"""Unit tests for task delegation functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.planner import PlannerAgent
from src.agent.coordination import AgentCoordinator
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity


class TestTaskDelegation:
    """Tests for task delegation functionality."""

    @pytest.mark.asyncio
    async def test_delegation_by_task_complexity(self) -> None:
        """Test task delegation based on different complexity levels."""
        # Create mock agents
        mock_architect = MagicMock(spec=ArchitectAgent)
        mock_planner = MagicMock(spec=PlannerAgent)
        mock_executor = MagicMock()

        # Setup coordinator with mock registry
        mock_registry = MagicMock()
        coordinator = AgentCoordinator(mock_registry)

        # Configure mock registry responses
        mock_registry.get_agent.side_effect = lambda agent_id: {
            "architect_1": mock_architect,
            "planner_1": mock_planner,
            "executor_1": mock_executor,
        }.get(agent_id)

        # Configure mock process responses
        mock_architect.process = AsyncMock(return_value=Result(success=True, data="Task processed by architect"))
        mock_planner.process = AsyncMock(return_value=Result(success=True, data="Task processed by planner"))
        mock_executor.process = AsyncMock(return_value=Result(success=True, data="Task processed by executor"))

        # Test SIMPLE complexity - should go to executor
        simple_task = Task(description="Implement a simple function", complexity=TaskComplexity.SIMPLE)
        with patch.object(coordinator, "_find_agent_by_complexity", return_value="executor_1"):
            result = await coordinator.delegate_task_flexible(
                source_agent_id="architect_1",
                task=simple_task.description,
                complexity=TaskComplexity.SIMPLE.value,
            )
            assert result.success is True
            assert result.data == "Task processed by executor"
            mock_executor.process.assert_called_once()

        mock_executor.process.reset_mock()

        # Test MODERATE complexity - should go to planner
        moderate_task = Task(description="Implement a feature", complexity=TaskComplexity.MODERATE)
        with patch.object(coordinator, "_find_agent_by_complexity", return_value="planner_1"):
            result = await coordinator.delegate_task_flexible(
                source_agent_id="architect_1",
                task=moderate_task.description,
                complexity=TaskComplexity.MODERATE.value,
            )
            assert result.success is True
            assert result.data == "Task processed by planner"
            mock_planner.process.assert_called_once()

        mock_planner.process.reset_mock()

        # Test COMPLEX complexity - should go to architect
        complex_task = Task(description="Design a system", complexity=TaskComplexity.COMPLEX)
        with patch.object(coordinator, "_find_agent_by_complexity", return_value="architect_1"):
            result = await coordinator.delegate_task_flexible(
                source_agent_id="planner_1",
                task=complex_task.description,
                complexity=TaskComplexity.COMPLEX.value,
            )
            assert result.success is True
            assert result.data == "Task processed by architect"
            mock_architect.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_subtask_relationship_handling(self) -> None:
        """Test proper handling of parent-child task relationships during delegation."""
        # Create mock agents
        mock_architect = MagicMock(spec=ArchitectAgent)
        mock_planner = MagicMock(spec=PlannerAgent)
        mock_executor = MagicMock()

        # Setup coordinator with mock registry
        mock_registry = MagicMock()
        coordinator = AgentCoordinator(mock_registry)

        # Configure mock registry responses
        mock_registry.get_agent.side_effect = lambda agent_id: {
            "architect_1": mock_architect,
            "planner_1": mock_planner,
            "executor_1": mock_executor,
        }.get(agent_id)

        # Create parent task
        parent_task = Task(description="Design user authentication system", complexity=TaskComplexity.COMPLEX)
        parent_task.task_id = "parent-123"  # Set a known task ID for testing

        # Create child tasks
        child_task_1 = Task(
            description="Implement login functionality",
            complexity=TaskComplexity.MODERATE,
            parent_task_id=parent_task.task_id,
        )
        child_task_2 = Task(
            description="Implement password reset",
            complexity=TaskComplexity.SIMPLE,
            parent_task_id=parent_task.task_id,
        )

        # Configure mock process responses
        mock_architect.process = AsyncMock(
            return_value=Result(
                success=True,
                data={
                    "subtasks": [child_task_1, child_task_2],
                    "message": "Parent task broken down into subtasks",
                },
            ),
        )
        mock_planner.process = AsyncMock(return_value=Result(success=True, data="Child task 1 processed"))
        mock_executor.process = AsyncMock(return_value=Result(success=True, data="Child task 2 processed"))

        # Test parent task delegation
        with patch.object(coordinator, "_find_agent_by_complexity", return_value="architect_1"):
            parent_result = await coordinator.delegate_task_flexible(
                source_agent_id="system",
                task=parent_task.description,
                complexity=TaskComplexity.COMPLEX.value,
            )

            # Verify parent task delegation
            assert parent_result.success is True
            assert "Parent task broken down into subtasks" in parent_result.data["message"]
            mock_architect.process.assert_called_once()

            # Verify subtasks were created with correct relationships
            subtasks = parent_result.data["subtasks"]
            assert len(subtasks) == 2
            assert all(task.parent_task_id == parent_task.task_id for task in subtasks)

        # Test child task delegations
        with patch.object(coordinator, "_find_agent_by_complexity", side_effect=["planner_1", "executor_1"]):
            # Delegate first child task (MODERATE complexity)
            child1_result = await coordinator.delegate_task_flexible(
                source_agent_id="architect_1",
                task=child_task_1.description,
                complexity=TaskComplexity.MODERATE.value,
            )
            assert child1_result.success is True
            assert child1_result.data == "Child task 1 processed"
            mock_planner.process.assert_called_once()

            # Delegate second child task (SIMPLE complexity)
            child2_result = await coordinator.delegate_task_flexible(
                source_agent_id="architect_1",
                task=child_task_2.description,
                complexity=TaskComplexity.SIMPLE.value,
            )
            assert child2_result.success is True
            assert child2_result.data == "Child task 2 processed"
            mock_executor.process.assert_called_once()
