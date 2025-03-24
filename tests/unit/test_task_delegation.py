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
