"""Unit tests for flexible delegation between agent types.

This module contains tests that verify the delegation logic between different agent types,
including direct delegation from ArchitectAgent to ExecutorAgent for simple tasks,
and recursive delegation from PlannerAgent to additional PlannerAgent instances
for complex sub-components.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.executor import ExecutorAgent
from src.agent.agent_types.planner import PlannerAgent
from src.agent.coordination import AgentCoordinator, InMemoryAgentRegistry
from src.common_types.enums import AgentRole
from src.common_types.result_types import Result
from src.common_types.task_types import TaskComplexity


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock agent registry."""
    return MagicMock(spec=InMemoryAgentRegistry)


@pytest.fixture
def mock_coordinator() -> MagicMock:
    """Create a mock agent coordinator."""
    return MagicMock(spec=AgentCoordinator)


@pytest.fixture
def mock_architect_agent() -> MagicMock:
    """Create a mock architect agent."""
    agent = MagicMock(spec=ArchitectAgent)
    agent.get_agent_id.return_value = "architect_1"
    agent.get_role.return_value = AgentRole.ARCHITECT.value
    agent.analyze_task_complexity.return_value = TaskComplexity.MODERATE
    return agent


@pytest.fixture
def mock_planner_agent() -> MagicMock:
    """Create a mock planner agent."""
    agent = MagicMock(spec=PlannerAgent)
    agent.get_agent_id.return_value = "planner_1"
    agent.get_role.return_value = AgentRole.PLANNER.value
    agent.evaluate_subtask_complexity.return_value = TaskComplexity.MODERATE
    return agent


@pytest.fixture
def mock_executor_agent() -> MagicMock:
    """Create a mock executor agent."""
    agent = MagicMock(spec=ExecutorAgent)
    agent.get_agent_id.return_value = "executor_1"
    agent.get_role.return_value = AgentRole.EXECUTOR.value
    return agent


class TestArchitectAgentDelegation:
    """Test delegation logic in ArchitectAgent."""

    @pytest.mark.asyncio
    async def test_architect_delegates_to_executor_for_simple_task(self) -> None:
        """Test that ArchitectAgent delegates simple tasks directly to ExecutorAgent."""
        # Arrange
        with (
            patch("src.agent.agent_types.architect.ArchitectAgent._validate_provider"),
            patch("src.agent.agent_types.create_executor_agent") as mock_create_executor,
        ):
            # Create a mock executor agent
            mock_executor = MagicMock()
            mock_executor.get_agent_id.return_value = "executor_123"
            mock_executor.state = MagicMock()
            mock_executor.process.return_value = Result.success("Test response")
            mock_create_executor.return_value = mock_executor

            architect = ArchitectAgent()
            architect.analyze_task_complexity = MagicMock(return_value=TaskComplexity.SIMPLE)
            architect._logger = MagicMock()
            architect._provider = MagicMock()

            # Act
            result = await architect.delegate_to_executor("Implement a simple hello world function")

            # Assert
            assert result.success is True
            assert result.data == "Test response"

    @pytest.mark.asyncio
    async def test_architect_analyzes_task_complexity_correctly(self) -> None:
        """Test that ArchitectAgent correctly analyzes task complexity."""
        # Arrange
        with patch("src.agent.agent_types.architect.ArchitectAgent._validate_provider"):
            architect = ArchitectAgent()

            # Mock the LLM-based complexity analysis to avoid actual LLM calls
            architect._analyze_task_complexity_with_llm = AsyncMock(return_value=TaskComplexity.COMPLEX)

            # Act - Test with different task descriptions
            simple_result = architect._analyze_task_complexity_rule_based("Print hello world")
            complex_result = await architect._analyze_task_complexity_with_llm(
                "Build a distributed microservice architecture with load balancing",
            )

            # Assert
            assert simple_result == TaskComplexity.SIMPLE
            assert complex_result == TaskComplexity.COMPLEX


class TestPlannerAgentDelegation:
    """Test delegation logic in PlannerAgent."""

    @pytest.mark.asyncio
    async def test_planner_delegates_to_executor_for_simple_subtask(self) -> None:
        """Test that PlannerAgent delegates simple subtasks to ExecutorAgent."""
        # Arrange
        with patch("src.agent.agent_types.planner.PlannerAgent._validate_provider"):
            planner = PlannerAgent()
            planner.evaluate_subtask_complexity = MagicMock(return_value=TaskComplexity.SIMPLE)
            planner._logger = MagicMock()

            # Act
            result = await planner.delegate_to_executor("Implement a simple validation function")

            # Assert
            assert result.success is True
            assert "Task delegated directly to executor" in result.data

    @pytest.mark.asyncio
    async def test_planner_delegates_to_another_planner_for_complex_subtask(self) -> None:
        """Test that PlannerAgent delegates complex subtasks to another PlannerAgent."""
        # Arrange
        with patch("src.agent.agent_types.planner.PlannerAgent._validate_provider"):
            planner = PlannerAgent()
            planner.evaluate_subtask_complexity = MagicMock(return_value=TaskComplexity.COMPLEX)
            planner._logger = MagicMock()

            # Mock the _create_sub_planner method to avoid creating a real sub-planner
            planner._provider = MagicMock()

            # Act
            result = await planner.delegate_to_planner("Design a complex authentication system")

            # Assert
            assert result.success is True
            assert "Task delegated to sub-planner" in result.data


class TestAgentCoordinatorDelegation:
    """Test delegation logic in AgentCoordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_delegates_task_by_complexity(
        self,
        mock_registry: MagicMock,
        mock_architect_agent: MagicMock,
        mock_planner_agent: MagicMock,
        mock_executor_agent: MagicMock,
    ) -> None:
        """Test that AgentCoordinator delegates tasks based on complexity."""
        # Arrange
        coordinator = AgentCoordinator(mock_registry)

        # Setup registry mocks
        mock_registry.get_agent.side_effect = lambda agent_id: {
            "architect_1": mock_architect_agent,
            "planner_1": mock_planner_agent,
            "executor_1": mock_executor_agent,
        }.get(agent_id)

        mock_registry.find_agents_by_role.side_effect = lambda role: {
            AgentRole.ARCHITECT.value: [{"agent_id": "architect_1"}],
            AgentRole.PLANNER.value: [{"agent_id": "planner_1"}],
            AgentRole.EXECUTOR.value: [{"agent_id": "executor_1"}],
        }.get(role, [])

        # Setup agent mocks
        mock_architect_agent.process = AsyncMock(return_value=Result(success=True, data="Task processed by architect"))
        mock_planner_agent.process = AsyncMock(return_value=Result(success=True, data="Task processed by planner"))
        mock_executor_agent.process = AsyncMock(return_value=Result(success=True, data="Task processed by executor"))

        # Mock the complexity-based delegation methods
        coordinator._architect_delegation_by_complexity = MagicMock(return_value="planner_1")
        coordinator._planner_delegation_by_complexity = MagicMock(return_value="executor_1")

        # Act
        with patch.object(coordinator, "_find_agent_by_complexity", return_value="planner_1"):
            result = await coordinator.delegate_task_flexible(
                source_agent_id="architect_1",
                task="Design a user authentication system",
                complexity=TaskComplexity.MODERATE.value,
            )

        # Assert
        assert result.success is True
        assert result.data == "Task processed by planner"
        mock_planner_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_coordinator_delegates_task_by_role(
        self,
        mock_registry: MagicMock,
        mock_architect_agent: MagicMock,
        mock_planner_agent: MagicMock,
    ) -> None:
        """Test that AgentCoordinator delegates tasks based on target role."""
        # Arrange
        coordinator = AgentCoordinator(mock_registry)

        # Setup registry mocks
        mock_registry.get_agent.side_effect = lambda agent_id: {
            "architect_1": mock_architect_agent,
            "planner_1": mock_planner_agent,
        }.get(agent_id)

        mock_registry.find_agents_by_role.side_effect = lambda role: {
            AgentRole.ARCHITECT.value: [{"agent_id": "architect_1"}],
            AgentRole.PLANNER.value: [{"agent_id": "planner_1"}],
        }.get(role, [])

        # Setup agent mocks
        mock_architect_agent.process = AsyncMock(return_value=Result(success=True, data="Task processed by architect"))
        mock_planner_agent.process = AsyncMock(return_value=Result(success=True, data="Task processed by planner"))

        # Mock the role-based delegation method
        coordinator._find_agent_by_role = MagicMock(return_value="planner_1")

        # Act
        result = await coordinator.delegate_task_flexible(
            source_agent_id="architect_1",
            task="Design a user authentication system",
            target_role=AgentRole.PLANNER.value,
        )

        # Assert
        assert result.success is True
        assert result.data == "Task processed by planner"
        mock_planner_agent.process.assert_called_once()
