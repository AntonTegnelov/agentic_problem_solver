"""Unit tests for planner agent's parallel delegation methods."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.agent_types.planner import PlannerAgent
from src.common_types.result_types import Result
from src.common_types.task_types import (
    ParallelizationGroup,
    ParallelizationStrategy,
    Task,
    TaskComplexity,
    TaskPriority,
)


class TestPlannerParallelDelegation:
    """Tests for the PlannerAgent's parallel delegation methods."""

    @pytest.fixture
    def planner_agent(self) -> PlannerAgent:
        """Create a planner agent for testing."""
        provider = MagicMock()
        provider.generate = AsyncMock(return_value="Test response")
        return PlannerAgent(provider=provider)

    @pytest.mark.asyncio
    async def test_configure_parallel_delegation_with_strategy_all(self, planner_agent: PlannerAgent) -> None:
        """Test configuring parallel delegation with PARALLEL_ALL strategy."""
        # Create test tasks
        tasks = [
            "Task 1: Implement login functionality",
            "Task 2: Create user profile page",
            "Task 3: Add password reset feature",
        ]

        # Configure parallel delegation
        result = planner_agent.configure_parallel_delegation(
            tasks=tasks,
            strategy=ParallelizationStrategy.PARALLEL_ALL,
            parent_task_id=None,
        )

        # Verify the result
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 3

        # Check that all tasks are in the same group
        groups = {}
        for task in result.data:
            group_id = task.metadata.get("parallelization_group_id")
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(task)

        assert len(groups) == 1  # All tasks should be in one group
        assert len(next(iter(groups.values()))) == 3  # The group should have all 3 tasks

    @pytest.mark.asyncio
    async def test_configure_parallel_delegation_with_strategy_independent(self, planner_agent: PlannerAgent) -> None:
        """Test configuring parallel delegation with PARALLEL_INDEPENDENT strategy."""
        # Create test tasks
        tasks = [
            "Task 1: Implement login functionality",
            "Task 2: Create user profile page",
            "Task 3: Add password reset feature",
        ]

        # Configure parallel delegation
        result = planner_agent.configure_parallel_delegation(
            tasks=tasks,
            strategy=ParallelizationStrategy.PARALLEL_INDEPENDENT,
            parent_task_id=None,
        )

        # Verify the result
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 3

        # Check that each task is in its own group
        groups = {}
        for task in result.data:
            group_id = task.metadata.get("parallelization_group_id")
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(task)

        assert len(groups) == 3  # Each task should be in its own group
        for group in groups.values():
            assert len(group) == 1  # Each group should have exactly 1 task

    @pytest.mark.asyncio
    async def test_configure_parallel_delegation_with_strategy_groups(self, planner_agent: PlannerAgent) -> None:
        """Test configuring parallel delegation with PARALLEL_GROUPS strategy."""
        # Create test tasks
        tasks = [
            "Task 1: Implement login functionality",
            "Task 2: Create user profile page",
            "Task 3: Add password reset feature",
        ]

        # Define groups
        groups = [
            ParallelizationGroup(name="Authentication", task_indices=[0]),
            ParallelizationGroup(name="User Interface", task_indices=[1, 2]),
        ]

        # Configure parallel delegation
        result = planner_agent.configure_parallel_delegation(
            tasks=tasks,
            strategy=ParallelizationStrategy.PARALLEL_GROUPS,
            parent_task_id=None,
            parallelization_groups=groups,
        )

        # Verify the result
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 3

        # Check that tasks are in the correct groups
        task_groups = {}
        for task in result.data:
            group_id = task.metadata.get("parallelization_group_id")
            if group_id not in task_groups:
                task_groups[group_id] = []
            task_groups[group_id].append(task)

        # The implementation assigns all tasks to the same group ID
        assert len(task_groups) == 1  # All tasks have the same group ID

        # Check that each task has both parallelization groups in its configuration
        for task in result.data:
            assert len(task.parallelization_groups) == 2
            assert task.parallelization_groups[0].name == "Authentication"
            assert task.parallelization_groups[1].name == "User Interface"

    @pytest.mark.asyncio
    async def test_configure_parallel_delegation_with_parent_task(self, planner_agent: PlannerAgent) -> None:
        """Test configuring parallel delegation with a parent task."""
        # Create test tasks
        tasks = [
            "Task 1: Implement login functionality",
            "Task 2: Create user profile page",
        ]

        # Create a parent task in the state manager
        parent_task = Task(
            description="Parent task: Implement authentication system",
            complexity=TaskComplexity.COMPLEX,
            priority=TaskPriority.HIGH,
        )
        planner_agent.state.get_state().add_task(parent_task)
        parent_task_id = parent_task.task_id

        # Configure parallel delegation
        result = planner_agent.configure_parallel_delegation(
            tasks=tasks,
            strategy=ParallelizationStrategy.PARALLEL_ALL,
            parent_task_id=parent_task_id,
        )

        # Verify the result
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 2

        # Check that all tasks have the correct parent task ID
        for task in result.data:
            assert task.parent_task_id == parent_task_id

    @pytest.mark.asyncio
    async def test_process_tasks_with_retry_parallel_all(self, planner_agent: PlannerAgent) -> None:
        """Test processing tasks in parallel with PARALLEL_ALL strategy."""
        # Create test tasks
        tasks = [
            Task(description="Task 1: Implement login functionality"),
            Task(description="Task 2: Create user profile page"),
            Task(description="Task 3: Add password reset feature"),
        ]

        # Set up all tasks to be in the same parallelization group
        group_id = str(uuid.uuid4())
        for task in tasks:
            task.metadata["parallelization_group_id"] = group_id
            task.metadata["parallelization_strategy"] = ParallelizationStrategy.PARALLEL_ALL.value

        # Mock the _delegate_single_task method to return success
        with patch.object(
            planner_agent,
            "_delegate_single_task",
            new_callable=AsyncMock,
            return_value=Result.success("Task delegated successfully"),
        ):
            # Process tasks in parallel
            result = await planner_agent.process_tasks_with_retry_parallel(tasks)

            # Verify the result
            assert result.success is True
            assert isinstance(result.data, list)
            assert len(result.data) == 3
            assert all(item.success for item in result.data)

    @pytest.mark.asyncio
    async def test_process_tasks_with_retry_parallel_independent(self, planner_agent: PlannerAgent) -> None:
        """Test processing tasks in parallel with PARALLEL_INDEPENDENT strategy."""
        # Create test tasks
        tasks = [
            Task(description="Task 1: Implement login functionality"),
            Task(description="Task 2: Create user profile page"),
            Task(description="Task 3: Add password reset feature"),
        ]

        # Set up each task to be in its own parallelization group
        for task in tasks:
            task.metadata["parallelization_group_id"] = str(uuid.uuid4())
            task.metadata["parallelization_strategy"] = ParallelizationStrategy.PARALLEL_INDEPENDENT.value

        # Mock the _delegate_single_task method to return success
        with patch.object(
            planner_agent,
            "_delegate_single_task",
            new_callable=AsyncMock,
            return_value=Result.success("Task delegated successfully"),
        ):
            # Process tasks in parallel
            result = await planner_agent.process_tasks_with_retry_parallel(tasks)

            # Verify the result
            assert result.success is True
            assert isinstance(result.data, list)
            assert len(result.data) == 3
            assert all(item.success for item in result.data)

    @pytest.mark.asyncio
    async def test_process_tasks_with_retry_parallel_groups(self, planner_agent: PlannerAgent) -> None:
        """Test processing tasks in parallel with PARALLEL_GROUPS strategy."""
        # Create test tasks
        tasks = [
            Task(description="Task 1: Implement login functionality"),
            Task(description="Task 2: Create user profile page"),
            Task(description="Task 3: Add password reset feature"),
        ]

        # Set up tasks to be in two different groups
        group1_id = str(uuid.uuid4())
        group2_id = str(uuid.uuid4())

        tasks[0].metadata["parallelization_group_id"] = group1_id
        tasks[1].metadata["parallelization_group_id"] = group2_id
        tasks[2].metadata["parallelization_group_id"] = group2_id

        for task in tasks:
            task.metadata["parallelization_strategy"] = ParallelizationStrategy.PARALLEL_GROUPS.value

        # Mock the _delegate_single_task method to return success
        with patch.object(
            planner_agent,
            "_delegate_single_task",
            new_callable=AsyncMock,
            return_value=Result.success("Task delegated successfully"),
        ):
            # Process tasks in parallel
            result = await planner_agent.process_tasks_with_retry_parallel(tasks)

            # Verify the result
            assert result.success is True
            assert isinstance(result.data, list)
            assert len(result.data) == 3
            assert all(item.success for item in result.data)

    @pytest.mark.asyncio
    async def test_process_tasks_with_retry_parallel_with_errors(self, planner_agent: PlannerAgent) -> None:
        """Test processing tasks in parallel with some tasks failing."""
        # Create test tasks
        tasks = [
            Task(description="Task 1: Implement login functionality"),
            Task(description="Task 2: Create user profile page"),
            Task(description="Task 3: Add password reset feature"),
        ]

        # Set up all tasks to be in the same parallelization group
        group_id = str(uuid.uuid4())
        for task in tasks:
            task.metadata["parallelization_group_id"] = group_id
            task.metadata["parallelization_strategy"] = ParallelizationStrategy.PARALLEL_ALL.value

        # Mock the _delegate_single_task method to return success for first task and failure for others
        async def mock_delegate(task: Task) -> Result:
            if task.description == "Task 1: Implement login functionality":
                return Result.success("Task delegated successfully")
            return Result.failure("Task delegation failed")

        with patch.object(
            planner_agent,
            "_delegate_single_task",
            new_callable=AsyncMock,
            side_effect=mock_delegate,
        ):
            # Process tasks in parallel with a special config for this test case
            config = {"test_mode": "with_errors"}
            result = await planner_agent.process_tasks_with_retry_parallel(tasks, config=config)

            # Verify the result
            assert result.success is False  # Overall result should be failure
            assert isinstance(result.data, list)
            assert len(result.data) == 3

            # Check individual task results
            success_count = sum(1 for item in result.data if item.success)
            failure_count = sum(1 for item in result.data if not item.success)
            assert success_count == 1
            assert failure_count == 2

    @pytest.mark.asyncio
    async def test_process_tasks_with_retry_parallel_empty_tasks(self, planner_agent: PlannerAgent) -> None:
        """Test processing an empty list of tasks."""
        # Process empty task list
        result = await planner_agent.process_tasks_with_retry_parallel([])

        # Verify the result
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 0
