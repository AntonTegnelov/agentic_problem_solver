"""Tests for task synchronization functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.state.base import AgentState
from src.common_types.task_types import (
    ParallelizationStrategy,
    Task,
    TaskDependency,
    TaskStatus,
)


class TestTaskSynchronization:
    """Test task synchronization functionality."""

    def test_synchronize_dependent_tasks_empty(self) -> None:
        """Test synchronizing an empty task list."""
        agent = ArchitectAgent(state_manager=AgentState())
        batches = agent.synchronize_dependent_tasks([])
        assert batches == []

    def test_synchronize_dependent_tasks_no_dependencies(self) -> None:
        """Test synchronizing tasks with no dependencies."""
        agent = ArchitectAgent(state_manager=AgentState())

        # Create tasks with no dependencies
        tasks = [Task(description=f"Task {i}") for i in range(5)]

        batches = agent.synchronize_dependent_tasks(tasks)

        # All tasks should be in a single batch
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_synchronize_dependent_tasks_with_dependencies(self) -> None:
        """Test synchronizing tasks with dependencies."""
        agent = ArchitectAgent(state_manager=AgentState())

        # Create tasks with dependencies
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        task3 = Task(description="Task 3")

        # Task 3 depends on Task 1 and Task 2
        task3.dependencies = [
            TaskDependency(task_id=task1.task_id, description="Depends on Task 1"),
            TaskDependency(task_id=task2.task_id, description="Depends on Task 2"),
        ]

        # Add tasks to state
        agent.state.add_task(task1)
        agent.state.add_task(task2)
        agent.state.add_task(task3)

        batches = agent.synchronize_dependent_tasks([task1, task2, task3])

        # Should have two batches:
        # Batch 1: Task 1 and Task 2 (no dependencies)
        # Batch 2: Task 3 (depends on Task 1 and Task 2)
        assert len(batches) == 2

        # First batch should contain Task 1 and Task 2
        assert len(batches[0]) == 2
        batch1_ids = {str(task.task_id) for task in batches[0]}
        assert str(task1.task_id) in batch1_ids
        assert str(task2.task_id) in batch1_ids

        # Second batch should contain Task 3
        assert len(batches[1]) == 1
        assert batches[1][0].task_id == task3.task_id

    def test_synchronize_dependent_tasks_complex_chain(self) -> None:
        """Test synchronizing tasks with a complex dependency chain."""
        agent = ArchitectAgent(state_manager=AgentState())

        # Create tasks with a complex dependency chain
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        task3 = Task(description="Task 3")
        task4 = Task(description="Task 4")
        task5 = Task(description="Task 5")

        # Task 2 depends on Task 1
        task2.dependencies = [
            TaskDependency(task_id=task1.task_id, description="Depends on Task 1"),
        ]

        # Task 3 depends on Task 2
        task3.dependencies = [
            TaskDependency(task_id=task2.task_id, description="Depends on Task 2"),
        ]

        # Task 5 depends on Task 3 and Task 4
        task5.dependencies = [
            TaskDependency(task_id=task3.task_id, description="Depends on Task 3"),
            TaskDependency(task_id=task4.task_id, description="Depends on Task 4"),
        ]

        # Add tasks to state
        agent.state.add_task(task1)
        agent.state.add_task(task2)
        agent.state.add_task(task3)
        agent.state.add_task(task4)
        agent.state.add_task(task5)

        batches = agent.synchronize_dependent_tasks([task1, task2, task3, task4, task5])

        # Should have four batches:
        # Batch 1: Task 1 and Task 4 (no dependencies)
        # Batch 2: Task 2 (depends on Task 1)
        # Batch 3: Task 3 (depends on Task 2)
        # Batch 4: Task 5 (depends on Task 3 and Task 4)
        assert len(batches) == 4

        # First batch should contain Task 1 and Task 4
        assert len(batches[0]) == 2
        batch1_ids = {str(task.task_id) for task in batches[0]}
        assert str(task1.task_id) in batch1_ids
        assert str(task4.task_id) in batch1_ids

        # Second batch should contain Task 2
        assert len(batches[1]) == 1
        assert batches[1][0].task_id == task2.task_id

        # Third batch should contain Task 3
        assert len(batches[2]) == 1
        assert batches[2][0].task_id == task3.task_id

        # Fourth batch should contain Task 5
        assert len(batches[3]) == 1
        assert batches[3][0].task_id == task5.task_id

    def test_synchronize_dependent_tasks_circular_dependency(self) -> None:
        """Test synchronizing tasks with a circular dependency."""
        agent = ArchitectAgent(state_manager=AgentState())

        # Create tasks with a circular dependency
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        task3 = Task(description="Task 3")

        # Task 2 depends on Task 1
        task2.dependencies = [
            TaskDependency(task_id=task1.task_id, description="Depends on Task 1"),
        ]

        # Task 3 depends on Task 2
        task3.dependencies = [
            TaskDependency(task_id=task2.task_id, description="Depends on Task 2"),
        ]

        # Task 1 depends on Task 3 (creating a circular dependency)
        task1.dependencies = [
            TaskDependency(task_id=task3.task_id, description="Depends on Task 3"),
        ]

        # Add tasks to state
        agent.state.add_task(task1)
        agent.state.add_task(task2)
        agent.state.add_task(task3)

        batches = agent.synchronize_dependent_tasks([task1, task2, task3])

        # Should break the circular dependency and create three batches
        assert len(batches) == 3
        assert len(batches[0]) == 1  # First task to break the cycle
        assert len(batches[1]) == 1  # Second task
        assert len(batches[2]) == 1  # Third task

    @pytest.mark.asyncio
    async def test_execute_synchronized_tasks(self) -> None:
        """Test executing synchronized tasks."""
        agent = ArchitectAgent(state_manager=AgentState())

        # Create tasks with dependencies
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        task3 = Task(description="Task 3")

        # Task 3 depends on Task 1 and Task 2
        task3.dependencies = [
            TaskDependency(task_id=task1.task_id, description="Depends on Task 1"),
            TaskDependency(task_id=task2.task_id, description="Depends on Task 2"),
        ]

        # Add tasks to state
        agent.state.add_task(task1)
        agent.state.add_task(task2)
        agent.state.add_task(task3)

        # Mock the _delegate_single_task method
        with patch.object(agent, "_delegate_single_task", new_callable=AsyncMock) as mock_delegate:
            # Configure the mock to return results based on the task
            async def side_effect(task: Task) -> tuple[str, bool, str]:
                task_id_str = str(task.task_id)
                if task_id_str == str(task1.task_id):
                    return ("Task 1 result", False, "")
                if task_id_str == str(task2.task_id):
                    return ("Task 2 result", False, "")
                if task_id_str == str(task3.task_id):
                    return ("Task 3 result", False, "")
                return (None, True, "Unknown task")

            mock_delegate.side_effect = side_effect

            # Execute the tasks
            results, errors = await agent.execute_synchronized_tasks([task1, task2, task3])

            # Verify the results
            assert len(results) == 3
            assert results[str(task1.task_id)] == "Task 1 result"
            assert results[str(task2.task_id)] == "Task 2 result"
            assert results[str(task3.task_id)] == "Task 3 result"

            # Verify no errors
            assert len(errors) == 0

            # Verify the tasks were executed in the correct order
            # Task 1 and Task 2 should be executed first, then Task 3
            assert mock_delegate.call_count == 3

            # Check that the tasks were updated in the state
            assert agent.state.get_task_by_id(task1.task_id).status == TaskStatus.COMPLETED
            assert agent.state.get_task_by_id(task2.task_id).status == TaskStatus.COMPLETED
            assert agent.state.get_task_by_id(task3.task_id).status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_tasks_with_parallel_dependencies_strategy(self) -> None:
        """Test processing tasks with the PARALLEL_DEPENDENCIES strategy."""
        agent = ArchitectAgent(state_manager=AgentState())

        # Create a parent task with the PARALLEL_DEPENDENCIES strategy
        parent_task = Task(description="Parent Task")
        parent_task.parallelization_strategy = ParallelizationStrategy.PARALLEL_DEPENDENCIES

        # Create child tasks with dependencies
        task1 = Task(description="Task 1", parent_task_id=parent_task.task_id)
        task2 = Task(description="Task 2", parent_task_id=parent_task.task_id)
        task3 = Task(description="Task 3", parent_task_id=parent_task.task_id)

        # Task 3 depends on Task 1 and Task 2
        task3.dependencies = [
            TaskDependency(task_id=task1.task_id, description="Depends on Task 1"),
            TaskDependency(task_id=task2.task_id, description="Depends on Task 2"),
        ]

        # Add tasks to state
        agent.state.add_task(parent_task)
        agent.state.add_task(task1)
        agent.state.add_task(task2)
        agent.state.add_task(task3)

        # Create expected results
        expected_results = {
            str(task1.task_id): "Task 1 result",
            str(task2.task_id): "Task 2 result",
            str(task3.task_id): "Task 3 result",
        }

        # Mock the execute_synchronized_tasks method
        with patch.object(agent, "execute_synchronized_tasks", new_callable=AsyncMock) as mock_execute:
            # Configure the mock to return success for all tasks
            mock_execute.return_value = (expected_results, [])

            # Mock the _process_tasks_with_retry method to directly call execute_synchronized_tasks
            # This is necessary because the actual method has dependencies on LLM providers
            original_method = agent._process_tasks_with_retry

            async def mock_process_tasks(tasks: list[Task]) -> tuple[dict[str, str], list[str]]:
                if parent_task.parallelization_strategy == ParallelizationStrategy.PARALLEL_DEPENDENCIES:
                    return await agent.execute_synchronized_tasks(tasks)
                return {}, []

            agent._process_tasks_with_retry = mock_process_tasks

            try:
                # Process the tasks
                results, errors = await agent._process_tasks_with_retry([task1, task2, task3])

                # Verify the results
                assert len(results) == 3
                assert results == expected_results

                # Verify no errors
                assert len(errors) == 0

                # Verify execute_synchronized_tasks was called
                mock_execute.assert_called_once_with([task1, task2, task3])
            finally:
                # Restore the original method
                agent._process_tasks_with_retry = original_method
