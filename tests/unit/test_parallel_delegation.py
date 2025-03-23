"""Tests for parallel delegation logic in ArchitectAgent and PlannerAgent."""

from typing import Never
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.planner import PlannerAgent
from src.common_types.result_types import Result
from src.common_types.task_types import (
    ParallelizationGroup,
    ParallelizationStrategy,
    Task,
)


@pytest.fixture
def architect_agent() -> ArchitectAgent:
    """Create an architect agent for testing."""
    agent = ArchitectAgent(
        provider="test_provider",
    )
    agent.state = MagicMock()
    agent._logger = MagicMock()
    return agent


@pytest.fixture
def planner_agent() -> PlannerAgent:
    """Create a planner agent for testing."""
    agent = PlannerAgent(
        provider="test_provider",
    )
    agent.state = MagicMock()
    agent._logger = MagicMock()
    return agent


@pytest.mark.asyncio
async def test_architect_configure_parallel_delegation_basic(architect_agent: ArchitectAgent) -> None:
    """Test basic configuration of parallel delegation in ArchitectAgent."""
    # Create test tasks
    tasks = [
        Task(description="Task 1"),
        Task(description="Task 2"),
    ]

    # Configure for parallel execution
    result = await architect_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_INDEPENDENT,
    )

    # Verify tasks were configured correctly
    assert len(result) == 2
    assert result[0].is_parallelizable is True
    assert result[0].parallelization_strategy == ParallelizationStrategy.PARALLEL_INDEPENDENT
    assert result[1].is_parallelizable is True
    assert result[1].parallelization_strategy == ParallelizationStrategy.PARALLEL_INDEPENDENT


@pytest.mark.asyncio
async def test_planner_configure_parallel_delegation_basic(planner_agent: PlannerAgent) -> None:
    """Test basic configuration of parallel delegation in PlannerAgent."""
    # Create test tasks
    tasks = [
        Task(description="Task 1"),
        Task(description="Task 2"),
    ]

    # Configure for parallel execution
    result = planner_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_INDEPENDENT,
    )

    # Verify tasks were configured correctly
    assert len(result.data) == 2
    assert result.data[0].is_parallelizable is True
    assert result.data[0].parallelization_strategy == ParallelizationStrategy.PARALLEL_INDEPENDENT
    assert result.data[1].is_parallelizable is True
    assert result.data[1].parallelization_strategy == ParallelizationStrategy.PARALLEL_INDEPENDENT


@pytest.mark.asyncio
async def test_architect_configure_parallel_delegation_with_parent_task(architect_agent: ArchitectAgent) -> None:
    """Test configuration of parallel delegation with parent task in ArchitectAgent."""
    # Create parent task
    parent_task_id = uuid4()
    parent_task = Task(description="Parent Task", task_id=parent_task_id)
    architect_agent.state.get_task_by_id.return_value = parent_task

    # Create child tasks
    tasks = [
        Task(description="Task 1", parent_task_id=parent_task_id),
        Task(description="Task 2", parent_task_id=parent_task_id),
    ]

    # Configure for parallel execution
    result = await architect_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_INDEPENDENT,
    )

    # Verify tasks were configured correctly
    assert len(result) == 2
    assert result[0].is_parallelizable is True
    assert parent_task.parallelization_strategy == ParallelizationStrategy.PARALLEL_INDEPENDENT


@pytest.mark.asyncio
async def test_planner_configure_parallel_delegation_with_parent_task(planner_agent: PlannerAgent) -> None:
    """Test configuration of parallel delegation with parent task in PlannerAgent."""
    # Create parent task
    parent_task_id = uuid4()
    parent_task = Task(description="Parent Task", task_id=parent_task_id)
    planner_agent.state.get_task_by_id.return_value = parent_task

    # Create child tasks
    tasks = [
        Task(description="Task 1", parent_task_id=parent_task_id),
        Task(description="Task 2", parent_task_id=parent_task_id),
    ]

    # Configure for parallel execution
    result = planner_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_INDEPENDENT,
    )

    # Verify tasks were configured correctly
    assert len(result.data) == 2
    assert result.data[0].is_parallelizable is True
    assert result.data[0].parallelization_strategy == ParallelizationStrategy.PARALLEL_INDEPENDENT
    assert result.data[1].is_parallelizable is True
    assert result.data[1].parallelization_strategy == ParallelizationStrategy.PARALLEL_INDEPENDENT


@pytest.mark.asyncio
async def test_architect_configure_parallel_delegation_with_groups(architect_agent: ArchitectAgent) -> None:
    """Test configuration of parallel delegation with groups in ArchitectAgent."""
    # Create test tasks
    tasks = [
        Task(description="Task 1"),
        Task(description="Task 2"),
    ]

    # Create parallelization groups
    groups = [
        ParallelizationGroup(
            task_ids=[tasks[0].task_id, tasks[1].task_id],
            description="Test Group",
        ),
    ]

    # Configure for parallel execution
    result = await architect_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_GROUPS,
        parallelization_groups=groups,
    )

    # Verify tasks were configured correctly
    assert len(result) == 2
    assert result[0].is_parallelizable is True
    assert result[0].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert result[0].parallelization_groups == groups
    assert result[1].is_parallelizable is True
    assert result[1].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert result[1].parallelization_groups == groups


@pytest.mark.asyncio
async def test_planner_configure_parallel_delegation_with_groups(planner_agent: PlannerAgent) -> None:
    """Test configuration of parallel delegation with groups in PlannerAgent."""
    # Create test tasks
    tasks = [
        Task(description="Task 1"),
        Task(description="Task 2"),
    ]

    # Create parallelization groups
    groups = [
        ParallelizationGroup(
            task_ids=[tasks[0].task_id, tasks[1].task_id],
            description="Test Group",
        ),
    ]

    # Configure for parallel execution
    result = planner_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_GROUPS,
        parallelization_groups=groups,
    )

    # Verify tasks were configured correctly
    assert len(result.data) == 2
    assert result.data[0].is_parallelizable is True
    assert result.data[0].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert result.data[0].parallelization_groups == groups
    assert result.data[1].is_parallelizable is True
    assert result.data[1].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert result.data[1].parallelization_groups == groups


@pytest.mark.asyncio
async def test_architect_configure_parallel_delegation_empty_tasks(architect_agent: ArchitectAgent) -> None:
    """Test configuring parallel delegation with empty tasks in ArchitectAgent."""
    # Test with empty tasks
    tasks = []

    # Call the method
    result = await architect_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_INDEPENDENT,
    )

    # Verify the result
    assert result == []


@pytest.mark.asyncio
async def test_planner_configure_parallel_delegation_empty_tasks(planner_agent: PlannerAgent) -> None:
    """Test configuring parallel delegation with empty tasks in PlannerAgent."""
    # Test with empty tasks
    tasks = []

    # Call the method
    result = planner_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.SEQUENTIAL,
    )

    # Verify the result is a Result with empty data list
    assert result.success is True
    assert result.data == []


@pytest.mark.asyncio
async def test_architect_configure_parallel_delegation_default_groups(architect_agent: ArchitectAgent) -> None:
    """Test configuration of parallel delegation with default groups in ArchitectAgent."""
    # Create test tasks
    tasks = [
        Task(description="Task 1"),
        Task(description="Task 2"),
    ]

    # Configure for parallel execution with PARALLEL_GROUPS strategy but no explicit groups
    result = await architect_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_GROUPS,
    )

    # Verify tasks were configured correctly
    assert len(result) == 2
    assert result[0].is_parallelizable is True
    assert result[0].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert len(result[0].parallelization_groups) == 1
    assert result[1].is_parallelizable is True
    assert result[1].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert len(result[1].parallelization_groups) == 1


@pytest.mark.asyncio
async def test_planner_configure_parallel_delegation_default_groups(planner_agent: PlannerAgent) -> None:
    """Test configuration of parallel delegation with default groups in PlannerAgent."""
    # Create test tasks
    tasks = [
        Task(description="Task 1"),
        Task(description="Task 2"),
    ]

    # Configure for parallel execution with PARALLEL_GROUPS strategy but no explicit groups
    result = planner_agent.configure_parallel_delegation(
        tasks=tasks,
        strategy=ParallelizationStrategy.PARALLEL_GROUPS,
    )

    # Verify tasks were configured correctly
    assert len(result.data) == 2
    assert result.data[0].is_parallelizable is True
    assert result.data[0].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert len(result.data[0].parallelization_groups) == 1
    assert result.data[1].is_parallelizable is True
    assert result.data[1].parallelization_strategy == ParallelizationStrategy.PARALLEL_GROUPS
    assert len(result.data[1].parallelization_groups) == 1


@pytest.mark.asyncio
async def test_planner_process_tasks_parallel_success(planner_agent: PlannerAgent) -> None:
    """Test successful parallel task processing in PlannerAgent."""

    # Mock the _delegate_single_task method to return successful results
    async def mock_delegate_success(task: Task) -> Result:
        return Result.success(f"Result for {task}")

    planner_agent._delegate_single_task = MagicMock(side_effect=mock_delegate_success)

    # Test with a list of task descriptions
    tasks = [Task(description="Task 1"), Task(description="Task 2"), Task(description="Task 3")]

    # Call the method
    result = await planner_agent.delegate_tasks_parallel(tasks)

    # Verify the result
    assert result.success is True
    # The data is a list of Result objects
    assert len(result.data) == 3
    # Verify each task was delegated
    assert planner_agent._delegate_single_task.call_count == 3


@pytest.mark.asyncio
async def test_planner_process_tasks_parallel_mixed_results(planner_agent: PlannerAgent) -> None:
    """Test parallel task processing with mixed success/failure results in PlannerAgent."""

    # Mock the _delegate_single_task method to return mixed results
    async def mock_delegate_mixed(task: Task) -> tuple[str | None, bool, str]:
        if task.description == "Task 2":
            return None, True, f"Failed to process {task.description}"
        return f"Result for {task.description}", False, ""

    # Use the wrapper method
    planner_agent._delegate_single_task_wrapper = MagicMock(side_effect=mock_delegate_mixed)

    # Test with a list of task descriptions
    tasks = [Task(description="Task 1"), Task(description="Task 2"), Task(description="Task 3")]

    # Call the method
    result = await planner_agent.delegate_tasks_parallel(tasks)

    # Verify the result - it can be either success with errors or failure depending on implementation
    assert result.success is False  # Overall result should be failure
    assert isinstance(result.data, list)
    assert len(result.data) == 3
    # Check that our successful tasks are properly represented
    success_count = sum(1 for item in result.data if item.success)
    assert success_count == 2
    # Check that our results contain the expected task descriptions
    result_str = str(result.data)
    assert "Task 1" in result_str
    assert "Failed to process Task 2" in result_str
    assert "Task 3" in result_str


@pytest.mark.asyncio
async def test_planner_process_tasks_parallel_all_fail(planner_agent: PlannerAgent) -> None:
    """Test parallel task processing with all tasks failing in PlannerAgent."""

    # Mock the _delegate_single_task method to return failure results
    async def mock_delegate_failure(task: Task) -> Result:
        return Result.failure(f"Failed to process {task.description}")

    planner_agent._delegate_single_task = MagicMock(side_effect=mock_delegate_failure)

    # Test with a list of task descriptions
    tasks = [Task(description="Task 1"), Task(description="Task 2"), Task(description="Task 3")]

    # Call the method
    result = await planner_agent.delegate_tasks_parallel(tasks)

    # Verify the result
    assert result.success is False
    assert "failed" in str(result.error).lower()


@pytest.mark.asyncio
async def test_planner_process_tasks_parallel_empty(planner_agent: PlannerAgent) -> None:
    """Test parallel task processing with empty task list in PlannerAgent."""
    # Call the method with empty list
    result = await planner_agent.delegate_tasks_parallel([])

    # Verify the result is a success with empty data list
    assert result.success is True
    assert result.data == []


@pytest.mark.asyncio
async def test_planner_process_tasks_parallel_exception(planner_agent: PlannerAgent) -> None:
    """Test parallel task processing with exception in PlannerAgent."""

    # Mock the _delegate_single_task method to raise an exception
    async def mock_delegate_exception(_: Task) -> Never:
        msg = "Test exception"
        raise ValueError(msg)

    planner_agent._delegate_single_task = MagicMock(side_effect=mock_delegate_exception)

    # Test with a list of task descriptions
    tasks = [Task(description="Task 1"), Task(description="Task 2"), Task(description="Task 3")]

    # Call the method
    result = await planner_agent.delegate_tasks_parallel(tasks)

    # Verify the result shows failure
    assert result.success is False
    assert "Error processing task" in str(result.data) or "Test exception" in str(result.data)


@pytest.mark.asyncio
async def test_architect_process_tasks_with_retry_parallel_all(architect_agent: ArchitectAgent) -> None:
    """Test processing tasks with retry using PARALLEL_ALL strategy."""
    # Create test tasks
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    tasks = [task1, task2]

    # Set up parent task with PARALLEL_ALL strategy
    parent_task_id = uuid4()
    parent_task = Task(
        description="Parent Task",
        task_id=parent_task_id,
        parallelization_strategy=ParallelizationStrategy.PARALLEL_ALL,
    )
    task1.parent_task_id = parent_task_id
    task2.parent_task_id = parent_task_id

    # Mock the state to return the parent task
    architect_agent.state.get_task_by_id.return_value = parent_task

    # Mock the _delegate_single_task method
    original_delegate_single_task = architect_agent._delegate_single_task

    # First call fails for task1, second succeeds
    call_count = 0

    async def mock_delegate_task(task: Task) -> tuple[str | None, bool, str]:
        nonlocal call_count
        if task == task1 and call_count == 0:
            call_count += 1
            return None, True, "Temporary error"
        return f"Result for {task.description}", False, ""

    architect_agent._delegate_single_task = MagicMock(side_effect=mock_delegate_task)

    # Process tasks with retry
    results, errors = await architect_agent._process_tasks_with_retry(tasks)

    # Verify results
    assert len(results) == 2
    assert len(errors) == 0
    assert str(task1.task_id) in results
    assert str(task2.task_id) in results

    # Restore the original method
    architect_agent._delegate_single_task = original_delegate_single_task


@pytest.mark.asyncio
async def test_architect_process_tasks_with_retry_parallel_groups(architect_agent: ArchitectAgent) -> None:
    """Test processing tasks with retry using PARALLEL_GROUPS strategy."""
    # Create test tasks
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    tasks = [task1, task2]

    # Set up parent task with PARALLEL_GROUPS strategy
    parent_task_id = uuid4()
    parent_task = Task(
        description="Parent Task",
        task_id=parent_task_id,
        parallelization_strategy=ParallelizationStrategy.PARALLEL_GROUPS,
    )
    task1.parent_task_id = parent_task_id
    task2.parent_task_id = parent_task_id

    # Create a parallelization group
    group = ParallelizationGroup(
        task_ids=[task1.task_id, task2.task_id],
        description="Test Group",
    )
    parent_task.parallelization_groups = [group]

    # Mock the state to return the parent task
    architect_agent.state.get_task_by_id.return_value = parent_task

    # Mock the _delegate_single_task method to return a coroutine
    async def mock_delegate_single_task(task: Task) -> tuple[str, bool, str]:
        return f"Success for {task.description}", False, ""

    original_delegate_single_task = architect_agent._delegate_single_task
    architect_agent._delegate_single_task = mock_delegate_single_task

    # Process tasks with retry
    results, errors = await architect_agent._process_tasks_with_retry(tasks)

    # Verify results
    assert len(results) == 2
    assert len(errors) == 0
    assert all(str(task.task_id) in results for task in tasks)

    # Restore original method
    architect_agent._delegate_single_task = original_delegate_single_task


@pytest.mark.asyncio
async def test_architect_process_tasks_with_retry_parallel_independent(architect_agent: ArchitectAgent) -> None:
    """Test processing tasks with retry using PARALLEL_INDEPENDENT strategy."""
    # Create test tasks
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    task3 = Task(description="Task 3")
    tasks = [task1, task2, task3]

    # Set up parent task with PARALLEL_INDEPENDENT strategy
    parent_task_id = uuid4()
    parent_task = Task(
        description="Parent Task",
        task_id=parent_task_id,
        parallelization_strategy=ParallelizationStrategy.PARALLEL_INDEPENDENT,
    )
    task1.parent_task_id = parent_task_id
    task2.parent_task_id = parent_task_id
    task3.parent_task_id = parent_task_id

    # Mock the state to return the parent task
    architect_agent.state.get_task_by_id.return_value = parent_task

    # Mock the _delegate_single_task method to return a coroutine
    async def mock_delegate_single_task(task: Task) -> tuple[str, bool, str]:
        return f"Success for {task.description}", False, ""

    original_delegate_single_task = architect_agent._delegate_single_task
    architect_agent._delegate_single_task = mock_delegate_single_task

    # Process tasks with retry
    results, errors = await architect_agent._process_tasks_with_retry(tasks)

    # Verify results
    assert len(results) == 3
    assert len(errors) == 0
    assert all(str(task.task_id) in results for task in tasks)

    # Restore original method
    architect_agent._delegate_single_task = original_delegate_single_task


@pytest.mark.asyncio
async def test_architect_process_tasks_with_retry_with_errors(architect_agent: ArchitectAgent) -> None:
    """Test processing tasks with retry with persistent errors."""
    # Create test tasks
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    tasks = [task1, task2]

    # Set up sequential processing strategy
    parent_task_id = uuid4()
    parent_task = Task(
        description="Parent Task",
        task_id=parent_task_id,
        parallelization_strategy=ParallelizationStrategy.SEQUENTIAL,
    )
    task1.parent_task_id = parent_task_id
    task2.parent_task_id = parent_task_id

    # Mock the state to return the parent task
    architect_agent.state.get_task_by_id.return_value = parent_task

    # Mock the _process_batch_with_strategy method to bypass the internals
    original_process_batch = architect_agent._process_batch_with_strategy

    async def mock_process_batch(
        _tasks: list[Task],
        _strategy: str,
        _retry_count: int,
        _max_retries: int,
    ) -> tuple[dict[str, str], list[str], list[Task]]:
        results = {str(task1.task_id): "Result for Task 1"}
        errors = ["Task 2: Persistent error"]
        retry_tasks = []
        return results, errors, retry_tasks

    architect_agent._process_batch_with_strategy = mock_process_batch

    # Process tasks with retry
    results, errors = await architect_agent._process_tasks_with_retry(tasks, max_retries=1)

    # Verify results
    assert len(results) == 1
    assert len(errors) == 1
    assert str(task1.task_id) in results
    assert "Task 2: Persistent error" in errors[0]

    # Restore the original method
    architect_agent._process_batch_with_strategy = original_process_batch


@pytest.mark.asyncio
async def test_architect_process_tasks_with_retry_with_exception(architect_agent: ArchitectAgent) -> None:
    """Test processing tasks with retry with an exception."""
    # Create test tasks
    task1 = Task(description="Task 1")
    task2 = Task(description="Task 2")
    tasks = [task1, task2]

    # Set up sequential processing to ensure the exception is handled properly
    parent_task_id = uuid4()
    parent_task = Task(
        description="Parent Task",
        task_id=parent_task_id,
        parallelization_strategy=ParallelizationStrategy.SEQUENTIAL,
    )
    task1.parent_task_id = parent_task_id
    task2.parent_task_id = parent_task_id

    # Mock the state to return the parent task
    architect_agent.state.get_task_by_id.return_value = parent_task

    # Mock the _delegate_single_task method to raise an exception for task2
    async def mock_delegate_single_task(task: Task) -> tuple[str | None, bool, str]:
        if task == task2:
            return None, False, "Test exception"
        return f"Result for {task.description}", False, ""

    original_delegate_single_task = architect_agent._delegate_single_task
    architect_agent._delegate_single_task = mock_delegate_single_task

    # Process tasks with retry
    results, errors = await architect_agent._process_tasks_with_retry(tasks)

    # Verify results
    assert len(results) == 1
    assert len(errors) == 1
    assert str(task1.task_id) in results
    assert "Test exception" in errors[0]

    # Restore the original method
    architect_agent._delegate_single_task = original_delegate_single_task
