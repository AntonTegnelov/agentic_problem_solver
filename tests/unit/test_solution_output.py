"""Tests for solution output retrieval from executor agents."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.agent_types.executor import ExecutorAgent
from src.common_types.enums import ExecutionStage
from src.common_types.task_types import Task, TaskStatus
from src.messages.creation import create_message


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider for testing."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value="Test solution")
    return provider


@pytest.fixture
def executor_agent(mock_provider: MagicMock) -> ExecutorAgent:
    """Create an executor agent for testing."""
    agent = ExecutorAgent(provider=mock_provider)

    # Mock the state
    state = MagicMock()
    agent.get_state = MagicMock(return_value=state)

    return agent


@pytest.mark.asyncio
async def test_process_includes_solution_output(executor_agent: ExecutorAgent) -> None:
    """Test that the process method includes solution output."""
    # Arrange
    message = create_message(role="human", content="Test message")

    # Act
    result = await executor_agent.process(message)

    # Assert
    assert result.success
    result_data = json.loads(result.data)
    assert "solution" in result_data
    assert "content" in result_data
    assert "timestamp" in result_data
    assert result_data["solution"] == "Test solution"


@pytest.mark.asyncio
async def test_update_task_with_result_extracts_solution(executor_agent: ExecutorAgent) -> None:
    """Test that _update_task_with_result extracts the solution from JSON."""
    # Arrange
    task = Task(
        task_id="test-task",
        description="Test task",
        status=TaskStatus.IN_PROGRESS,
        execution_stage=ExecutionStage.IMPLEMENTING,
        execution_metadata={},
    )
    result_json = json.dumps(
        {
            "solution": "Extracted solution",
            "content": "Full content",
            "timestamp": 123456789,
        },
    )

    # Act
    updated_task = executor_agent._update_task_with_result(task, result_json)

    # Assert
    assert updated_task.result == "Extracted solution"
    assert "execution_results" in updated_task.execution_metadata
    assert len(updated_task.execution_metadata["execution_results"]) == 1
    assert updated_task.execution_metadata["execution_results"][0]["solution"] == "Extracted solution"


@pytest.mark.asyncio
async def test_get_task_solution(executor_agent: ExecutorAgent) -> None:
    """Test retrieving a solution from a completed task."""
    # Arrange
    task = Task(
        task_id="test-task",
        description="Test task",
        status=TaskStatus.COMPLETED,
        execution_stage=ExecutionStage.FINALIZING,
        result="Final solution",
        completed_at=123456789,
    )

    # Mock the state to return our task
    state = executor_agent.get_state()
    state.get_tasks.return_value = [task]

    # Act
    result = executor_agent.get_task_solution("test-task")

    # Assert
    assert result.success
    solution_data = json.loads(result.data)
    assert solution_data["solution"] == "Final solution"
    assert solution_data["task_id"] == "test-task"
    assert solution_data["completed_at"] == 123456789


@pytest.mark.asyncio
async def test_get_latest_solution(executor_agent: ExecutorAgent) -> None:
    """Test retrieving the solution from the most recently completed task."""
    # Arrange
    tasks = [
        Task(
            task_id="old-task",
            description="Old task",
            status=TaskStatus.COMPLETED,
            execution_stage=ExecutionStage.FINALIZING,
            result="Old solution",
            completed_at=123456789,
        ),
        Task(
            task_id="new-task",
            description="New task",
            status=TaskStatus.COMPLETED,
            execution_stage=ExecutionStage.FINALIZING,
            result="New solution",
            completed_at=987654321,
        ),
        Task(
            task_id="in-progress-task",
            description="In progress task",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
        ),
    ]

    # Mock the state to return our tasks
    state = executor_agent.get_state()
    state.get_tasks.return_value = tasks

    # Act
    result = executor_agent.get_latest_solution()

    # Assert
    assert result.success
    solution_data = json.loads(result.data)
    assert solution_data["solution"] == "New solution"
    assert solution_data["task_id"] == "new-task"
    assert solution_data["completed_at"] == 987654321


@pytest.mark.asyncio
async def test_get_all_completed_solutions(executor_agent: ExecutorAgent) -> None:
    """Test retrieving solutions from all completed tasks."""
    # Arrange
    tasks = [
        Task(
            task_id="task-1",
            description="First task",
            status=TaskStatus.COMPLETED,
            execution_stage=ExecutionStage.FINALIZING,
            result="Solution 1",
            completed_at=123456789,
        ),
        Task(
            task_id="task-2",
            description="Second task",
            status=TaskStatus.COMPLETED,
            execution_stage=ExecutionStage.FINALIZING,
            result="Solution 2",
            completed_at=987654321,
        ),
        Task(
            task_id="task-3",
            description="In progress task",
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
        ),
    ]

    # Mock the state to return our tasks
    state = executor_agent.get_state()
    state.get_tasks.return_value = tasks

    # Act
    result = executor_agent.get_all_completed_solutions()

    # Assert
    assert result.success
    data = json.loads(result.data)
    assert "solutions" in data
    solutions = data["solutions"]
    assert len(solutions) == 2  # Only completed tasks

    # Solutions should be sorted by completion time (most recent first)
    assert solutions[0]["task_id"] == "task-2"
    assert solutions[0]["solution"] == "Solution 2"
    assert solutions[0]["completed_at"] == 987654321
    assert solutions[0]["description"] == "Second task"

    assert solutions[1]["task_id"] == "task-1"
    assert solutions[1]["solution"] == "Solution 1"
    assert solutions[1]["completed_at"] == 123456789
    assert solutions[1]["description"] == "First task"
