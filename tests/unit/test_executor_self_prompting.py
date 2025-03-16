"""Unit tests for ExecutorAgent self-prompting capabilities.

This module tests the ExecutorAgent's ability to systematically solve tasks
by iterating through execution stages and self-prompting until completion.
"""

import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.agent_types.executor import ExecutorAgent
from src.common_types.enums import ExecutionStage, VerificationStatus
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskStatus


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate = MagicMock(return_value="Mock response")
    provider.generate_stream = AsyncMock()
    return provider


@pytest.fixture
def executor_agent(mock_provider: MagicMock) -> ExecutorAgent:
    """Create an ExecutorAgent with a mock provider."""
    return ExecutorAgent(provider=mock_provider)


@pytest.fixture
def simple_task() -> Task:
    """Create a simple task for testing."""
    return Task(
        description="Implement a function to add two numbers",
        task_id=uuid.uuid4(),
        execution_stage=ExecutionStage.PLANNING,
        verification_status=VerificationStatus.PENDING,
    )


class TestExecutorSelfPrompting:
    """Test suite for ExecutorAgent self-prompting capabilities."""

    @patch.object(ExecutorAgent, "_process_task_iteration")
    async def test_iterate_task_basic_flow(
        self,
        mock_process: AsyncMock,
        executor_agent: ExecutorAgent,
        simple_task: Task,
    ) -> None:
        """Test the basic flow of task iteration."""
        # Mock the _process_task_iteration method to return a successful result
        mock_process.return_value = Result(success=True, data="Implemented function", error=None)

        # Call iterate_task
        result = await executor_agent.iterate_task(simple_task)

        # Verify the result
        assert result.success
        assert result.data.execution_attempts == 1
        assert result.data.execution_stage == ExecutionStage.IMPLEMENTING  # Should advance to next stage
        assert result.data.status == TaskStatus.IN_PROGRESS
        assert result.data.result == "Implemented function"

    @patch.object(ExecutorAgent, "_process_task_iteration")
    @patch.object(ExecutorAgent, "_handle_preverified_task")
    async def test_task_completion_criteria(
        self,
        mock_handle_preverified: AsyncMock,
        mock_process: AsyncMock,  # noqa: ARG002
        executor_agent: ExecutorAgent,
        simple_task: Task,
    ) -> None:
        """Test that task completion criteria are properly evaluated."""
        # Set up a task in the final stage with verification passed
        simple_task.execution_stage = ExecutionStage.FINALIZING
        simple_task.verification_status = VerificationStatus.PASSED
        simple_task.result = "function add(a, b) { return a + b; }"

        # Create a completed task result
        completed_task = Task(
            description=simple_task.description,
            task_id=simple_task.task_id,
            execution_stage=simple_task.execution_stage,
            verification_status=simple_task.verification_status,
            result=simple_task.result,
            status=TaskStatus.COMPLETED,
            completed_at=time.time(),
        )

        # Mock the _handle_preverified_task method
        mock_handle_preverified.return_value = Result(success=True, data=completed_task, error=None)

        # Call iterate_task
        result = await executor_agent.iterate_task(simple_task)

        # Verify the task is marked as completed
        assert result.success
        assert result.data.status == TaskStatus.COMPLETED
        assert result.data.completed_at is not None

    @patch.object(ExecutorAgent, "_process_task_iteration")
    @patch.object(ExecutorAgent, "_evaluate_completion_criteria")
    async def test_task_stage_progression(
        self,
        mock_evaluate: MagicMock,
        mock_process: AsyncMock,
        executor_agent: ExecutorAgent,
        simple_task: Task,
    ) -> None:
        """Test that task progresses through all execution stages."""
        # Mock the _process_task_iteration method to return a successful result
        mock_process.return_value = Result(success=True, data="Stage result", error=None)

        # Mock the _evaluate_completion_criteria method to return False until the final test
        mock_evaluate.return_value = (False, "Not complete yet")

        # Iterate through all stages
        task = simple_task
        stages = [
            ExecutionStage.PLANNING,
            ExecutionStage.IMPLEMENTING,
            ExecutionStage.TESTING,
            ExecutionStage.REFINING,
            ExecutionStage.FINALIZING,
        ]

        # Verify initial stage
        assert task.execution_stage == ExecutionStage.PLANNING

        # Iterate through each stage
        for expected_next_stage in stages[1:]:  # Skip PLANNING as it's the initial stage
            result = await executor_agent.iterate_task(task)
            assert result.success
            task = result.data
            assert task.execution_stage == expected_next_stage

        # After FINALIZING, one more iteration should complete the task if verification is passed
        task.verification_status = VerificationStatus.PASSED

        # Now mock _evaluate_completion_criteria to return True for the final iteration
        mock_evaluate.return_value = (True, "Task meets all completion criteria")

        result = await executor_agent.iterate_task(task)
        assert result.success
        assert result.data.status == TaskStatus.COMPLETED

    @patch.object(ExecutorAgent, "_process_task_iteration")
    @patch.object(ExecutorAgent, "_detect_failure")
    async def test_task_failure_handling(
        self,
        mock_detect_failure: MagicMock,
        mock_process: AsyncMock,
        executor_agent: ExecutorAgent,
        simple_task: Task,
    ) -> None:
        """Test handling of task failures."""
        # Mock the _process_task_iteration method to return a failure
        error_message = "Implementation error"
        mock_process.return_value = Result(success=False, data=None, error=error_message)

        # Mock the _detect_failure method to return our custom error
        mock_detect_failure.return_value = (True, "implementation_error", error_message)

        # Call iterate_task
        result = await executor_agent.iterate_task(simple_task)

        # Verify the result indicates failure but doesn't immediately fail the task
        assert not result.success
        assert error_message in result.error
        assert result.data.execution_attempts == 1
        assert result.data.status == TaskStatus.IN_PROGRESS  # Task should still be in progress for retry

    @patch("src.agent.agent_types.executor.time.time")
    @patch.object(ExecutorAgent, "_process_task_iteration")
    async def test_task_metadata_tracking(
        self,
        mock_process: AsyncMock,
        mock_time: MagicMock,
        executor_agent: ExecutorAgent,
        simple_task: Task,
    ) -> None:
        """Test that task metadata is properly tracked during iteration."""
        # Set up mock time values
        current_time = 1000.0
        mock_time.return_value = current_time

        # Mock the _process_task_iteration method
        mock_process.return_value = Result(success=True, data="Implementation result", error=None)

        # Call iterate_task
        result = await executor_agent.iterate_task(simple_task)

        # Verify metadata tracking
        assert result.success
        assert result.data.created_at == current_time
        assert result.data.updated_at == current_time
        assert len(result.data.execution_logs) > 0
        assert result.data.execution_attempts == 1

    @patch.object(ExecutorAgent, "_is_preverified_task")
    @patch.object(ExecutorAgent, "_handle_preverified_task")
    async def test_pre_verified_task_handling(
        self,
        mock_handle_preverified: AsyncMock,
        mock_is_preverified: MagicMock,
        executor_agent: ExecutorAgent,
    ) -> None:
        """Test handling of pre-verified tasks (for testing support)."""
        # Create a pre-verified task
        pre_verified_task = Task(
            description="Pre-verified task",
            task_id=uuid.uuid4(),
            execution_stage=ExecutionStage.FINALIZING,
            verification_status=VerificationStatus.PASSED,
            execution_attempts=0,  # Important for the _is_preverified_task check
        )

        # Mock the _is_preverified_task method to return True
        mock_is_preverified.return_value = True

        # Create a completed task result
        completed_task = Task(
            description=pre_verified_task.description,
            task_id=pre_verified_task.task_id,
            execution_stage=pre_verified_task.execution_stage,
            verification_status=pre_verified_task.verification_status,
            result="Pre-verified result",
            status=TaskStatus.COMPLETED,
            completed_at=time.time(),
        )

        # Mock the _handle_preverified_task method
        mock_handle_preverified.return_value = Result(success=True, data=completed_task, error=None)

        # Call iterate_task
        result = await executor_agent.iterate_task(pre_verified_task)

        # Verify the task is marked as completed immediately
        assert result.success
        assert result.data.status == TaskStatus.COMPLETED
        assert result.data.result == "Pre-verified result"
        assert result.data.completed_at is not None

    @patch.object(ExecutorAgent, "_process_task_iteration")
    @patch.object(ExecutorAgent, "_evaluate_completion_criteria")
    async def test_multiple_iterations_required(
        self,
        mock_evaluate: MagicMock,
        mock_process: AsyncMock,
        executor_agent: ExecutorAgent,
        simple_task: Task,
    ) -> None:
        """Test scenario where multiple iterations are required to complete a task."""
        # Mock the _process_task_iteration method to return successful results
        mock_process.return_value = Result(success=True, data="Iteration result", error=None)

        # Mock the _evaluate_completion_criteria method to return False until the final test
        mock_evaluate.return_value = (False, "Not complete yet")

        # First iteration - should advance from PLANNING to IMPLEMENTING
        result1 = await executor_agent.iterate_task(simple_task)
        assert result1.success
        assert result1.data.execution_stage == ExecutionStage.IMPLEMENTING
        assert result1.data.execution_attempts == 1

        # Second iteration - should advance to TESTING
        result2 = await executor_agent.iterate_task(result1.data)
        assert result2.success
        assert result2.data.execution_stage == ExecutionStage.TESTING
        assert result2.data.execution_attempts == 2

        # Third iteration - should advance to REFINING
        result3 = await executor_agent.iterate_task(result2.data)
        assert result3.success
        assert result3.data.execution_stage == ExecutionStage.REFINING
        assert result3.data.execution_attempts == 3

        # Fourth iteration - should advance to FINALIZING
        result4 = await executor_agent.iterate_task(result3.data)
        assert result4.success
        assert result4.data.execution_stage == ExecutionStage.FINALIZING
        assert result4.data.execution_attempts == 4

        # Set verification status to PASSED for completion
        result4.data.verification_status = VerificationStatus.PASSED

        # Now mock _evaluate_completion_criteria to return True for the final iteration
        mock_evaluate.return_value = (True, "Task meets all completion criteria")

        # Fifth iteration - should complete the task
        result5 = await executor_agent.iterate_task(result4.data)
        assert result5.success
        assert result5.data.status == TaskStatus.COMPLETED
        assert result5.data.execution_attempts == 5
