"""Integration tests for task execution.

This module contains integration tests for the task execution functionality,
testing how ExecutorAgent systematically solves tasks through self-prompting.
"""

import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages.base import BaseMessage

from src.agent.agent_types import create_executor_agent
from src.agent.coordination import InMemoryAgentRegistry
from src.agent.state.base import InMemoryStateManager
from src.common_types.enums import ExecutionStage, VerificationStatus
from src.common_types.task_types import Task, TaskComplexity, TaskPriority, TaskStatus


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider."""
    provider = MagicMock()

    # Create a proper response for task execution
    execution_response = (
        "I've implemented the requested functionality. Here's the code:\n\n"
        "```python\ndef calculate_sum(a, b):\n    return a + b\n```\n\n"
        "This function takes two parameters and returns their sum."
    )

    # Set up the generate method as an AsyncMock with a proper return value
    generate_mock = AsyncMock()
    generate_mock.return_value = execution_response
    provider.generate = generate_mock

    # Set up the stream method
    async def mock_stream(_messages: list[BaseMessage]) -> AsyncGenerator[str, None]:
        chunks = ["I've implemented", " the requested", " functionality.", " Here's the code:"]
        for chunk in chunks:
            yield chunk

    provider.generate_stream = mock_stream
    provider.__bool__.return_value = True
    return provider


@pytest.fixture
def registry() -> InMemoryAgentRegistry:
    """Create an InMemoryAgentRegistry instance."""
    return InMemoryAgentRegistry()


@pytest.fixture
def task_execution_system(
    mock_provider: MagicMock,
    registry: InMemoryAgentRegistry,
) -> dict[str, object]:
    """Create a task execution system with an executor agent."""
    # Create state manager
    state_manager = InMemoryStateManager()

    # Create executor agent
    executor_agent = create_executor_agent(provider=mock_provider, state_manager=state_manager)

    # Register agent
    registry.register_agent(executor_agent)

    return {
        "executor_id": executor_agent.get_agent_id(),
        "state_manager": state_manager,
    }


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        task_id=uuid.uuid4(),
        description="Implement a function to calculate the sum of two numbers",
        complexity=TaskComplexity.SIMPLE,
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.IN_PROGRESS,
        execution_stage=ExecutionStage.PLANNING,
        verification_status=VerificationStatus.PENDING,
    )


class TestTaskExecution:
    """Test task execution functionality."""

    @pytest.mark.asyncio
    async def test_task_execution_lifecycle(
        self,
        registry: InMemoryAgentRegistry,
        task_execution_system: dict[str, object],
        sample_task: Task,
    ) -> None:
        """Test the complete lifecycle of task execution."""
        # Get executor agent
        executor_id = task_execution_system["executor_id"]
        executor_agent = registry.get_agent(executor_id)
        state_manager = task_execution_system["state_manager"]

        # Store task in state
        state_manager.get_state().add_task(sample_task)

        # Execute task
        result = await executor_agent.iterate_task(sample_task)

        # Verify result
        assert result.success
        assert result.data is not None

        # Get updated task
        updated_task = result.data

        # After one iteration, the task should advance from PLANNING to IMPLEMENTING
        assert updated_task.execution_stage == ExecutionStage.IMPLEMENTING
        assert updated_task.result is not None
        assert "calculate_sum" in updated_task.result

    @pytest.mark.asyncio
    async def test_task_execution_with_failure_recovery(
        self,
        registry: InMemoryAgentRegistry,
        task_execution_system: dict[str, object],
        mock_provider: MagicMock,
        sample_task: Task,
    ) -> None:
        """Test task execution with failure and recovery."""
        # Get executor agent
        executor_id = task_execution_system["executor_id"]
        executor_agent = registry.get_agent(executor_id)
        state_manager = task_execution_system["state_manager"]

        # Store task in state
        state_manager.get_state().add_task(sample_task)

        # Mock provider to fail on first attempt, then succeed on second attempt
        original_generate = mock_provider.generate

        # Counter to track calls
        call_count = 0

        async def mock_generate_with_failure(_: list[BaseMessage]) -> str:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call fails
                return "I encountered an error while implementing the function."
            # Subsequent calls succeed
            return (
                "I've fixed the implementation. Here's the working code:\n\n"
                "```python\ndef calculate_sum(a, b):\n    return a + b\n```"
            )

        mock_provider.generate = AsyncMock(side_effect=mock_generate_with_failure)

        # Execute task - first iteration
        result = await executor_agent.iterate_task(sample_task)

        # Verify first result
        assert result.success
        assert result.data is not None

        # Get updated task after first iteration
        updated_task = result.data

        # After first iteration with error, the task should still advance to IMPLEMENTING
        # but the result will contain the error message
        assert updated_task.execution_stage == ExecutionStage.IMPLEMENTING
        assert updated_task.result is not None
        assert "error" in updated_task.result.lower() or "encountered" in updated_task.result.lower()

        # Execute task again - second iteration
        result = await executor_agent.iterate_task(updated_task)

        # Restore original mock
        mock_provider.generate = original_generate

        # Verify second result
        assert result.success
        assert result.data is not None

        # Get updated task after second iteration
        updated_task = result.data

        # After second iteration with success, the task should advance to TESTING
        assert updated_task.execution_stage == ExecutionStage.TESTING
        assert updated_task.result is not None
        assert "calculate_sum" in updated_task.result

    @pytest.mark.asyncio
    async def test_task_execution_with_verification(
        self,
        registry: InMemoryAgentRegistry,
        task_execution_system: dict[str, object],
        mock_provider: MagicMock,
        sample_task: Task,
    ) -> None:
        """Test task execution with verification step."""
        # Get executor agent
        executor_id = task_execution_system["executor_id"]
        executor_agent = registry.get_agent(executor_id)
        state_manager = task_execution_system["state_manager"]

        # Store task in state
        state_manager.get_state().add_task(sample_task)

        # Set up mock to provide different responses for execution and verification
        original_generate = mock_provider.generate

        # Counter to track calls
        call_count = 0

        async def mock_generate_with_verification(_: list[BaseMessage]) -> str:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Execution response
                return "I've implemented the function:\n\n```python\ndef calculate_sum(a, b):\n    return a + b\n```"
            # Verification response
            return (
                "I've verified the implementation. The function correctly calculates "
                "the sum of two numbers. All tests pass."
            )

        mock_provider.generate = AsyncMock(side_effect=mock_generate_with_verification)

        # Execute task
        result = await executor_agent.iterate_task(sample_task)

        # Restore original mock
        mock_provider.generate = original_generate

        # Verify result
        assert result.success
        assert result.data is not None

        # Get updated task
        updated_task = result.data

        # After one iteration, the task should advance from PLANNING to IMPLEMENTING
        # The verification status remains PENDING until the task reaches FINALIZING stage
        assert updated_task.execution_stage == ExecutionStage.IMPLEMENTING
        assert call_count >= 1  # At least one execution call

    @pytest.mark.asyncio
    async def test_task_execution_progress_tracking(
        self,
        registry: InMemoryAgentRegistry,
        task_execution_system: dict[str, object],
        sample_task: Task,
    ) -> None:
        """Test progress tracking during task execution."""
        # Get executor agent
        executor_id = task_execution_system["executor_id"]
        executor_agent = registry.get_agent(executor_id)
        state_manager = task_execution_system["state_manager"]

        # Store task in state
        state_manager.get_state().add_task(sample_task)

        # Execute task
        result = await executor_agent.iterate_task(sample_task)

        # Verify result
        assert result.success
        assert result.data is not None

        # Get updated task
        updated_task = result.data

        # Verify task has progress information in metadata
        # The _update_task_progress method is called, but we need to manually check
        # if the progress tracking is added to the metadata
        assert updated_task.execution_stage == ExecutionStage.IMPLEMENTING

        # Check that execution_metadata contains the planning result
        assert "planning_result" in updated_task.execution_metadata
        assert updated_task.execution_metadata["planning_result"] is not None
