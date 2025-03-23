"""Unit tests for task timeout handling functionality."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.agent.agent_types.architect import ArchitectAgent
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity
from src.config.agent import AgentConfig
from src.llm_providers.interface import LLMProvider


class TestTaskTimeoutHandling:
    """Tests for task timeout handling in delegation."""

    @pytest.mark.asyncio
    async def test_task_delegation_timeout(self) -> None:
        """Test that task delegation handles timeouts correctly."""
        # Create a mock provider that will never complete
        mock_provider = MagicMock(spec=LLMProvider)

        # Configure a short timeout for testing
        config = AgentConfig(task_timeout=1)  # Minimum allowed timeout is 1 second

        # Create an architect agent with the mock provider
        architect = ArchitectAgent(provider=mock_provider, config=config)

        # Create a task that will time out
        task = Task(description="This task will time out", complexity=TaskComplexity.SIMPLE)

        # Create a mock delegate_to_executor that never completes (sleeps forever)
        async def mock_delegate_that_sleeps_forever(*_: str, **__: dict) -> Result:
            await asyncio.sleep(1000)
            return Result.success("This should never be reached")

        # Patch the delegate_to_executor method
        with patch.object(architect, "delegate_to_executor", mock_delegate_that_sleeps_forever):
            # Call _delegate_single_task which should timeout
            result_data, should_retry, error_msg = await architect._delegate_single_task(task)

            # Check that the timeout was handled correctly
            assert result_data is None
            assert should_retry is True  # Timeouts should be marked for retry
            assert "timed out after 1 seconds" in error_msg
            assert "This task will time out" in error_msg

    @pytest.mark.asyncio
    async def test_task_delegation_completes_within_timeout(self) -> None:
        """Test that task delegation completes successfully within the timeout."""
        # Create a mock provider
        mock_provider = MagicMock(spec=LLMProvider)

        # Configure a reasonable timeout for testing
        config = AgentConfig(task_timeout=5)  # 5 second timeout

        # Create an architect agent with the mock provider
        architect = ArchitectAgent(provider=mock_provider, config=config)

        # Create a task
        task = Task(description="This task should complete", complexity=TaskComplexity.SIMPLE)

        # Create a mock delegate_to_executor that completes quickly with success
        async def mock_delegate_that_completes(*_: str, **__: dict) -> Result:
            return Result.success("Task completed successfully")

        # Patch the delegate_to_executor method
        with patch.object(architect, "delegate_to_executor", mock_delegate_that_completes):
            # Call _delegate_single_task which should complete within timeout
            result_data, should_retry, error_msg = await architect._delegate_single_task(task)

            # Check that the task completed successfully
            assert result_data == "Task completed successfully"
            assert should_retry is False
            assert error_msg == ""
