"""Tests for automatic retries in task delegation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.planner import PlannerAgent
from src.common_types.task_types import Task


@pytest.fixture
def architect_agent():
    """Create a test architect agent with configured mocks."""
    agent = ArchitectAgent(provider=AsyncMock())
    agent._logger = MagicMock()
    return agent


@pytest.fixture
def planner_agent():
    """Create a test planner agent with configured mocks."""
    agent = PlannerAgent(provider=AsyncMock())
    agent.logger = MagicMock()
    return agent


class TestAutomaticRetries:
    """Test cases for automatic retries functionality."""

    @pytest.mark.asyncio
    async def test_architect_retry_delegation_success_after_retry(self, architect_agent) -> None:
        """Test that architect successfully retries a failed delegation."""
        # Patch the _delegate_single_task method to fail once then succeed
        original_delegate = architect_agent._delegate_single_task
        call_count = 0

        async def mock_delegate(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None, True, "Temporary failure"
            return "Success after retry", False, ""

        architect_agent._delegate_single_task = mock_delegate

        # Create a test task
        task = Task(description="Test automatic retry")

        try:
            # Call the retry method
            result_data, is_retry, error_msg = await architect_agent.retry_delegation_until_success(
                task,
                max_retries=3,
                retry_delay=0.1,
            )

            # Verify the results
            assert call_count == 2
            assert result_data == "Success after retry"
            assert not is_retry
            assert error_msg == ""

            # Verify the logging
            assert architect_agent._logger.info.call_count == 1
            architect_agent._logger.info.assert_called_with(
                "Retry attempt %d/%d for task '%s...'",
                1,
                3,
                "Test automatic retry",
            )
        finally:
            # Restore original method
            architect_agent._delegate_single_task = original_delegate

    @pytest.mark.asyncio
    async def test_architect_retry_delegation_max_retries_reached(self, architect_agent) -> None:
        """Test that architect stops retrying after reaching max retries."""
        # Patch the _delegate_single_task method to always fail
        original_delegate = architect_agent._delegate_single_task
        call_count = 0

        async def mock_delegate(task):
            nonlocal call_count
            call_count += 1
            return None, True, "Persistent failure"

        architect_agent._delegate_single_task = mock_delegate

        # Create a test task
        task = Task(description="Test max retries")

        try:
            # Call the retry method with 2 max retries
            result_data, is_retry, error_msg = await architect_agent.retry_delegation_until_success(
                task,
                max_retries=2,
                retry_delay=0.1,
            )

            # Verify the results
            assert call_count == 3  # Initial attempt + 2 retries
            assert result_data is None
            assert not is_retry  # Should be False after max retries reached
            assert "Max retries (2) reached" in error_msg

            # Verify the logging
            assert architect_agent._logger.info.call_count == 2  # Two retry attempts logged
            assert architect_agent._logger.warning.call_count == 1  # Max retries warning
        finally:
            # Restore original method
            architect_agent._delegate_single_task = original_delegate

    @pytest.mark.asyncio
    async def test_planner_retry_delegation_success_after_retry(self, planner_agent) -> None:
        """Test that planner successfully retries a failed delegation."""
        # Patch the _delegate_single_task method to fail once then succeed
        original_delegate = planner_agent._delegate_single_task
        call_count = 0

        async def mock_delegate(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None, True, "Temporary failure"
            return "Success after retry", False, ""

        planner_agent._delegate_single_task = mock_delegate

        # Create a test task
        task = Task(description="Test automatic retry")

        try:
            # Call the retry method
            result_data, is_error, error_msg = await planner_agent.retry_delegation_until_success(
                task,
                max_retries=3,
                retry_delay=0.1,
            )

            # Verify the results
            assert call_count == 2
            assert result_data == "Success after retry"
            assert not is_error
            assert error_msg == ""

            # Verify the logging
            assert planner_agent.logger.info.call_count == 1
            planner_agent.logger.info.assert_called_with(
                "Retry attempt %d/%d for task '%s...'",
                1,
                3,
                "Test automatic retry",
            )
        finally:
            # Restore original method
            planner_agent._delegate_single_task = original_delegate

    @pytest.mark.asyncio
    async def test_planner_retry_delegation_max_retries_reached(self, planner_agent) -> None:
        """Test that planner stops retrying after reaching max retries."""
        # Patch the _delegate_single_task method to always fail
        original_delegate = planner_agent._delegate_single_task
        call_count = 0

        async def mock_delegate(task):
            nonlocal call_count
            call_count += 1
            return None, True, "Persistent failure"

        planner_agent._delegate_single_task = mock_delegate

        # Create a test task
        task = Task(description="Test max retries")

        try:
            # Call the retry method with 2 max retries
            result_data, is_error, error_msg = await planner_agent.retry_delegation_until_success(
                task,
                max_retries=2,
                retry_delay=0.1,
            )

            # Verify the results
            assert call_count == 3  # Initial attempt + 2 retries
            assert result_data is None
            assert not is_error  # Should be False after max retries reached
            assert "Max retries (2) reached" in error_msg

            # Verify the logging
            assert planner_agent.logger.info.call_count == 2  # Two retry attempts logged
            assert planner_agent.logger.warning.call_count == 1  # Max retries warning
        finally:
            # Restore original method
            planner_agent._delegate_single_task = original_delegate
