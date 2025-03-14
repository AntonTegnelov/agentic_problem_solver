"""Tests for task breakdown step."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.steps import TaskBreakdownStep
from src.common_types.enums import AgentRole
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity, TaskPriority
from src.utils.serialization import serialize_task


class TestTaskBreakdownStep:
    """Test task breakdown step."""

    def test_initialization(self) -> None:
        """Test initialization."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        assert step.name == "task_breakdown"
        assert step.agent_role == AgentRole.ARCHITECT

    def test_validate_inputs_success(self) -> None:
        """Test validate_inputs with valid inputs."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Should not raise an exception
        step._validate_inputs(task_description="Test task")  # noqa: SLF001 - Accessing protected method for testing

    def test_validate_inputs_missing_required(self) -> None:
        """Test validate_inputs with missing required keys."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        with pytest.raises(ValueError, match="Missing required keys: task_description"):
            step._validate_inputs()  # noqa: SLF001 - Accessing protected method for testing

    def test_create_task_breakdown_prompt(self) -> None:
        """Test create_task_breakdown_prompt."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Test with all parameters
        prompt = step._create_task_breakdown_prompt(  # noqa: SLF001 - Accessing protected method for testing
            task_description="Test task",
            complexity=TaskComplexity.MODERATE,
            priority=TaskPriority.MEDIUM,
        )

        assert isinstance(prompt, str)
        assert "Test task" in prompt
        assert "moderate" in prompt
        assert "medium" in prompt

        # Test with minimal parameters
        prompt = step._create_task_breakdown_prompt(  # noqa: SLF001 - Accessing protected method for testing
            task_description="Test task",
        )

        assert isinstance(prompt, str)
        assert "Test task" in prompt

    def test_parse_tasks_from_result_string(self) -> None:
        """Test parse_tasks_from_result with string result."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

        # Create a sample JSON result with some text before and after
        json_result = """
        Some text before the JSON.
        ```json
        [
          {
            "description": "Task 1",
            "complexity": "simple",
            "priority": "high",
            "dependencies": []
          },
          {
            "description": "Task 2",
            "complexity": "moderate",
            "priority": "medium",
            "dependencies": [
              {
                "task_index": 0,
                "description": "Depends on Task 1",
                "is_blocking": true
              }
            ]
          }
        ]
        ```
        Some text after the JSON.
        """

        tasks = step._parse_tasks_from_result(json_result, parent_task_id)  # noqa: SLF001 - Accessing protected method for testing

        assert len(tasks) == 2
        assert tasks[0].description == "Task 1"
        assert tasks[0].complexity == TaskComplexity.SIMPLE
        assert tasks[0].priority == TaskPriority.HIGH
        assert tasks[0].parent_task_id == parent_task_id
        assert tasks[0].assigned_role == AgentRole.ARCHITECT

        assert tasks[1].description == "Task 2"
        assert tasks[1].complexity == TaskComplexity.MODERATE
        assert tasks[1].priority == TaskPriority.MEDIUM
        assert tasks[1].parent_task_id == parent_task_id
        assert tasks[1].assigned_role == AgentRole.ARCHITECT

    def test_parse_tasks_from_result_list(self) -> None:
        """Test parse_tasks_from_result with list result."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

        # Create a sample list result
        list_result = [
            {
                "description": "Task 1",
                "complexity": "simple",
                "priority": "high",
                "dependencies": [],
            },
            {
                "description": "Task 2",
                "complexity": "moderate",
                "priority": "medium",
                "dependencies": [
                    {
                        "task_index": 0,
                        "description": "Depends on Task 1",
                        "is_blocking": True,
                    },
                ],
            },
        ]

        tasks = step._parse_tasks_from_result(list_result, parent_task_id)  # noqa: SLF001 - Accessing protected method for testing

        assert len(tasks) == 2
        assert tasks[0].description == "Task 1"
        assert tasks[0].complexity == TaskComplexity.SIMPLE
        assert tasks[0].priority == TaskPriority.HIGH
        assert tasks[0].parent_task_id == parent_task_id
        assert tasks[0].assigned_role == AgentRole.ARCHITECT

        assert tasks[1].description == "Task 2"
        assert tasks[1].complexity == TaskComplexity.MODERATE
        assert tasks[1].priority == TaskPriority.MEDIUM
        assert tasks[1].parent_task_id == parent_task_id
        assert tasks[1].assigned_role == AgentRole.ARCHITECT

    def test_parse_tasks_from_result_invalid_json(self) -> None:
        """Test parse_tasks_from_result with invalid JSON."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

        invalid_json = "This is not JSON"

        with pytest.raises(ValueError, match="No JSON array found in result"):
            step._parse_tasks_from_result(invalid_json, parent_task_id)  # noqa: SLF001 - Accessing protected method for testing

        invalid_json = "[This is invalid JSON]"

        with pytest.raises(ValueError, match="Failed to parse JSON from result"):
            step._parse_tasks_from_result(invalid_json, parent_task_id)  # noqa: SLF001 - Accessing protected method for testing

    def test_store_task_in_state(self) -> None:
        """Test store_task_in_state."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        mock_state = MagicMock()
        mock_state.get_context.return_value = []

        # Create a task
        task = Task(
            description="Test task",
            complexity=TaskComplexity.MODERATE,
            priority=TaskPriority.MEDIUM,
            assigned_role=AgentRole.ARCHITECT,
        )

        # Store task in state
        step._store_task_in_state(mock_state, task)  # noqa: SLF001 - Accessing protected method for testing

        # Verify state was updated
        mock_state.get_context.assert_called_once_with("tasks", [])
        mock_state.set_context.assert_called_once()

        # Check that the task was added to the list
        args, _ = mock_state.set_context.call_args
        assert args[0] == "tasks"
        assert len(args[1]) == 1
        assert args[1][0] == serialize_task(task)

    def test_update_parent_task_with_subtasks(self) -> None:
        """Test update_parent_task_with_subtasks."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        subtasks = [
            Task(description="Task 1"),
            Task(description="Task 2"),
        ]

        # Create a mock state with existing tasks
        mock_state = MagicMock()
        mock_state.get_context.return_value = [
            {"task_id": str(parent_task_id), "subtasks": []},
        ]

        # Update parent task
        step._update_parent_task_with_subtasks(mock_state, parent_task_id, subtasks)  # noqa: SLF001 - Accessing protected method for testing

        # Verify state was updated
        mock_state.get_context.assert_called_once_with("tasks", [])
        mock_state.set_context.assert_called_once()

        # Check that the subtasks were added to the parent task
        args, _ = mock_state.set_context.call_args
        assert args[0] == "tasks"
        assert len(args[1]) == 1
        assert args[1][0]["task_id"] == str(parent_task_id)
        assert len(args[1][0]["subtasks"]) == 2
        assert args[1][0]["subtasks"] == [str(subtask.task_id) for subtask in subtasks]

    @pytest.mark.asyncio
    @patch.object(TaskBreakdownStep, "_store_task_in_state")
    @patch.object(TaskBreakdownStep, "_update_parent_task_with_subtasks")
    @patch.object(TaskBreakdownStep, "_parse_tasks_from_result")
    async def test_call_success(
        self,
        mock_parse_tasks: MagicMock,
        mock_update_parent: MagicMock,
        mock_store_task: MagicMock,
    ) -> None:
        """Test __call__ with successful execution."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        mock_state = MagicMock()
        mock_agent = AsyncMock()
        mock_state.get_agent_for_step.return_value = mock_agent

        # Mock agent.process to return a successful result
        process_mock = AsyncMock()
        process_mock.return_value = Result(success=True, data="[{}]")
        mock_agent.process = process_mock

        # Mock parse_tasks_from_result to return a list of tasks
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        mock_parse_tasks.return_value = [task1, task2]

        # Call the step
        result = await step(
            mock_state,
            task_description="Test task",
            complexity=TaskComplexity.MODERATE,
            priority=TaskPriority.MEDIUM,
        )

        assert result.success
        assert result.data == [task1, task2]

        # Verify that the tasks were stored and parent task was updated
        assert mock_store_task.call_count == 2
        mock_store_task.assert_any_call(mock_state, task1)
        mock_store_task.assert_any_call(mock_state, task2)
        mock_update_parent.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_agent_failure(self) -> None:
        """Test __call__ with agent failure."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        mock_state = MagicMock()
        mock_agent = AsyncMock()
        mock_state.get_agent_for_step.return_value = mock_agent

        # Mock agent.process to return a failed result
        error_msg = "Agent processing failed"
        process_mock = AsyncMock()
        process_mock.return_value = Result(success=False, error=error_msg)
        mock_agent.process = process_mock

        # Call the step
        result = await step(mock_state, task_description="Test task")

        assert not result.success
        assert result.error == f"Agent failed: {error_msg}"

    @pytest.mark.asyncio
    @patch.object(TaskBreakdownStep, "_parse_tasks_from_result")
    async def test_call_parsing_failure(self, mock_parse_tasks: MagicMock) -> None:
        """Test __call__ with parsing failure."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        mock_state = MagicMock()
        mock_agent = AsyncMock()
        mock_state.get_agent_for_step.return_value = mock_agent

        # Mock agent.process to return a successful result
        process_mock = AsyncMock()
        process_mock.return_value = Result(success=True, data="[{}]")
        mock_agent.process = process_mock

        # Mock parse_tasks_from_result to raise an exception
        error_msg = "Parsing failed"
        mock_parse_tasks.side_effect = ValueError(error_msg)

        # Call the step
        result = await step(mock_state, task_description="Test task")

        assert not result.success
        assert result.error == error_msg
