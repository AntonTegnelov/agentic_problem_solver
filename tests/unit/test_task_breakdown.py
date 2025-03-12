"""Tests for task breakdown step."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.agent.steps import TaskBreakdownStep
from src.common_types.enums import AgentRole, AgentStep
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity, TaskPriority


class TestTaskBreakdownStep:
    """Test task breakdown step."""

    def test_initialization(self) -> None:
        """Test TaskBreakdownStep initialization."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        assert step.agent_role == AgentRole.ARCHITECT
        assert step.name == "task_breakdown"
        assert step.required_keys == ["task_description"]
        assert "parent_task_id" in step.optional_keys
        assert "complexity" in step.optional_keys
        assert "priority" in step.optional_keys
        assert step.retry_on_error is True
        assert step.max_retries == 3

    def test_validate_inputs_success(self) -> None:
        """Test validate_inputs with valid inputs."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        # Should not raise an exception
        step.validate_inputs(task_description="Test task")

    def test_validate_inputs_missing_required(self) -> None:
        """Test validate_inputs with missing required inputs."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        with pytest.raises(ValueError, match="Missing required keys: task_description"):
            step.validate_inputs()

    def test_create_task_breakdown_prompt(self) -> None:
        """Test create_task_breakdown_prompt."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        prompt = step.create_task_breakdown_prompt(
            task_description="Test task",
            complexity=TaskComplexity.MODERATE,
            priority=TaskPriority.MEDIUM,
        )
        assert "Test task" in prompt
        assert "moderate" in prompt.lower()
        assert "medium" in prompt.lower()
        assert "subtasks" in prompt.lower()
        assert "json" in prompt.lower()

    @patch("uuid.uuid4")
    def test_parse_tasks_from_result_string(self, mock_uuid4: MagicMock) -> None:
        """Test parse_tasks_from_result with string result."""
        # Mock UUID generation for predictable results
        mock_uuid4.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")

        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

        # Create a sample JSON result
        json_result = """
        Some text before the JSON.
        ```json
        [
          {
            "description": "Task 1",
            "complexity": "SIMPLE",
            "priority": "HIGH",
            "dependencies": []
          },
          {
            "description": "Task 2",
            "complexity": "MODERATE",
            "priority": "MEDIUM",
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

        tasks = step.parse_tasks_from_result(json_result, parent_task_id)

        assert len(tasks) == 2
        assert tasks[0].description == "Task 1"
        assert tasks[0].complexity == TaskComplexity.SIMPLE
        assert tasks[0].priority == TaskPriority.HIGH
        assert tasks[0].parent_task_id == parent_task_id
        assert tasks[0].dependencies == []

        assert tasks[1].description == "Task 2"
        assert tasks[1].complexity == TaskComplexity.MODERATE
        assert tasks[1].priority == TaskPriority.MEDIUM
        assert tasks[1].parent_task_id == parent_task_id
        assert len(tasks[1].dependencies) == 1
        assert tasks[1].dependencies[0].description == "Depends on Task 1"
        assert tasks[1].dependencies[0].is_blocking is True

    def test_parse_tasks_from_result_list(self) -> None:
        """Test parse_tasks_from_result with list result."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

        # Create a sample list result
        list_result = [
            {
                "description": "Task 1",
                "complexity": "SIMPLE",
                "priority": "HIGH",
                "dependencies": [],
            },
            {
                "description": "Task 2",
                "complexity": "MODERATE",
                "priority": "MEDIUM",
                "dependencies": [
                    {
                        "task_index": 0,
                        "description": "Depends on Task 1",
                        "is_blocking": True,
                    },
                ],
            },
        ]

        tasks = step.parse_tasks_from_result(list_result, parent_task_id)

        assert len(tasks) == 2
        assert tasks[0].description == "Task 1"
        assert tasks[1].description == "Task 2"
        assert len(tasks[1].dependencies) == 1

    def test_parse_tasks_from_result_invalid_json(self) -> None:
        """Test parse_tasks_from_result with invalid JSON."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

        invalid_json = "This is not JSON"

        with pytest.raises(ValueError, match="No JSON array found in result"):
            step.parse_tasks_from_result(invalid_json, parent_task_id)

        invalid_json = "[This is invalid JSON]"

        with pytest.raises(ValueError, match="Invalid JSON in result"):
            step.parse_tasks_from_result(invalid_json, parent_task_id)

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
        step.store_task_in_state(mock_state, task)

        # Verify state was updated
        mock_state.get_context.assert_called_once_with("tasks", [])
        mock_state.set_context.assert_called_once()

        # Check that the task was added to the list
        args, _ = mock_state.set_context.call_args
        assert args[0] == "tasks"
        assert len(args[1]) == 1
        assert args[1][0]["description"] == "Test task"
        assert args[1][0]["task_id"] == str(task.task_id)

    def test_update_parent_task_with_subtasks(self) -> None:
        """Test update_parent_task_with_subtasks."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        parent_task_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        subtask_ids = [
            uuid.UUID("11111111-1111-1111-1111-111111111111"),
            uuid.UUID("22222222-2222-2222-2222-222222222222"),
        ]

        # Create a mock state with existing tasks
        mock_state = MagicMock()
        mock_state.get_context.return_value = [
            {"task_id": str(parent_task_id), "subtasks": []},
        ]

        # Update parent task
        step.update_parent_task_with_subtasks(mock_state, parent_task_id, subtask_ids)

        # Verify state was updated
        mock_state.get_context.assert_called_once_with("tasks", [])
        mock_state.set_context.assert_called_once()

        # Check that the subtasks were added to the parent task
        args, _ = mock_state.set_context.call_args
        assert args[0] == "tasks"
        assert len(args[1]) == 1
        assert args[1][0]["task_id"] == str(parent_task_id)
        assert args[1][0]["subtasks"] == [str(subtask_id) for subtask_id in subtask_ids]

    @patch.object(TaskBreakdownStep, "store_task_in_state")
    @patch.object(TaskBreakdownStep, "update_parent_task_with_subtasks")
    @patch.object(TaskBreakdownStep, "parse_tasks_from_result")
    def test_call_success(
        self,
        mock_parse_tasks: MagicMock,
        mock_update_parent: MagicMock,
        mock_store_task: MagicMock,
    ) -> None:
        """Test __call__ with successful execution."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        mock_state = MagicMock()
        mock_agent = MagicMock()
        mock_state.get_agent_for_step.return_value = mock_agent

        # Mock agent.process to return a successful result
        mock_agent.process.return_value = Result(success=True, data="[{}]")

        # Mock parse_tasks_from_result to return a list of tasks
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        mock_parse_tasks.return_value = [task1, task2]

        # Call the step
        result = step(
            mock_state,
            task_description="Test task",
            complexity=TaskComplexity.MODERATE,
            priority=TaskPriority.MEDIUM,
        )

        # Verify the result
        assert result.success is True
        assert result.data == [task1, task2]

        # Verify the agent was called
        mock_state.get_agent_for_step.assert_called_once_with(AgentStep.UNDERSTAND)
        mock_agent.process.assert_called_once()

        # Verify tasks were stored and parent was updated
        assert mock_store_task.call_count >= 1  # Parent task + subtasks
        mock_update_parent.assert_called_once()

    def test_call_agent_failure(self) -> None:
        """Test __call__ with agent failure."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        mock_state = MagicMock()
        mock_agent = MagicMock()
        mock_state.get_agent_for_step.return_value = mock_agent

        # Mock agent.process to return a failed result
        error_msg = "Agent processing failed"
        mock_agent.process.return_value = Result(success=False, error=error_msg)

        # Call the step
        result = step(mock_state, task_description="Test task")

        # Verify the result
        assert result.success is False
        assert f"Agent failed: {error_msg}" == result.error

    @patch.object(TaskBreakdownStep, "parse_tasks_from_result")
    def test_call_parsing_failure(self, mock_parse_tasks: MagicMock) -> None:
        """Test __call__ with parsing failure."""
        step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

        # Create a mock state
        mock_state = MagicMock()
        mock_agent = MagicMock()
        mock_state.get_agent_for_step.return_value = mock_agent

        # Mock agent.process to return a successful result
        mock_agent.process.return_value = Result(success=True, data="[{}]")

        # Mock parse_tasks_from_result to raise an exception
        error_msg = "Parsing failed"
        mock_parse_tasks.side_effect = ValueError(error_msg)

        # Call the step
        result = step(mock_state, task_description="Test task")

        # Verify the result
        assert result.success is False
        assert f"Failed to parse tasks: {error_msg}" in result.error
