"""Tests for the TaskResultAggregator class."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.agent.coordination import TaskResultAggregator
from src.common_types.result_types import Result
from src.common_types.task_types import TaskStatus


class TestTaskResultAggregator:
    """Tests for the TaskResultAggregator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_registry = MagicMock()
        self.aggregator = TaskResultAggregator(self.mock_registry)

    def test_initialization(self) -> None:
        """Test initialization of the TaskResultAggregator."""
        assert self.aggregator._registry == self.mock_registry

    def test_collect_results_agent_not_found(self) -> None:
        """Test collect_results when agent is not found."""
        from src.common_types import AgentNotFoundError

        self.mock_registry.get_agent.side_effect = AgentNotFoundError("Agent not found")
        result = self.aggregator.collect_results("agent1")

        assert not result.success
        assert isinstance(result.error, AgentNotFoundError)
        assert "Agent not found" in str(result.error)

    def test_collect_results_no_state(self) -> None:
        """Test collect_results when agent has no state."""
        mock_agent = MagicMock()
        mock_agent.get_state.return_value = None
        self.mock_registry.get_agent.return_value = mock_agent

        result = self.aggregator.collect_results("agent1")

        assert not result.success
        assert isinstance(result.error, ValueError)
        assert "Cannot collect results without parent state" in result.message

    def test_collect_results_no_tasks(self) -> None:
        """Test collect_results when there are no tasks."""
        mock_agent = MagicMock()
        mock_state = MagicMock()
        mock_state.get_tasks.return_value = []
        mock_agent.get_state.return_value = mock_state
        self.mock_registry.get_agent.return_value = mock_agent

        result = self.aggregator.collect_results("agent1")

        assert result.success
        assert result.message == "No results to collect"
        assert result.data == {"message": "No tasks found for collection"}

    def test_collect_results_success(self) -> None:
        """Test collect_results with successful task collection."""
        mock_agent = MagicMock()
        mock_state = MagicMock()

        # Create mock tasks
        task_id = uuid.uuid4()
        task_data = {"task_id": str(task_id)}
        mock_state.get_tasks.return_value = [task_data]

        # Create mock task object
        mock_task = MagicMock()
        mock_task.task_id = task_id
        mock_task.description = "Test task"
        mock_task.status = TaskStatus.COMPLETED
        mock_task.result = "Task result"
        mock_task.assigned_agent_id = "executor1"

        mock_state.get_task_by_id.return_value = mock_task
        mock_agent.get_state.return_value = mock_state
        self.mock_registry.get_agent.return_value = mock_agent

        result = self.aggregator.collect_results("agent1")

        assert result.success
        assert result.message == "Successfully collected results from subtasks"
        assert len(result.data["results"]) == 1
        assert result.data["results"][0]["task_id"] == str(task_id)
        assert result.data["results"][0]["description"] == "Test task"
        assert result.data["results"][0]["status"] == TaskStatus.COMPLETED.value
        assert result.data["results"][0]["result"] == "Task result"
        assert result.data["results"][0]["assigned_agent_id"] == "executor1"

    def test_merge_results_empty(self) -> None:
        """Test merge_results with empty results list."""
        result = self.aggregator.merge_results([])

        assert result.success
        assert result.message == "No results to merge"
        assert result.data is None

    def test_merge_default_results(self) -> None:
        """Test merge_results with default strategy."""
        results = [
            {"task_id": "task1", "description": "Task 1", "status": "completed", "result": "Result 1"},
            {"task_id": "task2", "description": "Task 2", "status": "completed", "result": "Result 2"},
        ]

        result = self.aggregator.merge_results(results)

        assert result.success
        assert result.message == "Successfully merged results using default strategy"
        assert "task1" in result.data
        assert "task2" in result.data
        assert result.data["task1"]["result"] == "Result 1"
        assert result.data["task2"]["result"] == "Result 2"

    def test_merge_text_results(self) -> None:
        """Test merge_results with text strategy."""
        results = [
            {"task_id": "task1", "result": "Text result 1"},
            {"task_id": "task2", "result": "Text result 2"},
        ]

        result = self.aggregator.merge_results(results, result_type="text")

        assert result.success
        assert result.message == "Successfully merged text results"
        assert result.data == "Text result 1\n\nText result 2"

    def test_merge_text_results_with_dict(self) -> None:
        """Test merge_results with text strategy and dict results."""
        results = [
            {"task_id": "task1", "result": {"text": "Text in dict 1"}},
            {"task_id": "task2", "result": {"text": "Text in dict 2"}},
        ]

        result = self.aggregator.merge_results(results, result_type="text")

        assert result.success
        assert result.data == "Text in dict 1\n\nText in dict 2"

    def test_merge_code_results(self) -> None:
        """Test merge_results with code strategy."""
        results = [
            {"task_id": "task1", "result": {"code": "def func1():\n    pass", "file_path": "file1.py"}},
            {"task_id": "task2", "result": {"code": "def func2():\n    pass", "file_path": "file2.py"}},
        ]

        result = self.aggregator.merge_results(results, result_type="code")

        assert result.success
        assert result.message == "Successfully merged code results"
        assert "file1.py" in result.data["code_sections"]
        assert "file2.py" in result.data["code_sections"]
        assert result.data["code_sections"]["file1.py"] == "def func1():\n    pass"
        assert result.data["code_sections"]["file2.py"] == "def func2():\n    pass"

    def test_merge_results_exception(self) -> None:
        """Test merge_results with an exception."""
        with patch.object(self.aggregator, "_merge_default_results", side_effect=ValueError("Test error")):
            result = self.aggregator.merge_results([{"task_id": "task1"}])

            assert not result.success
            assert isinstance(result.error, ValueError)
            assert "Test error" in result.message

    def test_track_completion_status(self) -> None:
        """Test track_completion_status with successful collection."""
        # Mock collect_results to return a successful result
        mock_collection_result = Result.success(
            data={
                "results": [
                    {"task_id": "task1", "status": TaskStatus.COMPLETED.value},
                    {"task_id": "task2", "status": TaskStatus.IN_PROGRESS.value},
                    {"task_id": "task3", "status": TaskStatus.FAILED.value},
                ],
                "summary": {
                    "total_tasks": 3,
                    "status_counts": {
                        TaskStatus.COMPLETED.value: 1,
                        TaskStatus.IN_PROGRESS.value: 1,
                        TaskStatus.FAILED.value: 1,
                    },
                    "completion_percentage": 33.33,
                },
            },
            message="Successfully collected results from subtasks",
        )

        with patch.object(self.aggregator, "collect_results", return_value=mock_collection_result):
            result = self.aggregator.track_completion_status("agent1")

            assert result.success
            assert result.message == "Successfully tracked completion status"
            assert result.data["total_tasks"] == 3
            assert result.data["completed_count"] == 1
            assert result.data["in_progress_count"] == 1
            assert result.data["failed_count"] == 1
            assert result.data["completion_percentage"] == pytest.approx(33.33, abs=0.01)
            assert not result.data["is_complete"]
            assert not result.data["is_successful"]

    def test_track_completion_status_collection_failure(self) -> None:
        """Test track_completion_status when collection fails."""
        mock_collection_result = Result.failure(
            error=ValueError("Collection failed"),
            message="Failed to collect results",
        )

        with patch.object(self.aggregator, "collect_results", return_value=mock_collection_result):
            result = self.aggregator.track_completion_status("agent1")

            assert not result.success
            assert isinstance(result.error, ValueError)
            assert "Collection failed" in str(result.error)

    def test_track_completion_status_no_tasks(self) -> None:
        """Test track_completion_status when there are no tasks."""
        mock_collection_result = Result.success(
            data=None,
            message="No results to collect",
        )

        with patch.object(self.aggregator, "collect_results", return_value=mock_collection_result):
            result = self.aggregator.track_completion_status("agent1")

            assert result.success
            assert result.message == "No completion status to track"
            assert result.data == {"message": "No tasks found for tracking"}

    def test_calculate_tracking_metrics(self) -> None:
        """Test _calculate_tracking_metrics with various task statuses."""
        results = [
            {"task_id": "task1", "status": TaskStatus.COMPLETED.value},
            {"task_id": "task2", "status": TaskStatus.IN_PROGRESS.value},
            {"task_id": "task3", "status": TaskStatus.FAILED.value},
            {"task_id": "task4", "status": TaskStatus.PENDING.value},
            {"task_id": "task5", "status": TaskStatus.BLOCKED.value},
        ]

        summary = {
            "total_tasks": 5,
            "status_counts": {
                TaskStatus.COMPLETED.value: 1,
                TaskStatus.IN_PROGRESS.value: 1,
                TaskStatus.FAILED.value: 1,
                TaskStatus.PENDING.value: 1,
                TaskStatus.BLOCKED.value: 1,
            },
            "completion_percentage": 20.0,
        }

        metrics = self.aggregator._calculate_tracking_metrics(results, summary)

        assert metrics["total_tasks"] == 5
        assert metrics["completed_count"] == 1
        assert metrics["failed_count"] == 1
        assert metrics["in_progress_count"] == 1
        assert metrics["pending_count"] == 1
        assert metrics["blocked_count"] == 1
        assert metrics["completion_percentage"] == 20.0
        assert not metrics["is_complete"]
        assert not metrics["is_successful"]
        assert len(metrics["blocked_tasks"]) == 1
        assert metrics["blocked_tasks"][0]["status"] == TaskStatus.BLOCKED.value

    def test_calculate_tracking_metrics_all_completed(self) -> None:
        """Test _calculate_tracking_metrics with all tasks completed."""
        results = [
            {"task_id": "task1", "status": TaskStatus.COMPLETED.value},
            {"task_id": "task2", "status": TaskStatus.COMPLETED.value},
        ]

        summary = {
            "total_tasks": 2,
            "status_counts": {
                TaskStatus.COMPLETED.value: 2,
            },
            "completion_percentage": 100.0,
        }

        metrics = self.aggregator._calculate_tracking_metrics(results, summary)

        assert metrics["total_tasks"] == 2
        assert metrics["completed_count"] == 2
        assert metrics["failed_count"] == 0
        assert metrics["completion_percentage"] == 100.0
        assert metrics["is_complete"]
        assert metrics["is_successful"]
        assert len(metrics["blocked_tasks"]) == 0
