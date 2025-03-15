"""Tests for task progress tracking functionality."""

import uuid

import pytest

from src.agent.state.base import AgentState
from src.common_types.task_types import Task, TaskComplexity, TaskPriority, TaskStatus


class TestTaskProgressTracking:
    """Test task progress tracking functionality."""

    def test_track_delegated_task_progress(self) -> None:
        """Test tracking progress of a delegated task."""
        # Create a state with a task
        state = AgentState()
        task_id = uuid.uuid4()
        task = Task(
            description="Test task",
            task_id=task_id,
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            complexity=TaskComplexity.MODERATE,
        )
        state.add_task(task)

        # Track progress
        state.track_delegated_task_progress(task_id, 0.5, "Half done")

        # Verify progress was tracked
        updated_task = state.get_task_by_id(task_id)
        assert updated_task is not None
        assert "progress_tracking" in updated_task.metadata
        assert updated_task.metadata["progress_tracking"]["progress_percentage"] == 0.5
        assert updated_task.metadata["progress_tracking"]["status_message"] == "Half done"
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert len(updated_task.metadata["progress_tracking"]["progress_history"]) == 1

        # Track completion
        state.track_delegated_task_progress(task_id, 1.0, "Completed")

        # Verify completion was tracked
        updated_task = state.get_task_by_id(task_id)
        assert updated_task is not None
        assert updated_task.metadata["progress_tracking"]["progress_percentage"] == 1.0
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.completed_at is not None
        assert len(updated_task.metadata["progress_tracking"]["progress_history"]) == 2

    def test_track_delegated_task_progress_nonexistent_task(self) -> None:
        """Test tracking progress of a nonexistent task."""
        state = AgentState()
        nonexistent_task_id = uuid.uuid4()

        # This should not raise an exception
        state.track_delegated_task_progress(nonexistent_task_id, 0.5, "Half done")

    def test_track_delegated_task_progress_clamps_values(self) -> None:
        """Test that progress values are clamped between 0 and 1."""
        state = AgentState()
        task_id = uuid.uuid4()
        task = Task(description="Test task", task_id=task_id)
        state.add_task(task)

        # Track progress with value below 0
        state.track_delegated_task_progress(task_id, -0.5, "Negative progress")
        updated_task = state.get_task_by_id(task_id)
        assert updated_task.metadata["progress_tracking"]["progress_percentage"] == 0.0

        # Track progress with value above 1
        state.track_delegated_task_progress(task_id, 1.5, "Too much progress")
        updated_task = state.get_task_by_id(task_id)
        assert updated_task.metadata["progress_tracking"]["progress_percentage"] == 1.0

    def test_update_parent_task_progress(self) -> None:
        """Test updating parent task progress based on subtasks."""
        state = AgentState()

        # Create parent task
        parent_task_id = uuid.uuid4()
        parent_task = Task(
            description="Parent task",
            task_id=parent_task_id,
            status=TaskStatus.IN_PROGRESS,
        )
        state.add_task(parent_task)

        # Create subtasks
        subtask1_id = uuid.uuid4()
        subtask1 = Task(
            description="Subtask 1",
            task_id=subtask1_id,
            parent_task_id=parent_task_id,
            status=TaskStatus.IN_PROGRESS,
        )

        subtask2_id = uuid.uuid4()
        subtask2 = Task(
            description="Subtask 2",
            task_id=subtask2_id,
            parent_task_id=parent_task_id,
            status=TaskStatus.PENDING,
        )

        # Add subtasks to parent
        parent_task.subtasks.append(subtask1_id)
        parent_task.subtasks.append(subtask2_id)
        state.update_task(parent_task)

        # Add subtasks to state
        state.add_task(subtask1)
        state.add_task(subtask2)

        # Update progress of first subtask
        state.track_delegated_task_progress(subtask1_id, 0.5, "Half done")

        # Check parent progress (should be 0.25 = (0.5 + 0)/2)
        parent = state.get_task_by_id(parent_task_id)
        assert "progress_tracking" in parent.metadata
        assert parent.metadata["progress_tracking"]["progress_percentage"] == 0.25

        # Complete first subtask
        state.track_delegated_task_progress(subtask1_id, 1.0, "Completed")

        # Check parent progress (should be 0.5 = (1.0 + 0)/2)
        parent = state.get_task_by_id(parent_task_id)
        assert parent.metadata["progress_tracking"]["progress_percentage"] == 0.5

        # Update second subtask
        state.track_delegated_task_progress(subtask2_id, 0.5, "Half done")

        # Check parent progress (should be 0.75 = (1.0 + 0.5)/2)
        parent = state.get_task_by_id(parent_task_id)
        assert parent.metadata["progress_tracking"]["progress_percentage"] == 0.75

        # Complete second subtask
        state.track_delegated_task_progress(subtask2_id, 1.0, "Completed")

        # Check parent progress (should be 1.0 = (1.0 + 1.0)/2)
        parent = state.get_task_by_id(parent_task_id)
        assert parent.metadata["progress_tracking"]["progress_percentage"] == 1.0
        assert parent.status == TaskStatus.COMPLETED

    def test_multi_level_task_hierarchy_progress(self) -> None:
        """Test progress tracking in a multi-level task hierarchy."""
        state = AgentState()

        # Create root task
        root_task_id = uuid.uuid4()
        root_task = Task(
            description="Root task",
            task_id=root_task_id,
            status=TaskStatus.IN_PROGRESS,
        )
        state.add_task(root_task)

        # Create mid-level task
        mid_task_id = uuid.uuid4()
        mid_task = Task(
            description="Mid-level task",
            task_id=mid_task_id,
            parent_task_id=root_task_id,
            status=TaskStatus.IN_PROGRESS,
        )

        # Create leaf tasks
        leaf1_id = uuid.uuid4()
        leaf1 = Task(
            description="Leaf task 1",
            task_id=leaf1_id,
            parent_task_id=mid_task_id,
            status=TaskStatus.PENDING,
        )

        leaf2_id = uuid.uuid4()
        leaf2 = Task(
            description="Leaf task 2",
            task_id=leaf2_id,
            parent_task_id=mid_task_id,
            status=TaskStatus.PENDING,
        )

        # Set up hierarchy
        root_task.subtasks.append(mid_task_id)
        mid_task.subtasks.append(leaf1_id)
        mid_task.subtasks.append(leaf2_id)

        # Add all tasks to state
        state.update_task(root_task)
        state.add_task(mid_task)
        state.add_task(leaf1)
        state.add_task(leaf2)

        # Update leaf1 progress to 50%
        state.track_delegated_task_progress(leaf1_id, 0.5, "Half done")

        # Check progress propagation
        mid_task = state.get_task_by_id(mid_task_id)
        assert mid_task.metadata["progress_tracking"]["progress_percentage"] == 0.25

        root_task = state.get_task_by_id(root_task_id)
        assert root_task.metadata["progress_tracking"]["progress_percentage"] == 0.25

        # Complete both leaf tasks
        state.track_delegated_task_progress(leaf1_id, 1.0, "Completed")
        state.track_delegated_task_progress(leaf2_id, 1.0, "Completed")

        # Check that completion propagated up the hierarchy
        mid_task = state.get_task_by_id(mid_task_id)
        assert mid_task.metadata["progress_tracking"]["progress_percentage"] == 1.0
        assert mid_task.status == TaskStatus.COMPLETED

        root_task = state.get_task_by_id(root_task_id)
        assert root_task.metadata["progress_tracking"]["progress_percentage"] == 1.0
        assert root_task.status == TaskStatus.COMPLETED

    def test_get_task_progress(self) -> None:
        """Test getting progress information for a task."""
        state = AgentState()

        # Create a task with subtasks
        parent_id = uuid.uuid4()
        parent = Task(
            description="Parent task",
            task_id=parent_id,
            status=TaskStatus.IN_PROGRESS,
        )

        child1_id = uuid.uuid4()
        child1 = Task(
            description="Child task 1",
            task_id=child1_id,
            parent_task_id=parent_id,
            status=TaskStatus.COMPLETED,
        )

        child2_id = uuid.uuid4()
        child2 = Task(
            description="Child task 2",
            task_id=child2_id,
            parent_task_id=parent_id,
            status=TaskStatus.IN_PROGRESS,
        )

        # Set up hierarchy
        parent.subtasks.append(child1_id)
        parent.subtasks.append(child2_id)

        # Add tasks to state
        state.add_task(parent)
        state.add_task(child1)
        state.add_task(child2)

        # Update progress
        state.track_delegated_task_progress(child2_id, 0.5, "Half done")
        state.track_delegated_task_progress(parent_id, 0.75, "Almost done")

        # Get progress info
        progress_info = state.get_task_progress(parent_id)

        # Verify structure and content
        assert progress_info["task_id"] == str(parent_id)
        assert progress_info["description"] == "Parent task"
        assert progress_info["status"] == "in_progress"
        assert progress_info["progress"] == 0.75
        assert "last_updated" in progress_info
        assert progress_info["status_message"] == "Almost done"
        assert len(progress_info["subtasks_progress"]) == 2

        # Check subtask info
        subtask_progress = next(p for p in progress_info["subtasks_progress"] if p["task_id"] == str(child1_id))
        assert subtask_progress["progress"] == 1.0  # Completed task should have 100% progress
        assert subtask_progress["status"] == "completed"

        subtask_progress = next(p for p in progress_info["subtasks_progress"] if p["task_id"] == str(child2_id))
        assert subtask_progress["progress"] == 0.5
        assert subtask_progress["status"] == "in_progress"

    def test_get_overall_progress(self) -> None:
        """Test getting overall progress information for all tasks."""
        state = AgentState()

        # Create several tasks with different statuses
        task1_id = uuid.uuid4()
        task1 = Task(
            description="Task 1",
            task_id=task1_id,
            status=TaskStatus.COMPLETED,
        )

        task2_id = uuid.uuid4()
        task2 = Task(
            description="Task 2",
            task_id=task2_id,
            status=TaskStatus.IN_PROGRESS,
        )

        task3_id = uuid.uuid4()
        task3 = Task(
            description="Task 3",
            task_id=task3_id,
            status=TaskStatus.PENDING,
        )

        task4_id = uuid.uuid4()
        task4 = Task(
            description="Task 4",
            task_id=task4_id,
            status=TaskStatus.BLOCKED,
        )

        task5_id = uuid.uuid4()
        task5 = Task(
            description="Task 5",
            task_id=task5_id,
            status=TaskStatus.FAILED,
        )

        # Add tasks to state
        state.add_task(task1)
        state.add_task(task2)
        state.add_task(task3)
        state.add_task(task4)
        state.add_task(task5)

        # Update progress for in-progress task
        state.track_delegated_task_progress(task2_id, 0.5, "Half done")

        # Get overall progress
        overall_progress = state.get_overall_progress()

        # Verify structure and content
        assert "overall_progress" in overall_progress
        assert overall_progress["total_tasks"] == 5
        assert overall_progress["completed_tasks"] == 1
        assert overall_progress["in_progress_tasks"] == 1
        assert overall_progress["pending_tasks"] == 1
        assert overall_progress["blocked_tasks"] == 1
        assert overall_progress["failed_tasks"] == 1

        # Check overall progress calculation (1 completed + 0.5 * 1 in-progress) / 5 = 0.3
        assert overall_progress["overall_progress"] == pytest.approx(0.3)

        # Check root tasks
        assert len(overall_progress["root_tasks"]) == 5  # All tasks are root tasks
