"""Tests for task completion status and blockers tracking functionality."""

import uuid

from src.agent.state.base import AgentState
from src.common_types.enums import ExecutionStage, VerificationStatus
from src.common_types.task_types import Task, TaskDependency, TaskStatus


class TestTaskCompletionStatus:
    """Test task completion status tracking functionality."""

    def test_track_task_completion_status_successful_completion(self) -> None:
        """Test tracking successful task completion."""
        state = AgentState()
        task_id = uuid.uuid4()
        task = Task(
            description="Test task",
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.FINALIZING,
            verification_status=VerificationStatus.PASSED,
        )
        state.add_task(task)

        # Debug: Print task before tracking

        # Get the task from state to verify it was added correctly
        state.get_task_by_id(task_id)

        # Track completion status
        state.track_task_completion_status(task_id)

        # Debug: Print task after tracking
        updated_task = state.get_task_by_id(task_id)

        # Print the raw task data in the state
        state.get_tasks()

        # Verify task was marked as completed
        updated_task = state.get_task_by_id(task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.completed_at is not None

        # Verify progress tracking was updated
        assert "progress_tracking" in updated_task.metadata
        assert updated_task.metadata["progress_tracking"]["progress_percentage"] == 1.0
        assert "Task completed successfully" in updated_task.metadata["progress_tracking"]["status_message"]

    def test_track_task_completion_status_failure(self) -> None:
        """Test tracking task failure after multiple attempts."""
        state = AgentState()
        task_id = uuid.uuid4()
        task = Task(
            description="Test task",
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.TESTING,
            verification_status=VerificationStatus.FAILED,
            execution_attempts=3,
        )
        state.add_task(task)

        # Track completion status
        state.track_task_completion_status(task_id)

        # Verify task was marked as failed
        updated_task = state.get_task_by_id(task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.FAILED
        assert updated_task.error is not None
        assert "Failed after 3 attempts" in updated_task.error

        # Verify failure information was added to progress tracking
        assert "progress_tracking" in updated_task.metadata
        assert "failure_reason" in updated_task.metadata["progress_tracking"]
        assert "failed_at" in updated_task.metadata["progress_tracking"]

    def test_track_task_completion_status_in_progress(self) -> None:
        """Test tracking task that is still in progress."""
        state = AgentState()
        task_id = uuid.uuid4()
        task = Task(
            description="Test task",
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            execution_stage=ExecutionStage.IMPLEMENTING,
            verification_status=VerificationStatus.PENDING,
            execution_attempts=1,
        )
        state.add_task(task)

        # Track completion status
        state.track_task_completion_status(task_id)

        # Verify task status remains unchanged
        updated_task = state.get_task_by_id(task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.completed_at is None


class TestTaskBlockersTracking:
    """Test task blockers and dependencies tracking functionality."""

    def test_track_blockers_and_dependencies(self) -> None:
        """Test tracking blockers and dependencies."""
        state = AgentState()

        # Create dependency task
        dependency_task = Task(description="Dependency task")
        state.add_task(dependency_task)

        # Create dependent task
        dependent_task = Task(
            description="Dependent task",
            dependencies=[
                TaskDependency(
                    task_id=dependency_task.task_id,
                    description="Depends on dependency task",
                    is_blocking=True,
                ),
            ],
        )
        state.add_task(dependent_task)

        # Track blockers and dependencies
        state.track_blockers_and_dependencies()

        # Verify dependent task is blocked
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.BLOCKED

        # Verify blocker information was added
        assert "blockers" in updated_task.metadata
        assert "blocking_dependencies" in updated_task.metadata["blockers"]
        assert len(updated_task.metadata["blockers"]["blocking_dependencies"]) == 1
        assert updated_task.metadata["blockers"]["blocking_dependencies"][0]["task_id"] == str(dependency_task.task_id)
        assert "last_updated" in updated_task.metadata["blockers"]

    def test_track_blockers_resolution(self) -> None:
        """Test tracking resolution of blockers."""
        state = AgentState()

        # Create dependency task
        dependency_task = Task(description="Dependency task")
        state.add_task(dependency_task)

        # Create dependent task
        dependent_task = Task(
            description="Dependent task",
            dependencies=[
                TaskDependency(
                    task_id=dependency_task.task_id,
                    description="Depends on dependency task",
                    is_blocking=True,
                ),
            ],
        )
        state.add_task(dependent_task)

        # Track blockers and dependencies
        state.track_blockers_and_dependencies()

        # Verify dependent task is blocked
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.BLOCKED

        # Complete dependency task
        dependency_task.status = TaskStatus.COMPLETED
        state.update_task(dependency_task)

        # Track blockers and dependencies again
        state.track_blockers_and_dependencies()

        # Verify dependent task is no longer blocked
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.PENDING

        # Verify blocker information was updated
        assert "blockers" in updated_task.metadata
        assert "blocking_dependencies" in updated_task.metadata["blockers"]
        assert len(updated_task.metadata["blockers"]["blocking_dependencies"]) == 0

    def test_track_multiple_blockers(self) -> None:
        """Test tracking multiple blockers for a task."""
        state = AgentState()

        # Create dependency tasks
        dependency_task1 = Task(description="Dependency task 1")
        dependency_task2 = Task(description="Dependency task 2")
        state.add_task(dependency_task1)
        state.add_task(dependency_task2)

        # Create dependent task with multiple dependencies
        dependent_task = Task(
            description="Dependent task",
            dependencies=[
                TaskDependency(
                    task_id=dependency_task1.task_id,
                    description="Depends on dependency task 1",
                    is_blocking=True,
                ),
                TaskDependency(
                    task_id=dependency_task2.task_id,
                    description="Depends on dependency task 2",
                    is_blocking=True,
                ),
            ],
        )
        state.add_task(dependent_task)

        # Track blockers and dependencies
        state.track_blockers_and_dependencies()

        # Verify dependent task is blocked
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.BLOCKED

        # Verify blocker information was added for both dependencies
        assert "blockers" in updated_task.metadata
        assert "blocking_dependencies" in updated_task.metadata["blockers"]
        assert len(updated_task.metadata["blockers"]["blocking_dependencies"]) == 2

        # Complete first dependency
        dependency_task1.status = TaskStatus.COMPLETED
        state.update_task(dependency_task1)

        # Track blockers and dependencies again
        state.track_blockers_and_dependencies()

        # Verify task is still blocked but with only one blocker
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.BLOCKED
        assert len(updated_task.metadata["blockers"]["blocking_dependencies"]) == 1
        assert updated_task.metadata["blockers"]["blocking_dependencies"][0]["task_id"] == str(dependency_task2.task_id)

        # Complete second dependency
        dependency_task2.status = TaskStatus.COMPLETED
        state.update_task(dependency_task2)

        # Track blockers and dependencies again
        state.track_blockers_and_dependencies()

        # Verify task is no longer blocked
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.PENDING
        assert len(updated_task.metadata["blockers"]["blocking_dependencies"]) == 0
