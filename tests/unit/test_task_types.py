"""Tests for task types."""

import uuid
from datetime import datetime

from src.common_types.enums import AgentRole
from src.common_types.task_types import (
    Task,
    TaskComplexity,
    TaskDependency,
    TaskPriority,
    TaskStatus,
)


class TestTaskTypes:
    """Test task types."""

    def test_task_status_enum(self) -> None:
        """Test TaskStatus enum."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.BLOCKED.value == "blocked"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"

    def test_task_priority_enum(self) -> None:
        """Test TaskPriority enum."""
        assert TaskPriority.LOW.value == "low"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.CRITICAL.value == "critical"

    def test_task_complexity_enum(self) -> None:
        """Test TaskComplexity enum."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.VERY_COMPLEX.value == "very_complex"

    def test_task_dependency_initialization(self) -> None:
        """Test TaskDependency initialization."""
        task_id = uuid.uuid4()
        dependency = TaskDependency(task_id=task_id, description="Test dependency")
        assert dependency.task_id == task_id
        assert dependency.description == "Test dependency"
        assert dependency.is_blocking is True

        non_blocking_dependency = TaskDependency(
            task_id=task_id,
            description="Non-blocking dependency",
            is_blocking=False,
        )
        assert non_blocking_dependency.is_blocking is False

    def test_task_initialization(self) -> None:
        """Test Task initialization."""
        task = Task(description="Test task")
        assert task.description == "Test task"
        assert isinstance(task.task_id, uuid.UUID)
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.complexity == TaskComplexity.MODERATE
        assert task.dependencies == []
        assert task.parent_task_id is None
        assert task.subtasks == []
        assert task.assigned_role is None
        assert task.assigned_agent_id is None
        assert task.metadata == {}
        assert task.result is None
        assert task.error is None
        assert task.created_at is None
        assert task.updated_at is None
        assert task.completed_at is None

    def test_task_with_custom_values(self) -> None:
        """Test Task with custom values."""
        task_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        subtask_id = uuid.uuid4()
        dependency_id = uuid.uuid4()
        dependency = TaskDependency(
            task_id=dependency_id,
            description="Test dependency",
        )
        now = datetime.now().timestamp()

        task = Task(
            description="Custom task",
            task_id=task_id,
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            complexity=TaskComplexity.COMPLEX,
            dependencies=[dependency],
            parent_task_id=parent_id,
            subtasks=[subtask_id],
            assigned_role=AgentRole.EXECUTOR,
            assigned_agent_id="agent-123",
            metadata={"key": "value"},
            result="Task result",
            error=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )

        assert task.description == "Custom task"
        assert task.task_id == task_id
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.complexity == TaskComplexity.COMPLEX
        assert len(task.dependencies) == 1
        assert task.dependencies[0].task_id == dependency_id
        assert task.parent_task_id == parent_id
        assert task.subtasks == [subtask_id]
        assert task.assigned_role == AgentRole.EXECUTOR
        assert task.assigned_agent_id == "agent-123"
        assert task.metadata == {"key": "value"}
        assert task.result == "Task result"
        assert task.error is None
        assert task.created_at == now
        assert task.updated_at == now
        assert task.completed_at is None
