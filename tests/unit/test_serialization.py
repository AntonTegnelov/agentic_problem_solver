"""Tests for task serialization utilities."""

import json
import uuid
from datetime import UTC, datetime

import pytest

from src.common_types.enums import AgentRole
from src.common_types.task_types import Task, TaskComplexity, TaskDependency, TaskPriority, TaskStatus
from src.utils.serialization import (
    deserialize_task,
    deserialize_task_list,
    json_to_task,
    serialize_task,
    serialize_task_list,
    task_to_json,
)


class TestTaskSerialization:
    """Test task serialization utilities."""

    def test_serialize_task_basic(self) -> None:
        """Test basic task serialization."""
        task = Task(description="Test task")
        serialized = serialize_task(task)

        assert isinstance(serialized, dict)
        assert serialized["description"] == "Test task"
        assert isinstance(serialized["task_id"], str)
        assert serialized["priority"] == "medium"
        assert serialized["status"] == "pending"
        assert serialized["complexity"] == "moderate"
        assert serialized["dependencies"] == []
        assert serialized["subtasks"] == []

    def test_serialize_task_with_uuids(self) -> None:
        """Test serialization of task with UUID fields."""
        task_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        subtask_id = uuid.uuid4()
        dependency_id = uuid.uuid4()

        task = Task(
            description="Test task",
            task_id=task_id,
            parent_task_id=parent_id,
            subtasks=[subtask_id],
            dependencies=[
                TaskDependency(
                    task_id=dependency_id,
                    description="Test dependency",
                ),
            ],
        )

        serialized = serialize_task(task)

        assert serialized["task_id"] == str(task_id)
        assert serialized["parent_task_id"] == str(parent_id)
        assert serialized["subtasks"] == [str(subtask_id)]
        assert len(serialized["dependencies"]) == 1
        assert serialized["dependencies"][0]["task_id"] == str(dependency_id)

    def test_serialize_task_with_enums(self) -> None:
        """Test serialization of task with enum fields."""
        task = Task(
            description="Test task",
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            complexity=TaskComplexity.COMPLEX,
            assigned_role=AgentRole.EXECUTOR,
        )

        serialized = serialize_task(task)

        assert serialized["priority"] == "high"
        assert serialized["status"] == "in_progress"
        assert serialized["complexity"] == "complex"
        assert serialized["assigned_role"] == "executor"

    def test_deserialize_task_basic(self) -> None:
        """Test basic task deserialization."""
        task_dict = {
            "description": "Test task",
            "task_id": str(uuid.uuid4()),
            "priority": "medium",
            "status": "pending",
            "complexity": "moderate",
            "dependencies": [],
            "subtasks": [],
        }

        task = deserialize_task(task_dict)

        assert isinstance(task, Task)
        assert task.description == "Test task"
        assert isinstance(task.task_id, uuid.UUID)
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.complexity == TaskComplexity.MODERATE
        assert task.dependencies == []
        assert task.subtasks == []

    def test_deserialize_task_with_uuids(self) -> None:
        """Test deserialization of task with UUID fields."""
        task_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        subtask_id = uuid.uuid4()
        dependency_id = uuid.uuid4()

        task_dict = {
            "description": "Test task",
            "task_id": str(task_id),
            "parent_task_id": str(parent_id),
            "subtasks": [str(subtask_id)],
            "dependencies": [
                {
                    "task_id": str(dependency_id),
                    "description": "Test dependency",
                    "is_blocking": True,
                },
            ],
            "priority": "medium",
            "status": "pending",
            "complexity": "moderate",
        }

        task = deserialize_task(task_dict)

        assert task.task_id == task_id
        assert task.parent_task_id == parent_id
        assert task.subtasks == [subtask_id]
        assert len(task.dependencies) == 1
        assert task.dependencies[0].task_id == dependency_id
        assert task.dependencies[0].description == "Test dependency"
        assert task.dependencies[0].is_blocking is True

    def test_deserialize_task_with_enums(self) -> None:
        """Test deserialization of task with enum fields."""
        task_dict = {
            "description": "Test task",
            "task_id": str(uuid.uuid4()),
            "priority": "high",
            "status": "in_progress",
            "complexity": "complex",
            "assigned_role": "executor",
            "dependencies": [],
            "subtasks": [],
        }

        task = deserialize_task(task_dict)

        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.complexity == TaskComplexity.COMPLEX
        assert task.assigned_role == AgentRole.EXECUTOR

    def test_deserialize_task_invalid(self) -> None:
        """Test deserialization of invalid task dictionary."""
        # Missing required field
        task_dict = {
            "task_id": str(uuid.uuid4()),
            "priority": "medium",
            "status": "pending",
            "complexity": "moderate",
        }

        with pytest.raises(ValueError):
            deserialize_task(task_dict)

        # Invalid UUID
        task_dict = {
            "description": "Test task",
            "task_id": "not-a-uuid",
            "priority": "medium",
            "status": "pending",
            "complexity": "moderate",
        }

        with pytest.raises(ValueError):
            deserialize_task(task_dict)

        # Invalid enum value
        task_dict = {
            "description": "Test task",
            "task_id": str(uuid.uuid4()),
            "priority": "invalid-priority",
            "status": "pending",
            "complexity": "moderate",
        }

        with pytest.raises(ValueError):
            deserialize_task(task_dict)

    def test_serialize_task_list(self) -> None:
        """Test serialization of task list."""
        task1 = Task(description="Task 1")
        task2 = Task(description="Task 2")
        tasks = [task1, task2]

        serialized = serialize_task_list(tasks)

        assert isinstance(serialized, str)
        task_dicts = json.loads(serialized)
        assert len(task_dicts) == 2
        assert task_dicts[0]["description"] == "Task 1"
        assert task_dicts[1]["description"] == "Task 2"

    def test_deserialize_task_list(self) -> None:
        """Test deserialization of task list."""
        task1_id = uuid.uuid4()
        task2_id = uuid.uuid4()

        json_str = json.dumps(
            [
                {
                    "description": "Task 1",
                    "task_id": str(task1_id),
                    "priority": "medium",
                    "status": "pending",
                    "complexity": "moderate",
                    "dependencies": [],
                    "subtasks": [],
                },
                {
                    "description": "Task 2",
                    "task_id": str(task2_id),
                    "priority": "high",
                    "status": "in_progress",
                    "complexity": "complex",
                    "dependencies": [],
                    "subtasks": [],
                },
            ]
        )

        tasks = deserialize_task_list(json_str)

        assert len(tasks) == 2
        assert tasks[0].description == "Task 1"
        assert tasks[0].task_id == task1_id
        assert tasks[1].description == "Task 2"
        assert tasks[1].task_id == task2_id
        assert tasks[1].priority == TaskPriority.HIGH

    def test_deserialize_task_list_invalid(self) -> None:
        """Test deserialization of invalid task list JSON."""
        # Invalid JSON
        with pytest.raises(ValueError):
            deserialize_task_list("not-json")

        # Not a list
        with pytest.raises(TypeError):
            deserialize_task_list('{"description": "Not a list"}')

    def test_task_to_json(self) -> None:
        """Test conversion of task to JSON string."""
        task = Task(description="Test task")
        json_str = task_to_json(task)

        assert isinstance(json_str, str)
        task_dict = json.loads(json_str)
        assert task_dict["description"] == "Test task"

    def test_json_to_task(self) -> None:
        """Test conversion of JSON string to task."""
        task_id = uuid.uuid4()
        json_str = json.dumps(
            {
                "description": "Test task",
                "task_id": str(task_id),
                "priority": "medium",
                "status": "pending",
                "complexity": "moderate",
                "dependencies": [],
                "subtasks": [],
            }
        )

        task = json_to_task(json_str)

        assert isinstance(task, Task)
        assert task.description == "Test task"
        assert task.task_id == task_id

    def test_json_to_task_invalid(self) -> None:
        """Test conversion of invalid JSON string to task."""
        with pytest.raises(ValueError):
            json_to_task("not-json")

    def test_roundtrip_serialization(self) -> None:
        """Test roundtrip serialization and deserialization."""
        # Create a complex task
        now = datetime.now(UTC).timestamp()
        original_task = Task(
            description="Complex task",
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            complexity=TaskComplexity.COMPLEX,
            assigned_role=AgentRole.PLANNER,
            assigned_agent_id="agent-123",
            metadata={"key": "value"},
            result="Task result",
            error=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )

        # Add dependencies and subtasks
        dependency_id = uuid.uuid4()
        original_task.dependencies = [
            TaskDependency(
                task_id=dependency_id,
                description="Test dependency",
                is_blocking=True,
            ),
        ]
        subtask_id = uuid.uuid4()
        original_task.subtasks = [subtask_id]
        parent_id = uuid.uuid4()
        original_task.parent_task_id = parent_id

        # Serialize and deserialize
        serialized = serialize_task(original_task)
        deserialized_task = deserialize_task(serialized)

        # Verify all fields match
        assert deserialized_task.description == original_task.description
        assert deserialized_task.task_id == original_task.task_id
        assert deserialized_task.priority == original_task.priority
        assert deserialized_task.status == original_task.status
        assert deserialized_task.complexity == original_task.complexity
        assert deserialized_task.assigned_role == original_task.assigned_role
        assert deserialized_task.assigned_agent_id == original_task.assigned_agent_id
        assert deserialized_task.metadata == original_task.metadata
        assert deserialized_task.result == original_task.result
        assert deserialized_task.error == original_task.error
        assert deserialized_task.created_at == original_task.created_at
        assert deserialized_task.updated_at == original_task.updated_at
        assert deserialized_task.completed_at == original_task.completed_at
        assert deserialized_task.parent_task_id == original_task.parent_task_id
        assert deserialized_task.subtasks == original_task.subtasks
        assert len(deserialized_task.dependencies) == len(original_task.dependencies)
        assert deserialized_task.dependencies[0].task_id == original_task.dependencies[0].task_id
        assert deserialized_task.dependencies[0].description == original_task.dependencies[0].description
        assert deserialized_task.dependencies[0].is_blocking == original_task.dependencies[0].is_blocking
