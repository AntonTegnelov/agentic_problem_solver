"""Tests for validation utilities."""

import uuid
from typing import Any

from src.common_types.task_types import Task, TaskDependency
from src.utils.validation import (
    validate_dict_as_task,
    validate_task,
    validate_task_dependencies,
    validate_task_list,
)


def test_validate_task_valid() -> None:
    """Test that a valid task passes validation."""
    task = Task(description="Test task")
    is_valid, error = validate_task(task)
    assert is_valid
    assert error is None


def test_validate_task_empty_description() -> None:
    """Test that a task with an empty description fails validation."""
    task = Task(description="")
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "description cannot be empty" in error


def test_validate_task_invalid_task_id() -> None:
    """Test that a task with an invalid task_id fails validation."""
    task = Task(description="Test task")
    task.task_id = "not-a-uuid"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Task ID must be a UUID" in error


def test_validate_task_invalid_priority() -> None:
    """Test that a task with an invalid priority fails validation."""
    task = Task(description="Test task")
    task.priority = "invalid"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Task priority must be a TaskPriority enum" in error


def test_validate_task_invalid_status() -> None:
    """Test that a task with an invalid status fails validation."""
    task = Task(description="Test task")
    task.status = "invalid"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Task status must be a TaskStatus enum" in error


def test_validate_task_invalid_complexity() -> None:
    """Test that a task with an invalid complexity fails validation."""
    task = Task(description="Test task")
    task.complexity = "invalid"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Task complexity must be a TaskComplexity enum" in error


def test_validate_task_invalid_dependencies() -> None:
    """Test that a task with invalid dependencies fails validation."""
    task = Task(description="Test task")
    task.dependencies = "not-a-list"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Task dependencies must be a list" in error


def test_validate_task_invalid_dependency_type() -> None:
    """Test that a task with a dependency of the wrong type fails validation."""
    task = Task(description="Test task")
    task.dependencies = ["not-a-dependency"]  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Dependency at index 0 must be a TaskDependency" in error


def test_validate_task_invalid_dependency_task_id() -> None:
    """Test that a task with a dependency with an invalid task_id fails validation."""
    task = Task(description="Test task")
    dependency = TaskDependency(task_id="not-a-uuid", description="Dependency")  # type: ignore
    task.dependencies = [dependency]
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Dependency task_id at index 0 must be a UUID" in error


def test_validate_task_empty_dependency_description() -> None:
    """Test that a task with a dependency with an empty description fails validation."""
    task = Task(description="Test task")
    dependency = TaskDependency(task_id=uuid.uuid4(), description="")
    task.dependencies = [dependency]
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Dependency at index 0 must have a description" in error


def test_validate_task_invalid_parent_task_id() -> None:
    """Test that a task with an invalid parent_task_id fails validation."""
    task = Task(description="Test task")
    task.parent_task_id = "not-a-uuid"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Parent task ID must be a UUID" in error


def test_validate_task_invalid_subtasks() -> None:
    """Test that a task with invalid subtasks fails validation."""
    task = Task(description="Test task")
    task.subtasks = "not-a-list"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Subtasks must be a list" in error


def test_validate_task_invalid_subtask_id() -> None:
    """Test that a task with an invalid subtask ID fails validation."""
    task = Task(description="Test task")
    task.subtasks = ["not-a-uuid"]  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Subtask ID at index 0 must be a UUID" in error


def test_validate_task_invalid_metadata() -> None:
    """Test that a task with invalid metadata fails validation."""
    task = Task(description="Test task")
    task.metadata = "not-a-dict"  # type: ignore
    is_valid, error = validate_task(task)
    assert not is_valid
    assert "Metadata must be a dictionary" in error


def test_validate_task_list_valid() -> None:
    """Test that a valid task list passes validation."""
    tasks = [Task(description="Task 1"), Task(description="Task 2")]
    is_valid, error = validate_task_list(tasks)
    assert is_valid
    assert error is None


def test_validate_task_list_not_a_list() -> None:
    """Test that a non-list fails validation."""
    tasks = "not-a-list"  # type: ignore
    is_valid, error = validate_task_list(tasks)
    assert not is_valid
    assert "Expected a list of tasks" in error


def test_validate_task_list_invalid_item() -> None:
    """Test that a list with a non-Task item fails validation."""
    tasks = [Task(description="Task 1"), "not-a-task"]  # type: ignore
    is_valid, error = validate_task_list(tasks)
    assert not is_valid
    assert "Item at index 1 must be a Task" in error


def test_validate_task_list_invalid_task() -> None:
    """Test that a list with an invalid Task fails validation."""
    tasks = [Task(description="Task 1"), Task(description="")]
    is_valid, error = validate_task_list(tasks)
    assert not is_valid
    assert "Task at index 1 is invalid" in error


def test_validate_task_dependencies_valid() -> None:
    """Test that valid task dependencies pass validation."""
    task1_id = uuid.uuid4()
    task2_id = uuid.uuid4()

    task1 = Task(description="Task 1")
    task1.task_id = task1_id

    task2 = Task(description="Task 2")
    task2.task_id = task2_id
    task2.dependencies = [TaskDependency(task_id=task1_id, description="Depends on Task 1")]

    tasks = [task1, task2]
    is_valid, error = validate_task_dependencies(tasks)
    assert is_valid
    assert error is None


def test_validate_task_dependencies_nonexistent_dependency() -> None:
    """Test that a dependency on a non-existent task fails validation."""
    task1_id = uuid.uuid4()
    nonexistent_id = uuid.uuid4()

    task1 = Task(description="Task 1")
    task1.task_id = task1_id
    task1.dependencies = [TaskDependency(task_id=nonexistent_id, description="Depends on non-existent task")]

    tasks = [task1]
    is_valid, error = validate_task_dependencies(tasks)
    assert not is_valid
    assert "has dependency on non-existent task" in error


def test_validate_task_dependencies_circular() -> None:
    """Test that circular dependencies fail validation."""
    task1_id = uuid.uuid4()
    task2_id = uuid.uuid4()

    task1 = Task(description="Task 1")
    task1.task_id = task1_id
    task1.dependencies = [TaskDependency(task_id=task2_id, description="Depends on Task 2")]

    task2 = Task(description="Task 2")
    task2.task_id = task2_id
    task2.dependencies = [TaskDependency(task_id=task1_id, description="Depends on Task 1")]

    tasks = [task1, task2]
    is_valid, error = validate_task_dependencies(tasks)
    assert not is_valid
    assert "Circular dependency detected" in error


def test_validate_dict_as_task_valid() -> None:
    """Test that a valid dictionary passes validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "priority": "medium",
        "status": "pending",
        "complexity": "moderate",
        "dependencies": [],
    }
    is_valid, error = validate_dict_as_task(data)
    assert is_valid
    assert error is None


def test_validate_dict_as_task_missing_description() -> None:
    """Test that a dictionary without a description fails validation."""
    data: dict[str, Any] = {
        "priority": "medium",
        "status": "pending",
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "Missing required field: description" in error


def test_validate_dict_as_task_invalid_description_type() -> None:
    """Test that a dictionary with a non-string description fails validation."""
    data: dict[str, Any] = {
        "description": 123,
        "priority": "medium",
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "description must be a string" in error


def test_validate_dict_as_task_invalid_priority() -> None:
    """Test that a dictionary with an invalid priority fails validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "priority": "invalid",
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "Invalid priority value" in error


def test_validate_dict_as_task_invalid_status() -> None:
    """Test that a dictionary with an invalid status fails validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "status": "invalid",
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "Invalid status value" in error


def test_validate_dict_as_task_invalid_complexity() -> None:
    """Test that a dictionary with an invalid complexity fails validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "complexity": "invalid",
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "Invalid complexity value" in error


def test_validate_dict_as_task_invalid_dependencies_type() -> None:
    """Test that a dictionary with non-list dependencies fails validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "dependencies": "not-a-list",
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "dependencies must be a list" in error


def test_validate_dict_as_task_invalid_dependency_type() -> None:
    """Test that a dictionary with a non-dict dependency fails validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "dependencies": ["not-a-dict"],
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "Dependency at index 0 must be a dictionary" in error


def test_validate_dict_as_task_missing_dependency_task_id() -> None:
    """Test that a dictionary with a dependency missing task_id fails validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "dependencies": [{"description": "Dependency"}],
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "Dependency at index 0 missing required field: task_id" in error


def test_validate_dict_as_task_missing_dependency_description() -> None:
    """Test that a dictionary with a dependency missing description fails validation."""
    data: dict[str, Any] = {
        "description": "Test task",
        "dependencies": [{"task_id": str(uuid.uuid4())}],
    }
    is_valid, error = validate_dict_as_task(data)
    assert not is_valid
    assert "Dependency at index 0 missing required field: description" in error
