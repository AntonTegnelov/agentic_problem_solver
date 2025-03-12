"""Validation utilities.

This module provides validation functions for various data structures
used throughout the application, particularly focusing on task validation.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar
from uuid import UUID

from src.common_types.task_types import Task, TaskComplexity, TaskDependency, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")
ValidationResult = tuple[bool, str | None]


def _validate_task_basic_fields(task: Task) -> ValidationResult:
    """Validate the basic fields of a task.

    Args:
        task: The task to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    # Validate required fields
    if not task.description:
        return False, "Task description cannot be empty"

    if not isinstance(task.task_id, UUID):
        return False, f"Task ID must be a UUID, got {type(task.task_id)}"

    # Validate enum fields
    if not isinstance(task.priority, TaskPriority):
        return False, f"Task priority must be a TaskPriority enum, got {type(task.priority)}"

    if not isinstance(task.status, TaskStatus):
        return False, f"Task status must be a TaskStatus enum, got {type(task.status)}"

    if not isinstance(task.complexity, TaskComplexity):
        return False, f"Task complexity must be a TaskComplexity enum, got {type(task.complexity)}"

    return True, None


def _validate_task_dependencies_field(task: Task) -> ValidationResult:
    """Validate the dependencies field of a task.

    Args:
        task: The task to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    if not isinstance(task.dependencies, list):
        return False, f"Task dependencies must be a list, got {type(task.dependencies)}"

    for i, dependency in enumerate(task.dependencies):
        if not isinstance(dependency, TaskDependency):
            return False, f"Dependency at index {i} must be a TaskDependency, got {type(dependency)}"

        if not isinstance(dependency.task_id, UUID):
            return False, f"Dependency task_id at index {i} must be a UUID, got {type(dependency.task_id)}"

        if not dependency.description:
            return False, f"Dependency at index {i} must have a description"

    return True, None


def _validate_task_relationships(task: Task) -> ValidationResult:
    """Validate the parent-child relationships of a task.

    Args:
        task: The task to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    # Validate parent_task_id if present
    if task.parent_task_id is not None and not isinstance(task.parent_task_id, UUID):
        return False, f"Parent task ID must be a UUID, got {type(task.parent_task_id)}"

    # Validate subtasks
    if not isinstance(task.subtasks, list):
        return False, f"Subtasks must be a list, got {type(task.subtasks)}"

    for i, subtask_id in enumerate(task.subtasks):
        if not isinstance(subtask_id, UUID):
            return False, f"Subtask ID at index {i} must be a UUID, got {type(subtask_id)}"

    return True, None


def _validate_task_metadata(task: Task) -> ValidationResult:
    """Validate the metadata of a task.

    Args:
        task: The task to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    if not isinstance(task.metadata, dict):
        return False, f"Metadata must be a dictionary, got {type(task.metadata)}"

    return True, None


def validate_task(task: Task) -> ValidationResult:
    """Validate a task against the schema requirements.

    Args:
        task: The task to validate

    Returns:
        A tuple containing (is_valid, error_message)
        where is_valid is a boolean indicating if the task is valid,
        and error_message is an optional string with validation error details

    """
    # Validate basic fields
    is_valid, error = _validate_task_basic_fields(task)
    if not is_valid:
        return False, error

    # Validate dependencies
    is_valid, error = _validate_task_dependencies_field(task)
    if not is_valid:
        return False, error

    # Validate relationships
    is_valid, error = _validate_task_relationships(task)
    if not is_valid:
        return False, error

    # Validate metadata
    is_valid, error = _validate_task_metadata(task)
    if not is_valid:
        return False, error

    # All validations passed
    return True, None


def validate_task_list(tasks: list[Task]) -> ValidationResult:
    """Validate a list of tasks.

    Args:
        tasks: The list of tasks to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    if not isinstance(tasks, list):
        return False, f"Expected a list of tasks, got {type(tasks)}"

    for i, task in enumerate(tasks):
        if not isinstance(task, Task):
            return False, f"Item at index {i} must be a Task, got {type(task)}"

        is_valid, error = validate_task(task)
        if not is_valid:
            return False, f"Task at index {i} is invalid: {error}"

    return True, None


def _check_dependency_references(tasks: list[Task], task_map: dict[UUID, Task]) -> ValidationResult:
    """Check that all dependency task_ids reference existing tasks.

    Args:
        tasks: The list of tasks to validate
        task_map: A map of task_id to task for quick lookup

    Returns:
        A tuple containing (is_valid, error_message)

    """
    for task in tasks:
        for dependency in task.dependencies:
            if dependency.task_id not in task_map:
                return False, f"Task {task.task_id} has dependency on non-existent task {dependency.task_id}"

    return True, None


def _check_circular_dependencies(task_map: dict[UUID, Task]) -> ValidationResult:
    """Check for circular dependencies using depth-first search.

    Args:
        task_map: A map of task_id to task for quick lookup

    Returns:
        A tuple containing (is_valid, error_message)

    """
    visited = set()
    temp_visited = set()

    def has_cycle(task_id: UUID) -> bool:
        if task_id in temp_visited:
            return True

        if task_id in visited:
            return False

        temp_visited.add(task_id)
        visited.add(task_id)

        task = task_map[task_id]
        for dependency in task.dependencies:
            if has_cycle(dependency.task_id):
                return True

        temp_visited.remove(task_id)
        return False

    for task_id in task_map:
        if task_id not in visited and has_cycle(task_id):
            return False, f"Circular dependency detected involving task {task_id}"

    return True, None


def validate_task_dependencies(tasks: list[Task]) -> ValidationResult:
    """Validate task dependencies within a list of tasks.

    This function checks that:
    1. All dependency task_ids reference existing tasks
    2. There are no circular dependencies

    Args:
        tasks: The list of tasks to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    # Create a map of task_id to task for quick lookup
    task_map = {task.task_id: task for task in tasks}

    # Check that all dependency task_ids reference existing tasks
    is_valid, error = _check_dependency_references(tasks, task_map)
    if not is_valid:
        return False, error

    # Check for circular dependencies
    is_valid, error = _check_circular_dependencies(task_map)
    if not is_valid:
        return False, error

    return True, None


def _validate_dict_required_fields(data: dict[str, Any]) -> ValidationResult:
    """Validate required fields in a dictionary for task conversion.

    Args:
        data: The dictionary to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    # Check required fields
    if "description" not in data:
        return False, "Missing required field: description"

    if not isinstance(data.get("description"), str):
        return False, f"description must be a string, got {type(data.get('description'))}"

    return True, None


def _validate_dict_enum_fields(data: dict[str, Any]) -> ValidationResult:
    """Validate enum fields in a dictionary for task conversion.

    Args:
        data: The dictionary to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    # Check enum fields if present
    if "priority" in data:
        try:
            TaskPriority(data["priority"])
        except ValueError:
            return False, f"Invalid priority value: {data['priority']}"

    if "status" in data:
        try:
            TaskStatus(data["status"])
        except ValueError:
            return False, f"Invalid status value: {data['status']}"

    if "complexity" in data:
        try:
            TaskComplexity(data["complexity"])
        except ValueError:
            return False, f"Invalid complexity value: {data['complexity']}"

    return True, None


def _validate_dict_dependencies(data: dict[str, Any]) -> ValidationResult:
    """Validate dependencies in a dictionary for task conversion.

    Args:
        data: The dictionary to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    # Check dependencies if present
    if "dependencies" in data:
        if not isinstance(data["dependencies"], list):
            return False, f"dependencies must be a list, got {type(data['dependencies'])}"

        for i, dep in enumerate(data["dependencies"]):
            if not isinstance(dep, dict):
                return False, f"Dependency at index {i} must be a dictionary, got {type(dep)}"

            if "task_id" not in dep:
                return False, f"Dependency at index {i} missing required field: task_id"

            if "description" not in dep:
                return False, f"Dependency at index {i} missing required field: description"

    return True, None


def validate_dict_as_task(data: dict[str, Any]) -> ValidationResult:
    """Validate if a dictionary can be converted to a valid Task.

    This function checks if a dictionary has the required fields
    and correct types to be converted to a Task object.

    Args:
        data: The dictionary to validate

    Returns:
        A tuple containing (is_valid, error_message)

    """
    # Validate required fields
    is_valid, error = _validate_dict_required_fields(data)
    if not is_valid:
        return False, error

    # Validate enum fields
    is_valid, error = _validate_dict_enum_fields(data)
    if not is_valid:
        return False, error

    # Validate dependencies
    is_valid, error = _validate_dict_dependencies(data)
    if not is_valid:
        return False, error

    return True, None
