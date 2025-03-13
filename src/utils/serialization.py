"""Task serialization utilities.

This module provides utilities for serializing and deserializing task objects
for interchange between agents. It handles conversion between Task objects and
JSON-serializable dictionaries, ensuring proper handling of UUID fields and
nested objects.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from typing import Any

from src.common_types.enums import AgentRole
from src.common_types.task_types import Task, TaskComplexity, TaskDependency, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


def serialize_task(task: Task) -> dict[str, Any]:
    """Serialize a Task object to a JSON-serializable dictionary.

    Args:
        task: The Task object to serialize.

    Returns:
        A JSON-serializable dictionary representation of the task.

    """
    # Convert Task object to dict
    task_dict = asdict(task)

    # Convert UUID objects to strings for JSON serialization
    task_dict["task_id"] = str(task.task_id)
    if task.parent_task_id:
        task_dict["parent_task_id"] = str(task.parent_task_id)
    task_dict["subtasks"] = [str(subtask_id) for subtask_id in task.subtasks]
    task_dict["dependencies"] = [
        {
            "task_id": str(dep.task_id),
            "description": dep.description,
            "is_blocking": dep.is_blocking,
        }
        for dep in task.dependencies
    ]

    # Convert enum values to strings
    if task.assigned_role:
        task_dict["assigned_role"] = task.assigned_role.value

    task_dict["priority"] = task.priority.value
    task_dict["status"] = task.status.value
    task_dict["complexity"] = task.complexity.value

    return task_dict


def deserialize_task(task_dict: dict[str, Any]) -> Task:
    """Deserialize a dictionary into a Task object.

    Args:
        task_dict: A dictionary representation of a task.

    Returns:
        A Task object.

    Raises:
        ValueError: If the task dictionary is invalid.

    """
    try:
        # Create a copy to avoid modifying the original
        task_data = task_dict.copy()

        # Convert string UUIDs back to UUID objects
        if "task_id" in task_data:
            task_data["task_id"] = uuid.UUID(task_data["task_id"])
        if task_data.get("parent_task_id"):
            task_data["parent_task_id"] = uuid.UUID(task_data["parent_task_id"])
        if "subtasks" in task_data:
            task_data["subtasks"] = [uuid.UUID(subtask_id) for subtask_id in task_data["subtasks"]]

        # Convert dependencies
        if "dependencies" in task_data:
            task_data["dependencies"] = [
                TaskDependency(
                    task_id=uuid.UUID(dep["task_id"]),
                    description=dep["description"],
                    is_blocking=dep.get("is_blocking", True),
                )
                for dep in task_data["dependencies"]
            ]

        # Convert string enum values back to enum objects
        if "priority" in task_data:
            task_data["priority"] = TaskPriority(task_data["priority"])
        if "status" in task_data:
            task_data["status"] = TaskStatus(task_data["status"])
        if "complexity" in task_data:
            task_data["complexity"] = TaskComplexity(task_data["complexity"])
        if task_data.get("assigned_role"):
            task_data["assigned_role"] = AgentRole(task_data["assigned_role"])

        return Task(**task_data)
    except (ValueError, KeyError, TypeError) as e:
        logger.exception("Error deserializing task: %s", e)
        msg = f"Invalid task dictionary: {e}"
        raise ValueError(msg) from e


def serialize_task_list(tasks: list[Task]) -> str:
    """Serialize a list of Task objects to a JSON string.

    Args:
        tasks: The list of Task objects to serialize.

    Returns:
        A JSON string representation of the tasks.

    """
    task_dicts = [serialize_task(task) for task in tasks]
    return json.dumps(task_dicts)


def deserialize_task_list(json_str: str) -> list[Task]:
    """Deserialize a JSON string into a list of Task objects.

    Args:
        json_str: A JSON string representation of tasks.

    Returns:
        A list of Task objects.

    Raises:
        ValueError: If the JSON string is invalid.

    """
    try:
        task_dicts = json.loads(json_str)
        if not isinstance(task_dicts, list):
            msg = "Expected a JSON array of tasks"
            raise TypeError(msg)
        return [deserialize_task(task_dict) for task_dict in task_dicts]
    except json.JSONDecodeError as e:
        logger.exception("Error decoding JSON: %s", e)
        msg = f"Invalid JSON string: {e}"
        raise ValueError(msg) from e


def task_to_json(task: Task) -> str:
    """Convert a Task object to a JSON string.

    Args:
        task: The Task object to convert.

    Returns:
        A JSON string representation of the task.

    """
    return json.dumps(serialize_task(task))


def json_to_task(json_str: str) -> Task:
    """Convert a JSON string to a Task object.

    Args:
        json_str: A JSON string representation of a task.

    Returns:
        A Task object.

    Raises:
        ValueError: If the JSON string is invalid.

    """
    try:
        task_dict = json.loads(json_str)
        return deserialize_task(task_dict)
    except json.JSONDecodeError as e:
        logger.exception("Error decoding JSON: %s", e)
        msg = f"Invalid JSON string: {e}"
        raise ValueError(msg) from e
