"""Logging utilities."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Constants
MAX_TASK_DESCRIPTION_LENGTH = 100
MAX_TASK_TREE_DEPTH = 10  # Maximum depth for task tree visualization
MAX_DESCRIPTION_DISPLAY_LENGTH = 50
MAX_ID_DISPLAY_LENGTH = 8


def setup_logging(
    *,  # Force keyword arguments
    level: int = logging.INFO,
    log_file: str | None = None,
    verbose: bool = False,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level.
        log_file: Path to log file.
        verbose: Whether to enable verbose logging.

    """
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    console_handler.setFormatter(formatter)

    # Add console handler to root logger
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set verbose logging
    if verbose:
        root_logger.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.

    """
    return logging.getLogger(name)


@dataclass
class DelegationInfo:
    """Information about a delegation decision."""

    source_agent_id: str
    target_agent_id: str
    task: str
    reason: str
    additional_info: dict[str, Any] | None = None


@dataclass
class HierarchicalDelegationInfo:
    """Information about a hierarchical delegation decision."""

    source_agent_id: str
    parent_task_id: str | None = None
    parent_task: str | None = None
    total_subtasks: int = 0
    successful_delegations: int = 0
    failed_delegations: int = 0
    error: str | None = None
    additional_info: dict[str, Any] | None = None


def log_delegation_decision(
    logger: logging.Logger,
    delegation_info: DelegationInfo,
) -> None:
    """Log a delegation decision made by an agent.

    Args:
        logger: Logger instance.
        delegation_info: Information about the delegation decision.

    """
    log_data = asdict(delegation_info)
    logger.info("Delegation decision: %s", json.dumps(log_data))


def log_hierarchical_delegation(delegation_info: HierarchicalDelegationInfo) -> None:
    """Log detailed information about hierarchical task delegation.

    Args:
        delegation_info: Information about the hierarchical delegation decision.

    """
    logger = logging.getLogger("agent.delegation")

    # Create a serializable log data object
    log_data = asdict(delegation_info)

    # Log summary
    if not delegation_info.error:
        logger.info(
            "Hierarchical delegation from %s: %d subtasks delegated (%d successful, %d failed)",
            delegation_info.source_agent_id,
            delegation_info.total_subtasks,
            delegation_info.successful_delegations,
            delegation_info.failed_delegations,
        )
    else:
        logger.error(
            "Hierarchical delegation from %s failed: %s",
            delegation_info.source_agent_id,
            delegation_info.error,
        )

    # Log details in debug level
    try:
        logger.debug("Hierarchical delegation details: %s", json.dumps(log_data))
    except TypeError as e:
        logger.debug("Hierarchical delegation details (partial, serialization error: %s): %s", str(e), str(log_data))


def render_task_tree(task: str, subtasks: list, max_depth: int = MAX_TASK_TREE_DEPTH) -> str:
    """Render a hierarchical task tree as ASCII visualization.

    Args:
        task: The parent task description
        subtasks: List of subtask objects (Task objects or dictionaries)
        max_depth: Maximum depth to render (default: 10)

    Returns:
        String representation of task tree in ASCII format

    """
    lines = [f"Task: {task}"]

    if not subtasks:
        lines.append("  (No subtasks)")
        return "\n".join(lines)

    # Render the subtasks
    for i, subtask in enumerate(subtasks):
        prefix = "└── " if i == len(subtasks) - 1 else "├── "

        # Handle both Task objects and dictionaries
        if hasattr(subtask, "description"):
            # It's a Task object
            description = subtask.description
            task_id = str(subtask.task_id) if hasattr(subtask, "task_id") else ""
            complexity = subtask.complexity.name if hasattr(subtask, "complexity") else ""
            nested_subtasks = getattr(subtask, "subtasks", [])
        else:
            # It's a dictionary
            description = subtask.get("description", "")
            task_id = subtask.get("id", "")
            complexity = subtask.get("complexity", "")
            nested_subtasks = subtask.get("subtasks", [])

        # Truncate description if too long
        if len(description) > MAX_DESCRIPTION_DISPLAY_LENGTH:
            description = description[: MAX_DESCRIPTION_DISPLAY_LENGTH - 3] + "..."

        # Format each subtask line
        subtask_line = f"{prefix}{description}"
        if complexity:
            subtask_line += f" ({complexity})"
        if task_id:
            subtask_line += (
                f" [ID: {task_id[:MAX_ID_DISPLAY_LENGTH]}...]"
                if len(task_id) > MAX_ID_DISPLAY_LENGTH
                else f" [ID: {task_id}]"
            )

        lines.append(subtask_line)

        # Render nested subtasks if any
        if nested_subtasks and max_depth > 1:
            nested_prefix = "    " if i == len(subtasks) - 1 else "│   "
            nested_tree = _render_task_node(nested_subtasks, max_depth - 1, nested_prefix)
            lines.extend(nested_tree)

    return "\n".join(lines)


def _render_task_node(subtasks: list, depth: int, prefix: str) -> list:
    """Render a node in the task tree.

    Args:
        subtasks: List of subtask objects (Task objects or dictionaries)
        depth: Current depth in the tree
        prefix: String prefix for indentation

    Returns:
        List of formatted lines for this node

    """
    if depth <= 0 or not subtasks:
        return []

    lines = []
    for i, subtask in enumerate(subtasks):
        is_last = i == len(subtasks) - 1

        # Handle both Task objects and dictionaries
        if hasattr(subtask, "description"):
            # It's a Task object
            description = subtask.description
            task_id = str(subtask.task_id) if hasattr(subtask, "task_id") else ""
            complexity = subtask.complexity.name if hasattr(subtask, "complexity") else ""
            nested_subtasks = getattr(subtask, "subtasks", [])
        else:
            # It's a dictionary
            description = subtask.get("description", "")
            task_id = subtask.get("id", "")
            complexity = subtask.get("complexity", "")
            nested_subtasks = subtask.get("subtasks", [])

        # Truncate description if too long
        if len(description) > MAX_DESCRIPTION_DISPLAY_LENGTH:
            description = description[: MAX_DESCRIPTION_DISPLAY_LENGTH - 3] + "..."

        # Create the line prefix
        line_prefix = prefix + ("└── " if is_last else "├── ")

        # Format the subtask line
        subtask_line = f"{line_prefix}{description}"
        if complexity:
            subtask_line += f" ({complexity})"
        if task_id:
            subtask_line += (
                f" [ID: {task_id[:MAX_ID_DISPLAY_LENGTH]}...]"
                if len(task_id) > MAX_ID_DISPLAY_LENGTH
                else f" [ID: {task_id}]"
            )

        lines.append(subtask_line)

        # Render nested subtasks if any
        if nested_subtasks and depth > 1:
            next_prefix = prefix + ("    " if is_last else "│   ")
            nested_lines = _render_task_node(nested_subtasks, depth - 1, next_prefix)
            lines.extend(nested_lines)

    return lines
