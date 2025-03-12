"""Task type definitions.

This module contains task-related types used for task decomposition,
delegation, and tracking throughout the agent hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from src.common_types.enums import AgentRole

T = TypeVar("T")


class TaskStatus(str, Enum):
    """Task status enumeration.

    These statuses represent the current state of a task:
    - PENDING: Task has been created but not yet started
    - IN_PROGRESS: Task is currently being worked on
    - BLOCKED: Task is blocked by dependencies or other issues
    - COMPLETED: Task has been successfully completed
    - FAILED: Task has failed to complete successfully
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    """Task priority enumeration.

    These priorities represent the importance of a task:
    - LOW: Task is not urgent and can be deferred
    - MEDIUM: Task has normal priority
    - HIGH: Task is important and should be prioritized
    - CRITICAL: Task is extremely important and should be done immediately
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskComplexity(str, Enum):
    """Task complexity enumeration.

    These complexity levels help determine appropriate delegation:
    - SIMPLE: Task can be directly executed without further decomposition
    - MODERATE: Task may benefit from some planning but is relatively straightforward
    - COMPLEX: Task requires significant planning and decomposition
    - VERY_COMPLEX: Task requires multiple levels of planning and decomposition
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class TaskDependency:
    """Task dependency information.

    This class represents a dependency relationship between tasks.
    """

    task_id: UUID
    description: str
    is_blocking: bool = True  # If True, dependent task cannot start until this is completed


@dataclass
class Task:
    """Task information.

    This class represents a task in the system, with fields for description,
    priority, dependencies, status, and complexity estimation to support
    delegation decisions.
    """

    description: str
    task_id: UUID = field(default_factory=uuid4)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    complexity: TaskComplexity = TaskComplexity.MODERATE
    dependencies: list[TaskDependency] = field(default_factory=list)
    parent_task_id: UUID | None = None
    subtasks: list[UUID] = field(default_factory=list)
    assigned_role: AgentRole | None = None
    assigned_agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    completed_at: float | None = None
