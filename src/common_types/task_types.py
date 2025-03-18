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
    from src.common_types.enums import AgentRole, ExecutionStage, VerificationStatus

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


class ParallelizationStrategy(str, Enum):
    """Task parallelization strategy enumeration.

    These strategies determine how subtasks should be executed:
    - SEQUENTIAL: Execute subtasks one after another (default)
    - PARALLEL_ALL: Execute all subtasks in parallel
    - PARALLEL_INDEPENDENT: Execute only independent subtasks in parallel
    - PARALLEL_GROUPS: Execute groups of related subtasks in parallel
    - PARALLEL_DEPENDENCIES: Execute subtasks based on dependencies
    """

    SEQUENTIAL = "sequential"
    PARALLEL_ALL = "parallel_all"
    PARALLEL_INDEPENDENT = "parallel_independent"
    PARALLEL_GROUPS = "parallel_groups"
    PARALLEL_DEPENDENCIES = "parallel_dependencies"


@dataclass
class TaskDependency:
    """Task dependency information.

    This class represents a dependency relationship between tasks.
    """

    task_id: UUID
    description: str
    is_blocking: bool = True  # If True, dependent task cannot start until this is completed


@dataclass
class ParallelizationGroup:
    """Parallelization group information.

    This class represents a group of tasks that can be executed in parallel.
    """

    group_id: UUID = field(default_factory=uuid4)
    task_ids: list[UUID] = field(default_factory=list)
    task_indices: list[int] = field(default_factory=list)
    description: str = ""
    name: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM


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

    # Execution tracking fields
    execution_stage: ExecutionStage | None = None
    verification_status: VerificationStatus | None = None
    execution_attempts: int = 0
    execution_logs: list[str] = field(default_factory=list)
    verification_details: dict[str, Any] = field(default_factory=dict)
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    # Parallelization fields
    parallelization_strategy: ParallelizationStrategy = ParallelizationStrategy.SEQUENTIAL
    parallelization_groups: list[ParallelizationGroup] = field(default_factory=list)
    max_parallel_tasks: int | None = None
    is_parallelizable: bool = False
