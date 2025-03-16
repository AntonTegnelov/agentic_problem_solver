"""Message schemas for task execution reporting.

This module defines message schemas for reporting task execution progress and completion.
These schemas are used by executor agents to communicate their progress and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

from src.common_types.enums import ExecutionStage, VerificationStatus

if TYPE_CHECKING:
    from uuid import UUID


class ProgressStatus(str, Enum):
    """Status of task progress.

    These statuses represent the current state of task execution:
    - STARTED: Task execution has begun
    - IN_PROGRESS: Task is actively being worked on
    - BLOCKED: Task execution is blocked by dependencies or issues
    - COMPLETED: Task has been completed successfully
    - FAILED: Task execution has failed
    """

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressReport:
    """Progress report for task execution.

    This message is sent by executor agents to report their progress on a task.
    It includes information about the current execution stage, progress percentage,
    and any blockers or issues encountered.
    """

    task_id: UUID
    status: ProgressStatus
    execution_stage: ExecutionStage
    progress_percentage: float
    timestamp: datetime = field(default_factory=datetime.now)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate progress report fields after initialization."""
        # Ensure progress percentage is between 0 and 100
        if not 0 <= self.progress_percentage <= 100:
            self.progress_percentage = max(0, min(100, self.progress_percentage))


@dataclass
class CompletionReport:
    """Completion report for task execution.

    This message is sent by executor agents when a task is completed or failed.
    It includes the final status, verification results, and any outputs or artifacts
    produced during execution.
    """

    task_id: UUID
    status: ProgressStatus
    verification_status: VerificationStatus
    execution_stage: ExecutionStage
    execution_time: float  # Time in seconds
    timestamp: datetime = field(default_factory=datetime.now)
    verification_details: list[dict[str, Any]] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate completion report fields after initialization."""
        # Ensure execution time is non-negative
        self.execution_time = max(self.execution_time, 0)

        # Ensure status is either COMPLETED or FAILED
        if self.status not in (ProgressStatus.COMPLETED, ProgressStatus.FAILED):
            self.status = (
                ProgressStatus.COMPLETED
                if self.verification_status in (VerificationStatus.PASSED, VerificationStatus.PARTIAL)
                else ProgressStatus.FAILED
            )


# Type alias for execution reports
ExecutionReport = Union[ProgressReport, CompletionReport]
