"""Agent state module."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.common_types import AgentNotFoundError, ConfigError
from src.common_types.enums import AgentStep, ExecutionStage, VerificationStatus
from src.common_types.result_types import Result
from src.common_types.result_types import Result as StepResult
from src.common_types.task_types import Task, TaskComplexity, TaskDependency, TaskPriority, TaskStatus
from src.messages.utils import (
    get_message_at_index,
    get_metadata_at_index,
    set_metadata_at_index,
)

if TYPE_CHECKING:
    from src.agent.agent_types.agent_types import Agent
    from src.common_types.message_types import Message

T = TypeVar("T")

# Constants
MAX_EXECUTION_ATTEMPTS = 3
MIN_CIRCULAR_PATH_LENGTH = 2  # Minimum length for a circular dependency path


@dataclass
class Context:
    """Agent context."""

    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        """Compare context instances.

        Args:
            other: Other context instance to compare with.

        Returns:
            True if contexts are equal.

        """
        if not isinstance(other, Context):
            return NotImplemented

        return self.data == other.data and self.metadata == other.metadata

    def validate(self) -> bool:
        """Validate context data.

        Returns:
            True if context is valid.

        Raises:
            ConfigError: If context validation fails.

        """
        # Check for required metadata
        required_metadata = ["created_at", "updated_at"]
        for key in required_metadata:
            if key not in self.metadata:
                msg = f"Missing required metadata: {key}"
                raise ConfigError(msg)

        return True

    def track_changes(self) -> None:
        """Track context changes by updating metadata."""
        self.metadata["updated_at"] = datetime.now(UTC).isoformat()
        self.metadata["change_count"] = self.metadata.get("change_count", 0) + 1


@runtime_checkable
class StateManager(Protocol):
    """State manager protocol."""

    def get_state(self) -> AgentState:
        """Get the current state.

        Returns:
            The current state.

        """
        ...

    def set_state(self, state: AgentState) -> None:
        """Set the current state.

        Args:
            state: The state to set.

        """
        ...

    def clear_state(self) -> None:
        """Clear the current state."""
        ...

    def save_state(self, path: str | None = None) -> str:
        """Save the current state to a file.

        Args:
            path: The path to save the state to. If None, a default path is used.

        Returns:
            The path the state was saved to.

        Raises:
            FileNotFoundError: If the path does not exist.
            PermissionError: If the path is not writable.
            ConfigError: If the state is invalid.

        """
        ...

    def load_state(self, path: str) -> AgentState:
        """Load state from file.

        Args:
            path: Path to state file.

        Returns:
            Loaded state.

        Raises:
            ConfigError: If state loading fails.

        """

        def _raise_file_not_found(path: str) -> None:
            msg = f"State file not found: {path}"
            raise ConfigError(msg)

        try:
            # Check if file exists
            if not Path(path).exists():
                _raise_file_not_found(path)

            # Load from file
            state_dict = json.loads(Path(path).read_text(encoding="utf-8"))

            # Create state from dict
            state = AgentState.from_dict(state_dict)

            # Validate loaded state
            state.validate()

            # Set as current state
            self._state = state
        except Exception as e:
            msg = f"Failed to load state: {e}"
            raise ConfigError(msg) from e
        else:
            return state


@dataclass
class AgentState:
    """Agent state.

    This class manages the state of an agent during execution, including messages,
    context, execution results, and step tracking.
    """

    messages: list[Message] = field(default_factory=list)
    context: Context = field(default_factory=Context)
    execution_result: str = ""
    current_step: AgentStep = field(default=AgentStep.UNDERSTAND)
    step_count: int = field(default=0)
    task_completed: bool = field(default=False)
    error: str | None = field(default=None)
    step_results: dict[str, StepResult[Any]] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = field(default=None)
    child_ids: list[str] = field(default_factory=list)
    _agents: dict[str, Agent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize context metadata if empty."""
        if not self.context.metadata:
            self.context.metadata = {
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "change_count": 0,
            }

    def __eq__(self, other: object) -> bool:
        """Check if two instances are equal.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equal, False otherwise.

        """
        if not isinstance(other, AgentState):
            return False
        return (
            self.messages == other.messages
            and self.context == other.context
            and self.execution_result == other.execution_result
            and self.current_step == other.current_step
            and self.step_count == other.step_count
            and self.task_completed == other.task_completed
            and self.error == other.error
            and self.step_results == other.step_results
            and self.created_at == other.created_at
            and self.updated_at == other.updated_at
            and self.agent_id == other.agent_id
            and self.parent_id == other.parent_id
            and self.child_ids == other.child_ids
        )

    def clear(self) -> None:
        """Clear the state."""
        self.messages.clear()
        self.context = Context()
        self.context.metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "change_count": 0,
        }
        self.execution_result = ""
        self.step_count = 0
        self.task_completed = False
        self.error = None
        self.current_step = AgentStep.UNDERSTAND
        self.step_results.clear()
        self.updated_at = datetime.now(UTC).isoformat()
        # Generate a new agent_id when clearing the state
        self.agent_id = str(uuid.uuid4())
        self.parent_id = None
        self.child_ids.clear()
        self._agents.clear()

    def get_messages(self) -> list[Message]:
        """Get all messages in the state.

        Returns:
            List of messages.

        """
        return self.messages

    def add_message(self, message: Message) -> None:
        """Add message to state.

        Args:
            message: Message to add.

        """
        self.messages.append(message)
        self.updated_at = datetime.now(UTC).isoformat()

    def get_message(self, index: int) -> Message:
        """Get message at index.

        Args:
            index: Message index.

        Returns:
            Message at index.

        Raises:
            IndexError: If index is out of range.

        """
        return get_message_at_index(self.messages, index)

    def get_message_metadata(
        self,
        index: int,
        key: str,
        default: T | None = None,
    ) -> T | None:
        """Get metadata from a message at the specified index.

        Args:
            index: Message index.
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Message metadata value.

        """
        return get_metadata_at_index(self.messages, index, key, default)

    def set_message_metadata(
        self,
        index: int,
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Set metadata for a message at the specified index.

        Args:
            index: Message index.
            key: Metadata key.
            value: Metadata value.

        """
        set_metadata_at_index(self.messages, index, key, value)
        self.updated_at = datetime.now(UTC).isoformat()

    def get_context(self, key: str, default: T | None = None) -> T | None:
        """Get context value.

        Args:
            key: Context key.
            default: Default value if key not found.

        Returns:
            Context value.

        """
        return self.context.data.get(key, default)

    def set_context(self, key: str, value: T) -> None:
        """Set context value.

        Args:
            key: Context key.
            value: Context value.

        """
        self.context.data[key] = value
        self.context.track_changes()
        self.updated_at = datetime.now(UTC).isoformat()

    def validate(self) -> bool:
        """Validate state.

        Returns:
            True if state is valid.

        Raises:
            ConfigError: If state validation fails.

        """
        # Validate context
        self.context.validate()

        # Validate step
        if self.current_step not in AgentStep:
            msg = f"Invalid step: {self.current_step}"
            raise ConfigError(msg)

        # Validate timestamps
        try:
            datetime.fromisoformat(self.created_at)
            datetime.fromisoformat(self.updated_at)
        except ValueError as e:
            msg = f"Invalid timestamp format: {e}"
            raise ConfigError(msg) from e

        # Validate agent_id
        if not self.agent_id:
            msg = "Agent ID cannot be empty"
            raise ConfigError(msg)

        return True

    def record_step_result(self, step: AgentStep, result: StepResult[Any]) -> None:
        """Record result for a step.

        Args:
            step: Step to record result for.
            result: Step result.

        """
        self.step_results[step.value] = result
        self.updated_at = datetime.now(UTC).isoformat()

    def get_step_result(self, step: AgentStep) -> StepResult[Any] | None:
        """Get result for a step.

        Args:
            step: Step to get result for.

        Returns:
            Step result or None if not found.

        """
        return self.step_results.get(step.value)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary.

        Returns:
            State as dictionary.

        """
        # Convert to dict using asdict
        state_dict = asdict(self)

        # Handle message serialization
        state_dict["messages"] = []
        for msg in self.messages:
            # Determine role based on message type
            if isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            elif isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "unknown"

            state_dict["messages"].append({"role": role, "content": msg.content})

        # Handle step results serialization
        state_dict["step_results"] = {
            step: {
                "success": result.success,
                "error": result.error,
                "data": str(result.data),  # Convert data to string for serialization
            }
            for step, result in self.step_results.items()
        }

        return state_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Create AgentState from dictionary.

        Args:
            data: Dictionary data.

        Returns:
            AgentState instance.

        """
        # Handle message conversion
        messages = []
        for msg_data in data.get("messages", []):
            role = msg_data.get("role", "")
            content = msg_data.get("content", "")

            # Convert to appropriate message type
            if role == "human":
                msg = HumanMessage(content=content)
            elif role == "ai":
                msg = AIMessage(content=content)
            elif role == "system":
                msg = SystemMessage(content=content)
            elif role == "tool":
                msg = ToolMessage(content=content)
            else:
                # Default to system message
                msg = SystemMessage(content=content)
            messages.append(msg)

        # Convert context
        context_data = data.get("context", {})
        context = Context(
            data=context_data.get("data", {}),
            metadata=context_data.get("metadata", {}),
        )

        # Convert step results
        step_results = {}
        for step_key, result_data in data.get("step_results", {}).items():
            success = result_data.get("success", False)
            error = result_data.get("error", "")
            data_value = result_data.get("data", "")
            message = result_data.get("message", "")

            result = StepResult.success(data_value, message) if success else StepResult.failure(error, message)
            step_results[step_key] = result

        # Create the AgentState
        return cls(
            messages=messages,
            context=context,
            execution_result=data.get("execution_result", ""),
            current_step=AgentStep(
                data.get("current_step", AgentStep.UNDERSTAND.value),
            ),
            step_count=data.get("step_count", 0),
            task_completed=data.get("task_completed", False),
            error=data.get("error"),
            step_results=step_results,
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            agent_id=data.get("agent_id", ""),
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
        )

    def get_agent_for_step(self, step: AgentStep | str) -> Agent:
        """Get agent for step.

        Args:
            step: Step to get agent for.

        Returns:
            Agent for step.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        # Convert string step name to enum if needed
        step_name = step if isinstance(step, str) else step.value

        # Look up the agent ID for this step
        agent_id = self.get_agent_id_for_step(step_name)
        if not agent_id:
            msg = f"No agent found for step {step_name}"
            raise AgentNotFoundError(msg)

        # Get the agent from the registry
        return self.get_agent(agent_id)

    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register agent.

        Args:
            agent_id: Agent ID.
            agent: Agent instance.

        """
        self._agents[agent_id] = agent

    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent instance.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        return self._agents[agent_id]

    def get_registered_agents(self) -> dict[str, Agent]:
        """Get all registered agents.

        Returns:
            Dictionary of agent ID to agent instance.

        """
        return self._agents

    def get_agent_id_for_step(self, step_name: str) -> str | None:
        """Get agent ID for step.

        Args:
            step_name: Step name.

        Returns:
            Agent ID or None if not found.

        """
        # For tests, if we have a registered agent, return the first one
        if self._agents and len(self._agents) == 1:
            return next(iter(self._agents.keys()))

        # In a more complex implementation, this would map steps to specific agents
        # For now, we log the step_name for debugging purposes
        import logging

        logging.debug("Looking for agent to handle step: %s", step_name)

        # Return the current agent ID if we have agents registered
        return self.agent_id if self._agents else None

    def process_step(self, step: AgentStep) -> Result:
        """Process a step.

        Args:
            step: Step to process.

        Returns:
            Step result.

        Raises:
            ConfigError: If no agent registered.

        """
        agent = self.get_agent()
        if not agent:
            msg = "No agent registered"
            raise ConfigError(msg)

        # Get step prompt and process
        # Import here to avoid circular imports
        from src.agent.steps import get_step_prompt

        prompt = get_step_prompt(step)
        self.add_message(prompt)
        return agent.process(prompt)

    def get_tasks(self) -> list[Task]:
        """Get all tasks in the state.

        Returns:
            List of tasks.

        """
        return self.context.data.get("tasks", [])

    def add_task(self, task: Task) -> None:
        """Add a task to the state.

        Args:
            task: Task to add.

        """
        tasks = self.get_tasks()
        # Convert Task object to dict for storage
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

        # Handle execution stage and verification status
        if task.execution_stage:
            task_dict["execution_stage"] = task.execution_stage.value
        if task.verification_status:
            task_dict["verification_status"] = task.verification_status.value

        tasks.append(task_dict)
        self.set_context("tasks", tasks)
        self.updated_at = datetime.now(UTC).isoformat()

        # After adding a task, check for and resolve any potential deadlocks
        if task.dependencies:
            self.detect_and_resolve_deadlocks()

    def get_task_by_id(self, task_id: uuid.UUID) -> Task | None:
        """Get task by ID.

        Args:
            task_id: Task ID.

        Returns:
            Task or None if not found.

        """
        tasks = self.get_tasks()
        task_id_str = str(task_id)

        for task_dict in tasks:
            if task_dict["task_id"] == task_id_str:
                return self._convert_dict_to_task(task_dict)

        return None

    def update_task(self, task: Task) -> None:
        """Update a task in the state.

        Args:
            task: Task to update.

        """
        tasks = self.get_tasks()
        task_id_str = str(task.task_id)

        # Check if dependencies exist and store them for comparison
        original_task = self.get_task_by_id(task.task_id)

        # Check if dependencies have changed
        dependencies_changed = self._check_dependencies_changed(original_task, task)

        for i, task_dict in enumerate(tasks):
            if task_dict["task_id"] == task_id_str:
                # Convert Task object to dict for storage
                updated_task = self._prepare_task_for_storage(task)

                tasks[i] = updated_task
                self.set_context("tasks", tasks)
                self.updated_at = datetime.now(UTC).isoformat()

                # If dependencies changed, check for and resolve any potential deadlocks
                if dependencies_changed and task.dependencies:
                    self.detect_and_resolve_deadlocks()

                return

        # If task not found, add it
        self.add_task(task)

    def _check_dependencies_changed(self, original_task: Task | None, new_task: Task) -> bool:
        """Check if task dependencies have changed.

        Args:
            original_task: Original task or None if it doesn't exist
            new_task: New task to compare with

        Returns:
            True if dependencies have changed, False otherwise

        """
        if not original_task:
            return bool(new_task.dependencies)

        original_dependencies = original_task.dependencies

        # Simple check if number of dependencies changed
        if len(original_dependencies) != len(new_task.dependencies):
            return True

        # Check if any dependency details changed
        for i, dep in enumerate(new_task.dependencies):
            if i >= len(original_dependencies) or dep != original_dependencies[i]:
                return True

        return False

    def update_task_status_based_on_dependencies(self, task_id: uuid.UUID) -> None:
        """Update task status based on dependencies.

        Args:
            task_id: Task ID.

        """
        task = self.get_task_by_id(task_id)
        if not task:
            return

        # Only update if task is pending or blocked
        if task.status not in [TaskStatus.PENDING, TaskStatus.BLOCKED]:
            return

        if self.is_task_blocked_by_dependencies(task_id):
            task.status = TaskStatus.BLOCKED
        # If task was blocked but dependencies are now resolved, set to pending
        elif task.status == TaskStatus.BLOCKED:
            task.status = TaskStatus.PENDING

        self.update_task(task)

    def update_dependent_tasks(self, task_id: uuid.UUID) -> None:
        """Update status of tasks that depend on the given task.

        Args:
            task_id: Task ID.

        """
        tasks = self.get_tasks()
        task_id_str = str(task_id)

        for task_dict in tasks:
            # Check if this task depends on the given task
            for dep in task_dict.get("dependencies", []):
                if dep.get("task_id") == task_id_str:
                    # Update the status of this dependent task
                    dependent_task_id = uuid.UUID(task_dict["task_id"])
                    self.update_task_status_based_on_dependencies(dependent_task_id)

    def track_delegated_task_progress(
        self,
        task_id: uuid.UUID,
        progress: float,
        status_message: str | None = None,
    ) -> None:
        """Track progress of a delegated task.

        Args:
            task_id: Task ID.
            progress: Progress percentage (0.0 to 1.0).
            status_message: Optional status message.

        """
        task = self.get_task_by_id(task_id)
        if not task:
            return

        # Update task metadata with progress information
        if "progress_tracking" not in task.metadata:
            task.metadata["progress_tracking"] = {}

        progress_tracking = task.metadata["progress_tracking"]
        progress_tracking["progress_percentage"] = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        progress_tracking["last_updated"] = datetime.now(UTC).isoformat()

        if status_message:
            progress_tracking["status_message"] = status_message

        # Add progress history if it doesn't exist
        if "progress_history" not in progress_tracking:
            progress_tracking["progress_history"] = []

        # Add current progress to history
        progress_tracking["progress_history"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "progress": progress_tracking["progress_percentage"],
                "status_message": status_message if status_message else "Progress update",
            },
        )

        # Update task status based on progress
        if progress >= 1.0:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC).timestamp()
        elif progress > 0.0:
            task.status = TaskStatus.IN_PROGRESS

        # Update the task
        self.update_task(task)

        # If this is a parent task, update its progress based on subtasks
        if task.parent_task_id:
            self.update_parent_task_progress(task.parent_task_id)

    def update_parent_task_progress(self, parent_task_id: uuid.UUID) -> None:
        """Update progress of a parent task based on its subtasks.

        Args:
            parent_task_id: Parent task ID.

        """
        parent_task = self.get_task_by_id(parent_task_id)
        if not parent_task or not parent_task.subtasks:
            return

        # Use the enhanced rollup calculation
        rollup_progress = self.calculate_rollup_progress(parent_task_id)

        # Update parent task progress with the calculated rollup
        self.track_delegated_task_progress(
            parent_task_id,
            rollup_progress["progress"],
            rollup_progress["status_message"],
        )

        # If parent task has a parent, update that as well
        if parent_task.parent_task_id:
            self.update_parent_task_progress(parent_task.parent_task_id)

    def calculate_rollup_progress(self, task_id: uuid.UUID) -> dict[str, Any]:
        """Calculate rollup progress for a task based on its subtasks.

        This method provides an enhanced progress calculation that takes into account:
        - Task priorities (higher priority tasks have more weight)
        - Task complexity (more complex tasks have more weight)
        - Task status (completed, in progress, blocked, etc.)
        - Dependency relationships between tasks

        Args:
            task_id: Task ID.

        Returns:
            Dictionary with rollup progress information including:
            - progress: Overall progress value (0.0 to 1.0)
            - status_message: Status message describing the progress
            - weighted_progress: Progress weighted by priority and complexity
            - critical_path_progress: Progress of tasks on the critical path
            - blocking_tasks: List of tasks blocking progress

        """
        task = self.get_task_by_id(task_id)
        if not task or not task.subtasks:
            return {
                "progress": 0.0,
                "status_message": "No subtasks found",
                "weighted_progress": 0.0,
                "critical_path_progress": 0.0,
                "blocking_tasks": [],
            }

        # Process subtasks and collect progress data
        progress_data = self._collect_subtask_progress_data(task)

        # Calculate final progress metrics
        return self._calculate_final_progress_metrics(progress_data)

    def _collect_subtask_progress_data(self, task: Task) -> dict[str, Any]:
        """Collect progress data from all subtasks.

        Args:
            task: Parent task containing subtasks

        Returns:
            Dictionary with collected progress data

        """
        # Initialize counters and lists
        total_subtasks = len(task.subtasks)
        completed_subtasks = 0
        in_progress_subtasks = 0
        blocked_subtasks = 0
        failed_subtasks = 0

        # Priority and complexity weights
        priority_weights = {
            TaskPriority.LOW.value: 0.5,
            TaskPriority.MEDIUM.value: 1.0,
            TaskPriority.HIGH.value: 1.5,
            TaskPriority.CRITICAL.value: 2.0,
        }

        complexity_weights = {
            TaskComplexity.SIMPLE.value: 0.75,
            TaskComplexity.MODERATE.value: 1.0,
            TaskComplexity.COMPLEX.value: 1.5,
            TaskComplexity.VERY_COMPLEX.value: 2.0,
        }

        # Track weighted progress
        total_weight = 0.0
        weighted_progress = 0.0

        # Track critical path and blocking tasks
        critical_path_tasks = []
        blocking_tasks = []

        # Process each subtask
        for subtask_id in task.subtasks:
            subtask = self.get_task_by_id(subtask_id)
            if not subtask:
                continue

            # Get priority and complexity weights
            priority = subtask.priority.value if hasattr(subtask.priority, "value") else subtask.priority
            complexity = subtask.complexity.value if hasattr(subtask.complexity, "value") else subtask.complexity

            priority_weight = priority_weights.get(priority, 1.0)
            complexity_weight = complexity_weights.get(complexity, 1.0)

            # Calculate combined weight
            combined_weight = priority_weight * complexity_weight
            total_weight += combined_weight

            # Track task status
            if subtask.status == TaskStatus.COMPLETED:
                completed_subtasks += 1
                weighted_progress += combined_weight
            elif subtask.status == TaskStatus.IN_PROGRESS:
                in_progress_subtasks += 1
                # For in-progress tasks, use their reported progress
                if "progress_tracking" in subtask.metadata:
                    progress = subtask.metadata["progress_tracking"].get("progress_percentage", 0.0)
                    weighted_progress += combined_weight * progress
            elif subtask.status == TaskStatus.BLOCKED:
                blocked_subtasks += 1
                blocking_tasks.append(
                    {
                        "task_id": str(subtask_id),
                        "description": subtask.description,
                        "blockers": subtask.metadata.get("blockers", {}).get("blocking_dependencies", []),
                    },
                )
            elif subtask.status == TaskStatus.FAILED:
                failed_subtasks += 1

            # Check if task is on critical path (has dependencies or dependents)
            if subtask.dependencies or any(self.is_dependent_on(other_id, subtask_id) for other_id in task.subtasks):
                critical_path_tasks.append(subtask)

        return {
            "total_subtasks": total_subtasks,
            "completed_subtasks": completed_subtasks,
            "in_progress_subtasks": in_progress_subtasks,
            "blocked_subtasks": blocked_subtasks,
            "failed_subtasks": failed_subtasks,
            "total_weight": total_weight,
            "weighted_progress": weighted_progress,
            "critical_path_tasks": critical_path_tasks,
            "blocking_tasks": blocking_tasks,
        }

    def _calculate_final_progress_metrics(self, progress_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate final progress metrics based on collected data.

        Args:
            progress_data: Dictionary with collected progress data

        Returns:
            Dictionary with final progress metrics

        """
        # Extract data from progress_data
        total_subtasks = progress_data["total_subtasks"]
        completed_subtasks = progress_data["completed_subtasks"]
        in_progress_subtasks = progress_data["in_progress_subtasks"]
        blocked_subtasks = progress_data["blocked_subtasks"]
        failed_subtasks = progress_data["failed_subtasks"]
        total_weight = progress_data["total_weight"]
        weighted_progress = progress_data["weighted_progress"]
        critical_path_tasks = progress_data["critical_path_tasks"]
        blocking_tasks = progress_data["blocking_tasks"]

        # Calculate normalized weighted progress
        normalized_weighted_progress = weighted_progress / total_weight if total_weight > 0 else 0.0

        # Calculate critical path progress
        critical_path_progress = 0.0
        if critical_path_tasks:
            critical_path_completed = sum(1 for t in critical_path_tasks if t.status == TaskStatus.COMPLETED)
            critical_path_progress = critical_path_completed / len(critical_path_tasks)

        # Calculate overall progress using a weighted combination of metrics
        # - 60% based on weighted task progress
        # - 30% based on critical path progress
        # - 10% based on simple task count progress
        simple_progress = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
        overall_progress = 0.6 * normalized_weighted_progress + 0.3 * critical_path_progress + 0.1 * simple_progress

        # Generate status message
        status_message = (
            f"Progress: {completed_subtasks}/{total_subtasks} tasks completed"
            f" ({in_progress_subtasks} in progress, {blocked_subtasks} blocked, {failed_subtasks} failed)"
        )

        return {
            "progress": overall_progress,
            "status_message": status_message,
            "weighted_progress": normalized_weighted_progress,
            "critical_path_progress": critical_path_progress,
            "simple_progress": simple_progress,
            "completed_subtasks": completed_subtasks,
            "in_progress_subtasks": in_progress_subtasks,
            "blocked_subtasks": blocked_subtasks,
            "failed_subtasks": failed_subtasks,
            "total_subtasks": total_subtasks,
            "blocking_tasks": blocking_tasks,
        }

    def is_dependent_on(self, task_id: uuid.UUID, dependency_id: uuid.UUID) -> bool:
        """Check if a task is dependent on another task.

        Args:
            task_id: Task ID to check.
            dependency_id: Potential dependency task ID.

        Returns:
            True if task_id depends on dependency_id.

        """
        task = self.get_task_by_id(task_id)
        if not task:
            return False

        # Check direct dependencies
        return any(dependency.task_id == dependency_id for dependency in task.dependencies)

    def get_task_progress(self, task_id: uuid.UUID) -> dict[str, Any]:
        """Get progress information for a task.

        Args:
            task_id: Task ID.

        Returns:
            Dictionary with progress information.

        """
        task = self.get_task_by_id(task_id)
        if not task:
            return {"error": "Task not found", "progress": 0.0}

        # Get progress tracking information
        progress_tracking = task.metadata.get("progress_tracking", {})
        progress = progress_tracking.get("progress_percentage", 0.0)

        # If task is completed, progress is 100%
        if task.status == TaskStatus.COMPLETED:
            progress = 1.0

        # Build progress information
        progress_info = {
            "task_id": str(task_id),
            "description": task.description,
            "status": task.status.value if hasattr(task.status, "value") else task.status,
            "progress": progress,
            "last_updated": progress_tracking.get("last_updated", task.updated_at),
            "status_message": progress_tracking.get("status_message", ""),
            "subtasks_progress": [],
        }

        # Add subtask progress if available
        for subtask_id in task.subtasks:
            subtask_progress = self.get_task_progress(subtask_id)
            progress_info["subtasks_progress"].append(subtask_progress)

        return progress_info

    def get_overall_progress(self) -> dict[str, Any]:
        """Get overall progress information for all tasks.

        Returns:
            Dictionary with overall progress information.

        """
        tasks = self.get_tasks()

        # Find root tasks (tasks without parents)
        root_tasks = []
        for task_dict in tasks:
            if not task_dict.get("parent_task_id"):
                task = self._convert_dict_to_task(task_dict)
                root_tasks.append(task)

        # Get progress for each root task
        root_task_progress = []
        for task in root_tasks:
            progress_info = self.get_task_progress(task.task_id)
            root_task_progress.append(progress_info)

        # Calculate overall statistics
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task_dict in tasks if task_dict.get("status") == TaskStatus.COMPLETED.value)
        in_progress_tasks = sum(1 for task_dict in tasks if task_dict.get("status") == TaskStatus.IN_PROGRESS.value)
        blocked_tasks = sum(1 for task_dict in tasks if task_dict.get("status") == TaskStatus.BLOCKED.value)
        pending_tasks = sum(1 for task_dict in tasks if task_dict.get("status") == TaskStatus.PENDING.value)
        failed_tasks = sum(1 for task_dict in tasks if task_dict.get("status") == TaskStatus.FAILED.value)

        # Calculate overall progress percentage
        if total_tasks > 0:
            overall_progress = (completed_tasks / total_tasks) + (in_progress_tasks / total_tasks / 2)
        else:
            overall_progress = 0.0

        return {
            "overall_progress": overall_progress,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "blocked_tasks": blocked_tasks,
            "pending_tasks": pending_tasks,
            "failed_tasks": failed_tasks,
            "root_tasks": root_task_progress,
        }

    def track_task_completion_status(self, task_id: uuid.UUID) -> None:
        """Track completion status of a task.

        This method checks the execution stage and verification status of a task
        to determine if it should be marked as completed or failed.

        Args:
            task_id: Task ID.

        """
        task = self.get_task_by_id(task_id)
        if not task:
            return

        # Check execution stage and verification status
        if task.execution_stage == ExecutionStage.FINALIZING and task.verification_status == VerificationStatus.PASSED:
            # Task is complete
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC).timestamp()

            # Update progress tracking
            if "progress_tracking" not in task.metadata:
                task.metadata["progress_tracking"] = {}

            task.metadata["progress_tracking"]["progress_percentage"] = 1.0
            task.metadata["progress_tracking"]["status_message"] = "Task completed successfully"
            task.metadata["progress_tracking"]["last_updated"] = datetime.now(UTC).isoformat()

            # Add to progress history if it doesn't exist
            if "progress_history" not in task.metadata["progress_tracking"]:
                task.metadata["progress_tracking"]["progress_history"] = []

            # Add current progress to history
            task.metadata["progress_tracking"]["progress_history"].append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "progress": 1.0,
                    "status_message": "Task completed successfully",
                },
            )

            # Update the task
            self.update_task(task)

            # Update dependent tasks
            self.update_dependent_tasks(task_id)

        elif (
            task.verification_status == VerificationStatus.FAILED and task.execution_attempts >= MAX_EXECUTION_ATTEMPTS
        ):
            # Task has failed after multiple attempts
            task.status = TaskStatus.FAILED
            task.error = f"Failed after {task.execution_attempts} attempts"

            # Add failure information to progress tracking
            if "progress_tracking" not in task.metadata:
                task.metadata["progress_tracking"] = {}

            progress_tracking = task.metadata["progress_tracking"]
            progress_tracking["failure_reason"] = task.error
            progress_tracking["failed_at"] = datetime.now(UTC).isoformat()

            # Update the task
            self.update_task(task)

    def track_blockers_and_dependencies(self) -> None:
        """Track blockers and dependencies for all tasks.

        This method updates the status of all tasks based on their dependencies,
        identifies blocked tasks, and updates the status of tasks that were previously
        blocked but whose dependencies are now resolved.
        """
        tasks = self.get_tasks()

        # First pass: Update status of all tasks based on dependencies
        for task_dict in tasks:
            task_id = uuid.UUID(task_dict["task_id"])
            self.update_task_status_based_on_dependencies(task_id)

        # Second pass: Find tasks that are blocked and add blocker information
        for task_dict in tasks:
            task_id = uuid.UUID(task_dict["task_id"])
            task = self.get_task_by_id(task_id)

            if task and task.status == TaskStatus.BLOCKED:
                # Identify which dependencies are blocking this task
                blocking_dependencies = []

                for dependency in task.dependencies:
                    if dependency.is_blocking:
                        dep_task = self.get_task_by_id(dependency.task_id)
                        if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                            blocking_dependencies.append(
                                {
                                    "task_id": str(dependency.task_id),
                                    "description": dependency.description,
                                    "status": dep_task.status.value if dep_task else "unknown",
                                },
                            )

                # Add blocker information to task metadata
                if "blockers" not in task.metadata:
                    task.metadata["blockers"] = {}

                task.metadata["blockers"]["blocking_dependencies"] = blocking_dependencies
                task.metadata["blockers"]["last_updated"] = datetime.now(UTC).isoformat()

                # Update the task with blocker information
                self.update_task(task)
            elif task and task.status == TaskStatus.PENDING:
                # For pending tasks, clear any previous blocker information
                if "blockers" in task.metadata:
                    task.metadata["blockers"]["blocking_dependencies"] = []
                    task.metadata["blockers"]["last_updated"] = datetime.now(UTC).isoformat()
                    self.update_task(task)

    def detect_and_resolve_blocking_tasks(self) -> dict[str, Any]:
        """Detect and attempt to resolve blocking tasks.

        This method identifies tasks that are blocked, analyzes the blockers,
        and attempts to resolve the blocking situation by suggesting alternative
        execution paths, detecting circular dependencies, and providing detailed
        diagnostics.

        Returns:
            Dictionary with information about blocked tasks and resolution actions:
            - blocked_tasks: List of blocked task IDs
            - circular_dependencies: List of circular dependency chains detected
            - resolution_actions: List of suggested resolution actions
            - unresolvable_tasks: List of tasks that cannot be resolved automatically
            - critical_path_blockers: List of blockers on the critical path

        """
        tasks = self.get_tasks()
        blocked_tasks = []
        circular_dependencies = []
        resolution_actions = []
        unresolvable_tasks = []
        critical_path_blockers = []

        # First, ensure blocker information is up-to-date
        self.track_blockers_and_dependencies()

        # Identify all blocked tasks
        for task_dict in tasks:
            task_id = uuid.UUID(task_dict["task_id"])
            task = self.get_task_by_id(task_id)

            if task and task.status == TaskStatus.BLOCKED:
                blocked_tasks.append(str(task_id))

                # Check for circular dependencies
                circular_result = self._handle_circular_dependencies(task, task_id)
                if circular_result:
                    circular_dependencies.append(circular_result["circular_deps"])
                    resolution_actions.append(circular_result["resolution_action"])
                    continue  # Skip further analysis for tasks with circular dependencies

                # Analyze blockers to determine if they can be resolved
                blockers = task.metadata.get("blockers", {}).get("blocking_dependencies", [])
                if not blockers:
                    # Handle inconsistency (task is blocked but has no blockers)
                    inconsistency_result = self._handle_blocking_inconsistency(task, task_id)
                    resolution_actions.append(inconsistency_result["resolution_action"])
                    continue

                # Analyze and handle different types of blockers
                blocker_analysis = self._analyze_blockers(task, task_id, blockers)

                # Update our tracking collections with the analysis results
                if blocker_analysis["resolution_actions"]:
                    resolution_actions.extend(blocker_analysis["resolution_actions"])

                if blocker_analysis["critical_path_blockers"]:
                    critical_path_blockers.extend(blocker_analysis["critical_path_blockers"])

                # If no resolution was found, mark as unresolvable
                if not blocker_analysis["resolution_actions"]:
                    unresolvable_tasks.append(
                        {
                            "task_id": str(task_id),
                            "description": task.description,
                            "blockers": blockers,
                        },
                    )

        # Return summary of blocked tasks and resolution actions
        return {
            "blocked_tasks": blocked_tasks,
            "circular_dependencies": circular_dependencies,
            "resolution_actions": resolution_actions,
            "unresolvable_tasks": unresolvable_tasks,
            "critical_path_blockers": critical_path_blockers,
        }

    def _handle_circular_dependencies(self, task: Task, task_id: uuid.UUID) -> dict[str, Any] | None:
        """Handle circular dependencies for a task.

        Args:
            task: The task to check for circular dependencies
            task_id: The UUID of the task

        Returns:
            Dictionary with circular dependency information and resolution action, or None if no circular dependencies

        """
        circular_deps = self._detect_circular_dependencies(task_id)
        if not circular_deps:
            return None

        # Add resolution action for circular dependency
        resolution_action = {
            "task_id": str(task_id),
            "action_type": "BREAK_CIRCULAR_DEPENDENCY",
            "description": f"Break circular dependency chain: {' -> '.join(circular_deps)}",
            "suggested_changes": [
                {
                    "dependency_to_modify": circular_deps[0],
                    "action": "MAKE_NON_BLOCKING",
                    "reason": "Part of circular dependency chain",
                },
            ],
        }

        # Update task metadata with circular dependency information
        if "resolution_suggestions" not in task.metadata:
            task.metadata["resolution_suggestions"] = {}

        task.metadata["resolution_suggestions"]["circular_dependency"] = {
            "detected_at": datetime.now(UTC).isoformat(),
            "dependency_chain": circular_deps,
            "suggested_action": "Break circular dependency by making one of the dependencies non-blocking",
        }
        self.update_task(task)

        return {
            "circular_deps": circular_deps,
            "resolution_action": resolution_action,
        }

    def _handle_blocking_inconsistency(self, task: Task, task_id: uuid.UUID) -> dict[str, Any]:
        """Handle inconsistency when a task is marked as blocked but has no blocking dependencies.

        Args:
            task: The task with inconsistency
            task_id: The UUID of the task

        Returns:
            Dictionary with resolution action information

        """
        resolution_action = {
            "task_id": str(task_id),
            "action_type": "FIX_INCONSISTENCY",
            "description": "Task is marked as blocked but no blocking dependencies were found",
            "suggested_changes": [
                {
                    "action": "UPDATE_STATUS",
                    "new_status": "PENDING",
                    "reason": "No actual blockers found",
                },
            ],
        }

        # Update task status to pending
        task.status = TaskStatus.PENDING
        if "resolution_suggestions" not in task.metadata:
            task.metadata["resolution_suggestions"] = {}

        task.metadata["resolution_suggestions"]["status_inconsistency"] = {
            "detected_at": datetime.now(UTC).isoformat(),
            "issue": "Task marked as blocked but no blocking dependencies found",
            "action_taken": "Updated status to PENDING",
        }
        self.update_task(task)

        return {
            "resolution_action": resolution_action,
        }

    def _analyze_blockers(self, task: Task, task_id: uuid.UUID, blockers: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze blockers and suggest resolutions.

        Args:
            task: The blocked task
            task_id: The UUID of the task
            blockers: List of blocking dependencies

        Returns:
            Dictionary with analysis results and resolution actions

        """
        # Categorize the blockers by their status
        categorized_blockers = self._categorize_blockers(blockers)

        resolution_actions = []
        critical_path_blockers = []

        # Determine if this task is on the critical path
        is_on_critical_path = self._is_task_on_critical_path(task_id)

        # Handle different blocker categories and create resolution suggestions
        if categorized_blockers["failed"]:
            failed_result = self._handle_failed_blockers(
                task,
                task_id,
                categorized_blockers["failed"],
                is_on_critical_path=is_on_critical_path,
            )
            resolution_actions.append(failed_result["resolution_action"])
            if is_on_critical_path:
                critical_path_blockers.append(failed_result["critical_path_blocker"])

        if categorized_blockers["stalled"]:
            stalled_result = self._handle_stalled_blockers(
                task,
                task_id,
                categorized_blockers["stalled"],
                is_on_critical_path=is_on_critical_path,
            )
            resolution_actions.append(stalled_result["resolution_action"])
            if is_on_critical_path:
                critical_path_blockers.append(stalled_result["critical_path_blocker"])

        # Handle pending blockers only if there are no other types of blockers
        if (
            categorized_blockers["pending"]
            and not categorized_blockers["failed"]
            and not categorized_blockers["stalled"]
            and not categorized_blockers["in_progress"]
        ):
            pending_result = self._handle_pending_blockers(task, task_id, categorized_blockers["pending"])
            resolution_actions.append(pending_result["resolution_action"])

        return {
            "resolution_actions": resolution_actions,
            "critical_path_blockers": critical_path_blockers,
        }

    def _categorize_blockers(self, blockers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Categorize blockers by their status.

        Args:
            blockers: List of blocking dependencies

        Returns:
            Dictionary with categorized blockers (failed, stalled, in_progress, pending)

        """
        failed_blockers = []
        stalled_blockers = []
        in_progress_blockers = []
        pending_blockers = []

        for blocker in blockers:
            blocker_id = uuid.UUID(blocker["task_id"])
            blocker_task = self.get_task_by_id(blocker_id)

            if not blocker_task:
                # Blocker task doesn't exist - this is an error
                failed_blockers.append(
                    {
                        "task_id": blocker["task_id"],
                        "description": blocker["description"],
                        "issue": "Task not found",
                    },
                )
            elif blocker_task.status == TaskStatus.FAILED:
                failed_blockers.append(
                    {
                        "task_id": blocker["task_id"],
                        "description": blocker["description"],
                        "error": blocker_task.error,
                    },
                )
            elif blocker_task.status == TaskStatus.IN_PROGRESS:
                # Check if the blocker is stalled
                if blocker_task.metadata.get("stalled", {}).get("is_stalled", False):
                    stalled_blockers.append(
                        {
                            "task_id": blocker["task_id"],
                            "description": blocker["description"],
                            "hours_stalled": blocker_task.metadata["stalled"].get("hours_stalled", 0),
                        },
                    )
                else:
                    in_progress_blockers.append(
                        {
                            "task_id": blocker["task_id"],
                            "description": blocker["description"],
                            "progress": blocker_task.metadata.get("progress_tracking", {}).get(
                                "progress_percentage",
                                0,
                            ),
                        },
                    )
            elif blocker_task.status == TaskStatus.PENDING:
                pending_blockers.append(
                    {
                        "task_id": blocker["task_id"],
                        "description": blocker["description"],
                    },
                )

        return {
            "failed": failed_blockers,
            "stalled": stalled_blockers,
            "in_progress": in_progress_blockers,
            "pending": pending_blockers,
        }

    def _handle_failed_blockers(
        self,
        task: Task,
        task_id: uuid.UUID,
        failed_blockers: list[dict[str, Any]],
        *,
        is_on_critical_path: bool = False,
    ) -> dict[str, Any]:
        """Handle failed blockers for a task.

        Args:
            task: The task with failed blockers
            task_id: The UUID of the task
            failed_blockers: List of failed blockers
            is_on_critical_path: Whether the task is on the critical path (default: False)

        Returns:
            Dictionary with resolution action and critical path information

        """
        resolution_action = {
            "task_id": str(task_id),
            "action_type": "HANDLE_FAILED_BLOCKERS",
            "description": f"Task has {len(failed_blockers)} failed blocking dependencies",
            "suggested_changes": [
                {
                    "dependency_id": blocker["task_id"],
                    "action": "MAKE_NON_BLOCKING" if not is_on_critical_path else "RETRY",
                    "reason": f"Blocker failed: {blocker.get('error', 'Unknown error')}",
                }
                for blocker in failed_blockers
            ],
        }

        # Update task metadata with failed blocker information
        if "resolution_suggestions" not in task.metadata:
            task.metadata["resolution_suggestions"] = {}

        task.metadata["resolution_suggestions"]["failed_blockers"] = {
            "detected_at": datetime.now(UTC).isoformat(),
            "blockers": failed_blockers,
            "suggested_action": "Make failed dependencies non-blocking or retry them",
        }
        self.update_task(task)

        critical_path_blocker = {
            "task_id": str(task_id),
            "description": task.description,
            "failed_blockers": failed_blockers,
        }

        return {
            "resolution_action": resolution_action,
            "critical_path_blocker": critical_path_blocker,
        }

    def _handle_stalled_blockers(
        self,
        task: Task,
        task_id: uuid.UUID,
        stalled_blockers: list[dict[str, Any]],
        *,
        is_on_critical_path: bool = False,
    ) -> dict[str, Any]:
        """Handle stalled blockers for a task.

        Args:
            task: The task with stalled blockers
            task_id: The UUID of the task
            stalled_blockers: List of stalled blockers
            is_on_critical_path: Whether the task is on the critical path (default: False)

        Returns:
            Dictionary with resolution action and critical path information

        """
        resolution_action = {
            "task_id": str(task_id),
            "action_type": "HANDLE_STALLED_BLOCKERS",
            "description": f"Task has {len(stalled_blockers)} stalled blocking dependencies",
            "suggested_changes": [
                {
                    "dependency_id": blocker["task_id"],
                    "action": "URGENT_INTERVENTION" if is_on_critical_path else "INTERVENTION_NEEDED",
                    "reason": f"Blocker stalled for {blocker.get('hours_stalled', 0):.1f} hours",
                }
                for blocker in stalled_blockers
            ],
        }

        # Update task metadata with stalled blocker information
        if "resolution_suggestions" not in task.metadata:
            task.metadata["resolution_suggestions"] = {}

        task.metadata["resolution_suggestions"]["stalled_blockers"] = {
            "detected_at": datetime.now(UTC).isoformat(),
            "blockers": stalled_blockers,
            "suggested_action": "Urgent intervention needed"
            if is_on_critical_path
            else "Intervention needed for stalled dependencies",
        }
        self.update_task(task)

        critical_path_blocker = {
            "task_id": str(task_id),
            "description": task.description,
            "stalled_blockers": stalled_blockers,
        }

        return {
            "resolution_action": resolution_action,
            "critical_path_blocker": critical_path_blocker,
        }

    def _handle_pending_blockers(
        self,
        task: Task,
        task_id: uuid.UUID,
        pending_blockers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Handle pending blockers for a task.

        Args:
            task: The task with pending blockers
            task_id: The UUID of the task
            pending_blockers: List of pending blockers

        Returns:
            Dictionary with resolution action information

        """
        resolution_action = {
            "task_id": str(task_id),
            "action_type": "PRIORITIZE_BLOCKERS",
            "description": f"Task has {len(pending_blockers)} pending blocking dependencies",
            "suggested_changes": [
                {
                    "dependency_id": blocker["task_id"],
                    "action": "INCREASE_PRIORITY",
                    "reason": "Blocking a dependent task",
                }
                for blocker in pending_blockers
            ],
        }

        # Update task metadata with pending blocker information
        if "resolution_suggestions" not in task.metadata:
            task.metadata["resolution_suggestions"] = {}

        task.metadata["resolution_suggestions"]["pending_blockers"] = {
            "detected_at": datetime.now(UTC).isoformat(),
            "blockers": pending_blockers,
            "suggested_action": "Increase priority of pending dependencies",
        }
        self.update_task(task)

        return {
            "resolution_action": resolution_action,
        }

    def _detect_circular_dependencies(
        self,
        task_id: uuid.UUID,
        visited: set[str] | None = None,
        path: list[str] | None = None,
    ) -> list[str] | None:
        """Detect circular dependencies starting from the given task.

        Args:
            task_id: Starting task ID
            visited: Set of visited task IDs (for recursion)
            path: Current dependency path (for recursion)

        Returns:
            List of task IDs forming a circular dependency chain, or None if no circular dependency is found

        """
        if visited is None:
            visited = set()
        if path is None:
            path = []

        task = self.get_task_by_id(task_id)
        if not task:
            return None

        task_id_str = str(task_id)

        # If we've seen this task before in the current path, we have a circular dependency
        if task_id_str in path:
            # Return the circular path (from the first occurrence of this task to the end)
            return path[path.index(task_id_str) :] + [task_id_str]

        # If we've visited this task before but it's not in the current path, no need to check again
        if task_id_str in visited:
            return None

        # Add this task to visited and path
        visited.add(task_id_str)
        path.append(task_id_str)

        # Check all dependencies
        for dependency in task.dependencies:
            if dependency.is_blocking:
                circular_path = self._detect_circular_dependencies(dependency.task_id, visited, path.copy())
                if circular_path:
                    return circular_path

        return None

    def _is_task_on_critical_path(self, task_id: uuid.UUID) -> bool:
        """Determine if a task is on the critical path.

        A task is on the critical path if it has dependents that are blocked by it,
        and those dependents are either:
        1. Root tasks (tasks with no parent)
        2. Tasks with high priority
        3. Tasks that themselves have many dependents

        Args:
            task_id: Task ID to check

        Returns:
            True if the task is on the critical path, False otherwise

        """
        # Define a constant for the threshold of many dependents
        many_dependents_threshold = 3

        tasks = self.get_tasks()
        task_id_str = str(task_id)

        # Find all tasks that depend on this task
        dependent_tasks = [
            task_dict
            for task_dict in tasks
            for dep in task_dict.get("dependencies", [])
            if dep.get("task_id") == task_id_str and dep.get("is_blocking", True)
        ]

        if not dependent_tasks:
            return False  # No dependents, not on critical path

        # Check if any dependent is a root task (no parent)
        for dependent in dependent_tasks:
            if not dependent.get("parent_task_id"):
                return True  # Dependent is a root task, so this task is on critical path

        # Check if any dependent has high priority
        for dependent in dependent_tasks:
            priority = dependent.get("priority")
            if priority in [TaskPriority.HIGH.value, TaskPriority.CRITICAL.value]:
                return True  # Dependent has high priority, so this task is on critical path

        # Check if any dependent has many dependents itself (recursive check with depth limit)
        for dependent in dependent_tasks:
            dependent_id = uuid.UUID(dependent["task_id"])
            if self._count_recursive_dependents(dependent_id, depth=0, max_depth=2) >= many_dependents_threshold:
                return True  # Dependent has many dependents, so this task is on critical path

        return False

    def _count_recursive_dependents(self, task_id: uuid.UUID, depth: int = 0, max_depth: int = 2) -> int:
        """Count the number of tasks that depend on this task, recursively.

        Args:
            task_id: Task ID to check
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            Number of dependent tasks

        """
        if depth >= max_depth:
            return 0

        tasks = self.get_tasks()
        task_id_str = str(task_id)

        # Find all tasks that depend on this task
        dependent_tasks = [
            task_dict
            for task_dict in tasks
            for dep in task_dict.get("dependencies", [])
            if dep.get("task_id") == task_id_str and dep.get("is_blocking", True)
        ]

        # Count direct dependents
        count = len(dependent_tasks)

        # Add recursive dependents
        for dependent in dependent_tasks:
            dependent_id = uuid.UUID(dependent["task_id"])
            count += self._count_recursive_dependents(dependent_id, depth + 1, max_depth)

        return count

    def recalculate_all_task_progress(self) -> None:
        """Recalculate progress for all tasks in the hierarchy.

        This method performs a bottom-up recalculation of progress for all tasks,
        starting from leaf tasks (those without subtasks) and propagating up to
        root tasks. This ensures that all progress reporting is up-to-date and
        accurately reflects the current state of task execution.

        The method also detects stalled tasks (those that haven't made progress
        in a configurable time period) and updates their metadata accordingly.
        """
        tasks = self.get_tasks()

        # Step 1: Identify leaf tasks and parent task hierarchy
        leaf_tasks, parent_tasks_by_level = self._identify_task_hierarchy(tasks)

        # Step 2: Process leaf tasks (check for stalled tasks)
        self._process_leaf_tasks(leaf_tasks)

        # Step 3: Process parent tasks level by level, from lowest to highest
        self._process_parent_tasks_by_level(parent_tasks_by_level)

        # Step 4: Update the overall progress information
        overall_progress = self.get_overall_progress()
        self.set_context("overall_progress", overall_progress)

    def _identify_task_hierarchy(self, tasks: list[dict[str, Any]]) -> tuple[list[Task], dict[int, list[uuid.UUID]]]:
        """Identify leaf tasks and organize parent tasks by level.

        Args:
            tasks: List of task dictionaries

        Returns:
            Tuple containing:
            - List of leaf tasks (tasks without subtasks)
            - Dictionary mapping level to list of parent task IDs at that level

        """
        # First, identify all leaf tasks (tasks without subtasks)
        leaf_tasks = []
        for task_dict in tasks:
            task = self._convert_dict_to_task(task_dict)
            if not task.subtasks:
                leaf_tasks.append(task)

        # Next, identify all parent tasks and organize them by level
        # (where level 0 are parents of leaf tasks, level 1 are parents of level 0, etc.)
        parent_tasks_by_level = {}
        processed_task_ids = set()

        # Start with leaf tasks' parents
        current_level = 0
        current_parents = set()
        for leaf_task in leaf_tasks:
            if leaf_task.parent_task_id and leaf_task.parent_task_id not in processed_task_ids:
                current_parents.add(leaf_task.parent_task_id)
                processed_task_ids.add(leaf_task.parent_task_id)

        parent_tasks_by_level[current_level] = list(current_parents)

        # Continue up the hierarchy until we reach root tasks
        while current_parents:
            next_level = current_level + 1
            next_parents = set()

            for parent_id in current_parents:
                parent_task = self.get_task_by_id(parent_id)
                if parent_task and parent_task.parent_task_id and parent_task.parent_task_id not in processed_task_ids:
                    next_parents.add(parent_task.parent_task_id)
                    processed_task_ids.add(parent_task.parent_task_id)

            if next_parents:
                parent_tasks_by_level[next_level] = list(next_parents)
                current_parents = next_parents
                current_level = next_level
            else:
                break

        return leaf_tasks, parent_tasks_by_level

    def _process_leaf_tasks(self, leaf_tasks: list[Task]) -> None:
        """Process leaf tasks (check for stalled tasks).

        Args:
            leaf_tasks: List of leaf tasks

        """
        for leaf_task in leaf_tasks:
            self._check_for_stalled_task(leaf_task.task_id)

    def _process_parent_tasks_by_level(self, parent_tasks_by_level: dict[int, list[uuid.UUID]]) -> None:
        """Process parent tasks level by level, from lowest to highest.

        Args:
            parent_tasks_by_level: Dictionary mapping level to list of parent task IDs at that level

        """
        for level in sorted(parent_tasks_by_level.keys()):
            for parent_id in parent_tasks_by_level[level]:
                # Recalculate rollup progress for this parent
                rollup_progress = self.calculate_rollup_progress(parent_id)

                # Update the parent's progress
                self.track_delegated_task_progress(
                    parent_id,
                    rollup_progress["progress"],
                    rollup_progress["status_message"],
                )

                # Check if this parent task is stalled
                self._check_for_stalled_task(parent_id)

    def _check_for_stalled_task(self, task_id: uuid.UUID, stall_threshold_hours: float = 24.0) -> None:
        """Check if a task has stalled (not made progress in a while).

        Args:
            task_id: Task ID to check.
            stall_threshold_hours: Number of hours after which a task is considered stalled.

        """
        task = self.get_task_by_id(task_id)
        if not task:
            return

        # Skip completed or failed tasks
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return

        # Check when the task was last updated
        progress_tracking = task.metadata.get("progress_tracking", {})
        last_updated_str = progress_tracking.get("last_updated")

        if not last_updated_str:
            return

        try:
            last_updated = datetime.fromisoformat(last_updated_str)
            now = datetime.now(UTC)
            hours_since_update = (now - last_updated).total_seconds() / 3600

            # If the task hasn't been updated in the threshold period, mark it as stalled
            if hours_since_update > stall_threshold_hours:
                if "stalled" not in task.metadata:
                    task.metadata["stalled"] = {}

                task.metadata["stalled"]["is_stalled"] = True
                task.metadata["stalled"]["stalled_since"] = last_updated_str
                task.metadata["stalled"]["hours_stalled"] = hours_since_update
                task.metadata["stalled"]["detected_at"] = now.isoformat()

                # Add a note to the progress tracking
                if "status_message" in progress_tracking:
                    progress_tracking["status_message"] += f" (Stalled for {hours_since_update:.1f} hours)"
                else:
                    progress_tracking["status_message"] = f"Stalled for {hours_since_update:.1f} hours"

                # Update the task
                self.update_task(task)
            elif task.metadata.get("stalled", {}).get("is_stalled", False):
                # If the task was previously stalled but has been updated, clear the stalled flag
                task.metadata["stalled"]["is_stalled"] = False
                task.metadata["stalled"]["resolved_at"] = now.isoformat()

                # Update the task
                self.update_task(task)
        except (ValueError, TypeError):
            # If there's an error parsing the timestamp, just skip this check
            pass

    def _convert_dict_to_task(self, task_dict: dict) -> Task:
        """Convert task dictionary to Task object.

        Args:
            task_dict: Task dictionary.

        Returns:
            Task object.

        """
        # Convert string IDs back to UUID objects
        task_id = uuid.UUID(task_dict["task_id"])
        parent_task_id = uuid.UUID(task_dict["parent_task_id"]) if task_dict.get("parent_task_id") else None
        subtasks = [uuid.UUID(subtask_id) for subtask_id in task_dict.get("subtasks", [])]

        # Convert dependency dictionaries to TaskDependency objects using list comprehension
        dependencies = [
            TaskDependency(
                task_id=uuid.UUID(dep_dict["task_id"]),
                description=dep_dict["description"],
                is_blocking=dep_dict.get("is_blocking", True),
            )
            for dep_dict in task_dict.get("dependencies", [])
        ]

        # Handle execution stage and verification status
        execution_stage = None
        if task_dict.get("execution_stage"):
            if isinstance(task_dict["execution_stage"], str):
                execution_stage = ExecutionStage(task_dict["execution_stage"])
            else:
                execution_stage = task_dict["execution_stage"]

        verification_status = None
        if task_dict.get("verification_status"):
            if isinstance(task_dict["verification_status"], str):
                verification_status = VerificationStatus(task_dict["verification_status"])
            else:
                verification_status = task_dict["verification_status"]

        # Create Task object
        return Task(
            description=task_dict["description"],
            task_id=task_id,
            priority=task_dict.get("priority", "MEDIUM"),
            status=task_dict.get("status", "PENDING"),
            complexity=task_dict.get("complexity", "MODERATE"),
            dependencies=dependencies,
            parent_task_id=parent_task_id,
            subtasks=subtasks,
            assigned_role=task_dict.get("assigned_role"),
            assigned_agent_id=task_dict.get("assigned_agent_id"),
            metadata=task_dict.get("metadata", {}),
            result=task_dict.get("result"),
            error=task_dict.get("error"),
            created_at=task_dict.get("created_at"),
            updated_at=task_dict.get("updated_at"),
            completed_at=task_dict.get("completed_at"),
            execution_stage=execution_stage,
            verification_status=verification_status,
            execution_attempts=task_dict.get("execution_attempts", 0),
            execution_logs=task_dict.get("execution_logs", []),
            verification_details=task_dict.get("verification_details", {}),
            execution_metadata=task_dict.get("execution_metadata", {}),
        )

    def detect_and_resolve_deadlocks(self) -> list[dict[str, Any]]:
        """Detect and resolve circular dependencies (deadlocks) in the task graph.

        This method scans all tasks for circular dependencies and automatically resolves them
        by making the least critical dependency in the cycle non-blocking.

        Returns:
            List of resolved deadlocks with information about the resolved dependencies

        """
        tasks = self.get_tasks()
        resolved_deadlocks = []

        # Check each task for circular dependencies
        for task_dict in tasks:
            task_id = uuid.UUID(task_dict["task_id"])

            # Detect circular dependency starting from this task
            circular_path = self._detect_circular_dependencies(task_id)

            if circular_path:
                # We found a circular dependency - resolve it
                resolution_info = self._resolve_circular_dependency(circular_path)
                if resolution_info:
                    resolved_deadlocks.append(resolution_info)

                    # After resolving one cycle, check if there are more from this task
                    # This prevents cascading deadlocks
                    circular_path = self._detect_circular_dependencies(task_id)
                    while circular_path:
                        resolution_info = self._resolve_circular_dependency(circular_path)
                        if resolution_info:
                            resolved_deadlocks.append(resolution_info)
                        circular_path = self._detect_circular_dependencies(task_id)

        return resolved_deadlocks

    def _resolve_circular_dependency(self, circular_path: list[str]) -> dict[str, Any] | None:
        """Resolve a circular dependency by breaking the least critical link.

        Args:
            circular_path: List of task IDs forming a circular dependency

        Returns:
            Dictionary with resolved dependency information or None if not resolved

        """
        if not circular_path or len(circular_path) < MIN_CIRCULAR_PATH_LENGTH:
            return None

        # Find the dependency with the lowest priority or complexity to break
        best_task_to_modify, best_dependency_to_break = self._find_best_dependency_to_break(circular_path)

        # If we found a dependency to break, make it non-blocking
        if best_task_to_modify and best_dependency_to_break:
            # Update the task dependencies
            self._make_dependency_non_blocking(best_task_to_modify, best_dependency_to_break, circular_path)

            # Update status of tasks that may now be unblocked
            self.update_task_status_based_on_dependencies(best_task_to_modify.task_id)
            self.update_dependent_tasks(best_task_to_modify.task_id)

            # Return information about the resolution
            return {
                "task_id": str(best_task_to_modify.task_id),
                "description": best_task_to_modify.description,
                "circular_path": circular_path,
                "resolved_dependency": str(best_dependency_to_break.task_id),
                "dependency_description": best_dependency_to_break.description,
                "action": "Made dependency non-blocking",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        return None

    def _find_best_dependency_to_break(self, circular_path: list[str]) -> tuple[Task | None, TaskDependency | None]:
        """Find the best dependency to break in a circular path.

        Args:
            circular_path: List of task IDs forming a circular dependency

        Returns:
            Tuple of (task to modify, dependency to break)

        """
        best_task_to_modify = None
        best_dependency_to_break = None
        lowest_score = float("inf")  # Lower score means better candidate to break

        # For each task in the circular path, find its dependency on the next task in the path
        for i in range(len(circular_path) - 1):
            current_task_id = uuid.UUID(circular_path[i])
            next_task_id = uuid.UUID(circular_path[i + 1])

            current_task = self.get_task_by_id(current_task_id)
            if not current_task:
                continue

            # Find the dependency on the next task
            for dependency in current_task.dependencies:
                if dependency.task_id == next_task_id and dependency.is_blocking:
                    # Skip if this is already non-blocking
                    if not dependency.is_blocking:
                        continue

                    # Calculate the "score" of this dependency based on task priority and complexity
                    # Lower score means better candidate to break
                    priority_score = self._get_priority_score(current_task.priority)
                    complexity_score = self._get_complexity_score(current_task.complexity)

                    # Check if the next task is on the critical path
                    critical_path_score = 100 if self._is_task_on_critical_path(next_task_id) else 0

                    # Calculate final score - lower means better candidate to break
                    score = priority_score + complexity_score + critical_path_score

                    # If this is better than our current best, update it
                    if score < lowest_score:
                        lowest_score = score
                        best_task_to_modify = current_task
                        best_dependency_to_break = dependency

        return best_task_to_modify, best_dependency_to_break

    def _make_dependency_non_blocking(
        self,
        task: Task,
        dependency_to_break: TaskDependency,
        circular_path: list[str],
    ) -> None:
        """Make a dependency non-blocking to resolve a circular dependency.

        Args:
            task: Task to modify
            dependency_to_break: Dependency to make non-blocking
            circular_path: List of task IDs forming a circular dependency

        """
        # Make a copy of the dependency with is_blocking=False
        new_dependencies = []
        for dep in task.dependencies:
            if dep.task_id == dependency_to_break.task_id:
                new_dependencies.append(
                    TaskDependency(
                        task_id=dep.task_id,
                        description=dep.description,
                        is_blocking=False,  # Make non-blocking
                    ),
                )
            else:
                new_dependencies.append(dep)

        # Update the task's dependencies
        task.dependencies = new_dependencies

        # Add metadata about the deadlock resolution
        if "deadlock_resolutions" not in task.metadata:
            task.metadata["deadlock_resolutions"] = []

        task.metadata["deadlock_resolutions"].append(
            {
                "resolved_at": datetime.now(UTC).isoformat(),
                "circular_path": circular_path,
                "resolved_dependency": str(dependency_to_break.task_id),
                "reason": (
                    "Automatic deadlock prevention: Made dependency non-blocking to resolve circular dependency"
                ),
            },
        )

        # Update the task
        self.update_task(task)

    def _get_priority_score(self, priority: TaskPriority | str) -> int:
        """Convert task priority to a numeric score.

        Args:
            priority: Task priority enum or string

        Returns:
            Numeric score (lower is less critical)

        """
        if isinstance(priority, str):
            priority = TaskPriority(priority)

        priority_scores = {
            TaskPriority.LOW: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.HIGH: 4,
            TaskPriority.CRITICAL: 8,
        }

        return priority_scores.get(priority, 2)  # Default to MEDIUM if unknown

    def _get_complexity_score(self, complexity: TaskComplexity | str) -> int:
        """Convert task complexity to a numeric score.

        Args:
            complexity: Task complexity enum or string

        Returns:
            Numeric score (lower is less complex)

        """
        if isinstance(complexity, str):
            complexity = TaskComplexity(complexity)

        complexity_scores = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2,
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.VERY_COMPLEX: 4,
        }

        return complexity_scores.get(complexity, 2)  # Default to MODERATE if unknown

    def is_task_blocked_by_dependencies(self, task_id: uuid.UUID) -> bool:
        """Check if a task is blocked by dependencies.

        Args:
            task_id: Task ID.

        Returns:
            True if task is blocked by dependencies.

        """
        task = self.get_task_by_id(task_id)
        if not task:
            return False

        # Check if any blocking dependency is not completed
        for dependency in task.dependencies:
            if dependency.is_blocking:
                dep_task = self.get_task_by_id(dependency.task_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    return True

        return False

    def _prepare_task_for_storage(self, task: Task) -> dict[str, Any]:
        """Prepare a task for storage by converting it to a dictionary.

        Args:
            task: Task to convert

        Returns:
            Dictionary representation of the task

        """
        # Convert Task object to dict for storage
        updated_task = asdict(task)

        # Convert UUID objects to strings for JSON serialization
        updated_task["task_id"] = str(task.task_id)
        if task.parent_task_id:
            updated_task["parent_task_id"] = str(task.parent_task_id)
        updated_task["subtasks"] = [str(subtask_id) for subtask_id in task.subtasks]
        updated_task["dependencies"] = [
            {
                "task_id": str(dep.task_id),
                "description": dep.description,
                "is_blocking": dep.is_blocking,
            }
            for dep in task.dependencies
        ]

        # Handle execution stage and verification status
        if task.execution_stage:
            updated_task["execution_stage"] = task.execution_stage.value
        if task.verification_status:
            updated_task["verification_status"] = task.verification_status.value

        return updated_task


class InMemoryStateManager:
    """State manager implementation that stores agent state in memory."""

    def __init__(self, state: AgentState | None = None) -> None:
        """Initialize the in-memory state manager.

        Args:
            state: Optional initial state.

        """
        self._state = state or AgentState()
        self._saved_states = {}  # Store saved states for listing

    def __getattr__(self, name: str) -> object:
        """Forward attribute access to the underlying state object.

        Args:
            name: Attribute name.

        Returns:
            Attribute value from the state object.

        Raises:
            AttributeError: If attribute not found.

        """
        # Check if this is a state manager method that should be handled separately
        if name in dir(self.__class__):
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

        # Forward to the underlying state
        try:
            return getattr(self._state, name)
        except AttributeError:
            msg = f"Neither '{self.__class__.__name__}' nor 'AgentState' has attribute '{name}'"
            raise AttributeError(msg) from None

    def get_state(self) -> AgentState:
        """Get current state.

        Returns:
            Current state.

        """
        return self._state

    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: State to set.

        """
        self._state = state

    def clear_state(self) -> None:
        """Clear current state."""
        self._state = AgentState()

    def save_state(self, path: str | None = None) -> str:
        """Save the current state to memory and disk.

        Args:
            path: Optional path identifier or file path. If None, agent_id is used.

        Returns:
            The identifier or file path the state was saved under.

        """
        # Use agent_id as the default key if path is not specified
        key = path or self._state.agent_id

        # Save the state in memory (deep copy to prevent modification)
        self._saved_states[key] = deepcopy(self._state)

        # Determine file path for disk storage
        file_path = path
        if not file_path or not ("/" in file_path or "\\" in file_path):
            # No path provided or it doesn't look like a file path
            # Create default path as agent_id.json in current directory
            file_path = f"{self._state.agent_id}.json"

        # Ensure directory exists
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save to disk
        state_dict = self._state.to_dict()
        path_obj.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

        return file_path

    def load_state(self, path: str) -> AgentState:
        """Load state from memory or disk.

        Args:
            path: Path identifier or file path.

        Returns:
            Loaded state.

        Raises:
            ConfigError: If state loading fails.

        """
        # Check if path exists as a file, if it does, load from disk
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_file():
            try:
                state_dict = json.loads(path_obj.read_text(encoding="utf-8"))
                loaded_state = AgentState.from_dict(state_dict)
                self._state = loaded_state
            except Exception as e:
                msg = f"Failed to load state from file: {e}"
                raise ConfigError(msg) from e
            else:
                return self._state

        # Otherwise try to load from in-memory storage
        if path not in self._saved_states:
            msg = f"State not found: {path}"
            raise ConfigError(msg)

        # Set as current state (deep copy to prevent modification)
        self._state = deepcopy(self._saved_states[path])
        return self._state

    def list_states(self) -> list[str]:
        """List all saved states.

        Returns:
            List of state identifiers.

        """
        return list(self._saved_states.keys())

    def delete_state(self, state_id: str) -> None:
        """Delete a saved state.

        Args:
            state_id: The ID of the state to delete.

        Raises:
            FileNotFoundError: If the state file doesn't exist.

        """
        if state_id in self._saved_states:
            del self._saved_states[state_id]

        # If there's a file on disk, delete it
        file_path = Path(f"{state_id}.json")
        if file_path.exists():
            file_path.unlink()

    def get_state_by_id(self, agent_id: str) -> AgentState:
        """Get a saved state by agent ID.

        Args:
            agent_id: The agent ID to look up.

        Returns:
            The saved state.

        Raises:
            ConfigError: If state not found.

        """
        if agent_id not in self._saved_states:
            msg = f"State not found for agent ID: {agent_id}"
            raise ConfigError(msg)

        return deepcopy(self._saved_states[agent_id])


class FileStateManager:
    """File-based state manager.

    This class provides file-based state management for agents.
    It persists agent state to disk in JSON format.
    """

    def __init__(self, base_dir: str, state: AgentState | None = None) -> None:
        """Initialize FileStateManager.

        Args:
            base_dir: Base directory for state files.
            state: Initial state.

        """
        self.base_dir = base_dir
        self._state = state or AgentState()
        self._saved_path = None
        self._saved_states = {}

        # Create base directory if it doesn't exist
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> object:
        """Forward attribute access to the underlying state object.

        Args:
            name: Attribute name.

        Returns:
            Attribute value.

        Raises:
            AttributeError: If attribute not found.

        """
        try:
            return getattr(self._state, name)
        except AttributeError:
            # Try to get the attribute from AgentState
            try:
                return getattr(AgentState, name)
            except AttributeError:
                msg = f"Neither '{self.__class__.__name__}' nor 'AgentState' has attribute '{name}'"
                raise AttributeError(msg) from None

    def get_state(self) -> AgentState:
        """Get current state.

        Returns:
            Current state.

        """
        return self._state

    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: State to set.

        """
        self._state = state

    def clear_state(self) -> None:
        """Clear current state."""
        self._state = AgentState()

    def update_task(self, task: Task) -> None:
        """Update task.

        Args:
            task: Task to update.

        """
        self._state.update_task(task)

    def add_task(self, task: Task) -> None:
        """Add task.

        Args:
            task: Task to add.

        """
        self._state.add_task(task)

    def get_tasks(self) -> list[Task]:
        """Get tasks.

        Returns:
            List of tasks.

        """
        return self._state.get_tasks()

    def get_task_by_id(self, task_id: uuid.UUID) -> Task:
        """Get task by ID.

        Args:
            task_id: Task ID.

        Returns:
            Task.

        Raises:
            ValueError: If task not found.

        """
        task = self._state.get_task_by_id(task_id)
        if not task:
            msg = f"Task not found: {task_id}"
            raise ValueError(msg)
        return task

    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register agent.

        Args:
            agent_id: Agent ID.
            agent: Agent instance.

        """
        self._state.register_agent(agent_id, agent)

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent instance or None if not found.

        """
        try:
            agents = self._state.get_registered_agents()
            return agents.get(agent_id)
        except (KeyError, AttributeError):
            return None

    def get_agent_for_step(self, step: AgentStep) -> Agent:
        """Get agent for step.

        Args:
            step: Step to get agent for.

        Returns:
            Agent for step.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        return self._state.get_agent_for_step(step)

    def add_message(self, message: Message) -> None:
        """Add message.

        Args:
            message: Message to add.

        """
        self._state.add_message(message)

    def get_messages(self) -> list[Message]:
        """Get messages.

        Returns:
            List of messages.

        """
        return self._state.messages

    def save_state(self, path: str | None = None) -> str:
        """Save state to file.

        Args:
            path: Path to save state to. If None, a default path is used.

        Returns:
            Path state was saved to.

        """
        if path is None:
            path = str(Path(self.base_dir) / f"{self._state.agent_id}.json")

        state_dict = self._state.to_dict()
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)

        return path

    def load_state(self, path: str) -> AgentState:
        """Load state from file.

        Args:
            path: Path to load state from.

        Returns:
            Loaded state.

        Raises:
            ConfigError: If state file not found.

        """
        try:
            with Path(path).open(encoding="utf-8") as f:
                state_dict = json.load(f)
        except FileNotFoundError:
            msg = f"State file not found: {path}"
            raise ConfigError(msg) from None

        self._state = AgentState.from_dict(state_dict)
        return self._state

    def list_states(self) -> list[str]:
        """List available state files.

        Returns:
            List of agent IDs.

        """
        result = []
        for filename in os.listdir(self.base_dir):
            if filename.endswith(".json"):
                # Extract agent ID from filename (remove .json extension)
                agent_id = filename[:-5]
                result.append(agent_id)
        return result

    def get_state_by_id(self, agent_id: str) -> AgentState:
        """Get state by agent ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent state.

        Raises:
            ConfigError: If state file not found.

        """
        path = Path(self.base_dir) / f"{agent_id}.json"
        if not path.exists():
            msg = f"State file not found for agent: {agent_id}"
            raise ConfigError(msg)

        return self.load_state(str(path))

    def delete_state(self, agent_id: str) -> None:
        """Delete state file.

        Args:
            agent_id: Agent ID.

        Raises:
            ConfigError: If state file not found.

        """
        path = Path(self.base_dir) / f"{agent_id}.json"
        if not path.exists():
            msg = f"State file not found for agent: {agent_id}"
            raise ConfigError(msg)

        path.unlink()

    def track_delegated_task_progress(
        self,
        task_id: uuid.UUID,
        progress: float,
        status_message: str | None = None,
    ) -> None:
        """Track progress of a delegated task.

        Args:
            task_id: Task ID.
            progress: Progress percentage (0.0 to 1.0).
            status_message: Optional status message.

        """
        self._state.track_delegated_task_progress(task_id, progress, status_message)

    def update_parent_task_progress(self, parent_task_id: uuid.UUID) -> None:
        """Update progress of a parent task based on its subtasks.

        Args:
            parent_task_id: Parent task ID.

        """
        self._state.update_parent_task_progress(parent_task_id)

    def calculate_rollup_progress(self, task_id: uuid.UUID) -> dict[str, Any]:
        """Calculate rollup progress for a task based on its subtasks.

        This method provides an enhanced progress calculation that takes into account:
        - Task priorities (higher priority tasks have more weight)
        - Task complexity (more complex tasks have more weight)
        - Task status (completed, in progress, blocked, etc.)
        - Dependency relationships between tasks

        Args:
            task_id: Task ID.

        Returns:
            Dictionary with rollup progress information including:
            - progress: Overall progress value (0.0 to 1.0)
            - status_message: Status message describing the progress
            - weighted_progress: Progress weighted by priority and complexity
            - critical_path_progress: Progress of tasks on the critical path
            - blocking_tasks: List of tasks blocking progress

        """
        task = self.get_task_by_id(task_id)
        if not task or not task.subtasks:
            return {
                "progress": 0.0,
                "status_message": "No subtasks found",
                "weighted_progress": 0.0,
                "critical_path_progress": 0.0,
                "blocking_tasks": [],
            }

        # Process subtasks and collect progress data
        progress_data = self._collect_subtask_progress_data(task)

        # Calculate final progress metrics
        return self._calculate_final_progress_metrics(progress_data)

    def _collect_subtask_progress_data(self, task: Task) -> dict[str, Any]:
        """Collect progress data from all subtasks.

        Args:
            task: Parent task containing subtasks

        Returns:
            Dictionary with collected progress data

        """
        # Initialize counters and lists
        total_subtasks = len(task.subtasks)
        completed_subtasks = 0
        in_progress_subtasks = 0
        blocked_subtasks = 0
        failed_subtasks = 0

        # Priority and complexity weights
        priority_weights = {
            TaskPriority.LOW.value: 0.5,
            TaskPriority.MEDIUM.value: 1.0,
            TaskPriority.HIGH.value: 1.5,
            TaskPriority.CRITICAL.value: 2.0,
        }

        complexity_weights = {
            TaskComplexity.SIMPLE.value: 0.75,
            TaskComplexity.MODERATE.value: 1.0,
            TaskComplexity.COMPLEX.value: 1.5,
            TaskComplexity.VERY_COMPLEX.value: 2.0,
        }

        # Track weighted progress
        total_weight = 0.0
        weighted_progress = 0.0

        # Track critical path and blocking tasks
        critical_path_tasks = []
        blocking_tasks = []

        # Process each subtask
        for subtask_id in task.subtasks:
            subtask = self.get_task_by_id(subtask_id)
            if not subtask:
                continue

            # Get priority and complexity weights
            priority = subtask.priority.value if hasattr(subtask.priority, "value") else subtask.priority
            complexity = subtask.complexity.value if hasattr(subtask.complexity, "value") else subtask.complexity

            priority_weight = priority_weights.get(priority, 1.0)
            complexity_weight = complexity_weights.get(complexity, 1.0)

            # Calculate combined weight
            combined_weight = priority_weight * complexity_weight
            total_weight += combined_weight

            # Track task status
            if subtask.status == TaskStatus.COMPLETED:
                completed_subtasks += 1
                weighted_progress += combined_weight
            elif subtask.status == TaskStatus.IN_PROGRESS:
                in_progress_subtasks += 1
                # For in-progress tasks, use their reported progress
                if "progress_tracking" in subtask.metadata:
                    progress = subtask.metadata["progress_tracking"].get("progress_percentage", 0.0)
                    weighted_progress += combined_weight * progress
            elif subtask.status == TaskStatus.BLOCKED:
                blocked_subtasks += 1
                blocking_tasks.append(
                    {
                        "task_id": str(subtask_id),
                        "description": subtask.description,
                        "blockers": subtask.metadata.get("blockers", {}).get("blocking_dependencies", []),
                    },
                )
            elif subtask.status == TaskStatus.FAILED:
                failed_subtasks += 1

            # Check if task is on critical path (has dependencies or dependents)
            if subtask.dependencies or any(self.is_dependent_on(other_id, subtask_id) for other_id in task.subtasks):
                critical_path_tasks.append(subtask)

        return {
            "total_subtasks": total_subtasks,
            "completed_subtasks": completed_subtasks,
            "in_progress_subtasks": in_progress_subtasks,
            "blocked_subtasks": blocked_subtasks,
            "failed_subtasks": failed_subtasks,
            "total_weight": total_weight,
            "weighted_progress": weighted_progress,
            "critical_path_tasks": critical_path_tasks,
            "blocking_tasks": blocking_tasks,
        }

    def _calculate_final_progress_metrics(self, progress_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate final progress metrics based on collected data.

        Args:
            progress_data: Dictionary with collected progress data

        Returns:
            Dictionary with final progress metrics

        """
        # Extract data from progress_data
        total_subtasks = progress_data["total_subtasks"]
        completed_subtasks = progress_data["completed_subtasks"]
        in_progress_subtasks = progress_data["in_progress_subtasks"]
        blocked_subtasks = progress_data["blocked_subtasks"]
        failed_subtasks = progress_data["failed_subtasks"]
        total_weight = progress_data["total_weight"]
        weighted_progress = progress_data["weighted_progress"]
        critical_path_tasks = progress_data["critical_path_tasks"]
        blocking_tasks = progress_data["blocking_tasks"]

        # Calculate normalized weighted progress
        normalized_weighted_progress = weighted_progress / total_weight if total_weight > 0 else 0.0

        # Calculate critical path progress
        critical_path_progress = 0.0
        if critical_path_tasks:
            critical_path_completed = sum(1 for t in critical_path_tasks if t.status == TaskStatus.COMPLETED)
            critical_path_progress = critical_path_completed / len(critical_path_tasks)

        # Calculate overall progress using a weighted combination of metrics
        # - 60% based on weighted task progress
        # - 30% based on critical path progress
        # - 10% based on simple task count progress
        simple_progress = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
        overall_progress = 0.6 * normalized_weighted_progress + 0.3 * critical_path_progress + 0.1 * simple_progress

        # Generate status message
        status_message = (
            f"Progress: {completed_subtasks}/{total_subtasks} tasks completed"
            f" ({in_progress_subtasks} in progress, {blocked_subtasks} blocked, {failed_subtasks} failed)"
        )

        return {
            "progress": overall_progress,
            "status_message": status_message,
            "weighted_progress": normalized_weighted_progress,
            "critical_path_progress": critical_path_progress,
            "simple_progress": simple_progress,
            "completed_subtasks": completed_subtasks,
            "in_progress_subtasks": in_progress_subtasks,
            "blocked_subtasks": blocked_subtasks,
            "failed_subtasks": failed_subtasks,
            "total_subtasks": total_subtasks,
            "blocking_tasks": blocking_tasks,
        }

    def is_task_blocked_by_dependencies(self, task_id: uuid.UUID) -> bool:
        """Check if a task is blocked by dependencies.

        Args:
            task_id: Task ID.

        Returns:
            True if task is blocked by dependencies.

        """
        return self._state.is_task_blocked_by_dependencies(task_id)

    def update_task_status_based_on_dependencies(self, task_id: uuid.UUID) -> None:
        """Update task status based on dependencies.

        Args:
            task_id: Task ID.

        """
        self._state.update_task_status_based_on_dependencies(task_id)

    def update_dependent_tasks(self, task_id: uuid.UUID) -> None:
        """Update status of tasks that depend on the given task.

        Args:
            task_id: Task ID.

        """
        self._state.update_dependent_tasks(task_id)

    def recalculate_all_task_progress(self) -> None:
        """Recalculate progress for all tasks in the hierarchy."""
        self._state.recalculate_all_task_progress()
