"""Agent state module."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.agent_types.agent_types import Agent
from src.common_types import AgentNotFoundError, ConfigError
from src.common_types.enums import AgentStep, ExecutionStage, VerificationStatus
from src.common_types.message_types import Message
from src.common_types.result_types import Result
from src.common_types.result_types import Result as StepResult
from src.common_types.task_types import Task, TaskComplexity, TaskDependency, TaskPriority, TaskStatus
from src.messages.creation import create_structured_message
from src.messages.utils import (
    get_message_at_index,
    get_metadata_at_index,
    set_metadata_at_index,
)

if TYPE_CHECKING:
    from src.agent.agent_types.agent_types import Agent, Message, Result, StepResult

T = TypeVar("T")

# Constants
MAX_EXECUTION_ATTEMPTS = 3


@dataclass
class Context:
    """Agent context."""

    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

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


class StateManager(Protocol):
    """State manager protocol."""

    def get_state(self) -> AgentState:
        """Get current state.

        Returns:
            Current state.

        """

    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: New state.

        """

    def clear_state(self) -> None:
        """Clear current state."""

    def save_state(self, path: str | None = None) -> str:
        """Save state to file.

        Args:
            path: Optional file path to save state to.

        Returns:
            Path where state was saved.

        """
        if not self.state:
            msg = "No state to save"
            raise ConfigError(msg)

        # Generate path if not provided
        if path is None:
            path = str(Path(self.base_dir) / f"{self.state.agent_id}.json")

        # Convert state to dict
        state_dict = self.state.to_dict()

        # Save to file
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

        return path

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
    agent_id: str = field(default="")
    parent_agent_id: str | None = field(default=None)
    _agents: dict[str, Agent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize state with default values."""
        # Initialize context metadata if empty
        if not self.context.metadata:
            self.context.metadata = {
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "change_count": 0,
            }

        # Generate agent_id if not provided
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())

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

    def clear(self) -> None:
        """Clear state."""
        self.messages.clear()
        self.context.data.clear()
        self.context.metadata = {
            "created_at": self.created_at,
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
        self._agents.clear()

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
        """Create state from dictionary.

        Args:
            data: State dictionary.

        Returns:
            State instance.

        Raises:
            ConfigError: If state creation fails.

        """
        try:
            # Handle message deserialization
            messages = []
            for msg in data.get("messages", []):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                msg.get("metadata", {})

                # Map role to standard types
                if role == "unknown":
                    role = "system"  # Default to system for unknown roles

                messages.append(create_structured_message(role, content))

            # Handle step results deserialization
            from src.common_types.result_types import Result

            step_results = {
                step: Result(
                    success=result["success"],
                    error=result["error"],
                    data=result["data"],  # Keep as string
                )
                for step, result in data.get("step_results", {}).items()
            }

            # Create context
            context = Context(
                data=data.get("context", {}).get("data", {}),
                metadata=data.get("context", {}).get("metadata", {}),
            )

            # Create state
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
                parent_agent_id=data.get("parent_agent_id"),
            )

        except Exception as e:
            msg = f"Failed to create state from dictionary: {e}"
            raise ConfigError(msg) from e

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

        for i, task_dict in enumerate(tasks):
            if task_dict["task_id"] == task_id_str:
                # Convert Task object to dict for storage
                updated_task = asdict(task)
                # Convert UUID objects to strings for JSON serialization
                updated_task["task_id"] = task_id_str
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

                tasks[i] = updated_task
                self.set_context("tasks", tasks)
                self.updated_at = datetime.now(UTC).isoformat()
                return

        # If task not found, add it
        self.add_task(task)

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

        # Initialize counters and lists
        total_subtasks = len(task.subtasks)
        completed_subtasks = 0
        in_progress_subtasks = 0
        blocked_subtasks = 0
        failed_subtasks = 0

        # Priority weights - higher priority tasks count more toward progress
        priority_weights = {
            TaskPriority.LOW.value: 0.5,
            TaskPriority.MEDIUM.value: 1.0,
            TaskPriority.HIGH.value: 1.5,
            TaskPriority.CRITICAL.value: 2.0,
        }

        # Complexity weights - more complex tasks count more toward progress
        complexity_weights = {
            TaskComplexity.SIMPLE.value: 0.75,
            TaskComplexity.MODERATE.value: 1.0,
            TaskComplexity.COMPLEX.value: 1.5,
            TaskComplexity.VERY_COMPLEX.value: 2.0,
        }

        # Track weighted progress
        total_weight = 0.0
        weighted_progress = 0.0

        # Track critical path (tasks with dependencies)
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

        # Now process tasks bottom-up, starting with leaf tasks
        # First, check for stalled leaf tasks
        for leaf_task in leaf_tasks:
            self._check_for_stalled_task(leaf_task.task_id)

        # Then process parent tasks level by level, from lowest to highest
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

        # Finally, update the overall progress information
        overall_progress = self.get_overall_progress()
        self.set_context("overall_progress", overall_progress)

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


class InMemoryStateManager(StateManager):
    """In-memory state manager."""

    def __init__(self) -> None:
        """Initialize in-memory state manager."""
        super().__init__()
        self._state = AgentState()
        self._states: dict[str, AgentState] = {}

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
        if state.agent_id:
            self._states[state.agent_id] = state

    def clear_state(self) -> None:
        """Clear current state."""
        self._state.clear()

    def save_state(self, path: str | None = None) -> str:
        """Save state to file.

        Args:
            path: Optional file path to save state to.

        Returns:
            Path where state was saved.

        """
        # Validate state before saving
        self._state.validate()

        # Generate path if not provided
        if path is None:
            state_dir = Path("./state")
            state_dir.mkdir(exist_ok=True)
            path = str(state_dir / f"{self._state.agent_id}.json")

        # Convert state to dict
        state_dict = self._state.to_dict()

        # Save to file
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

        # Store state in memory
        self._states[self._state.agent_id] = self._state

        return path

    def load_state(self, path: str) -> AgentState:
        """Load state from file.

        Args:
            path: Path to state file.

        Returns:
            Loaded state.

        Raises:
            ConfigError: If state loading fails.

        """
        try:
            # Check if file exists
            if not Path(path).exists():
                self._raise_file_not_found(path)

            # Load from file
            state_dict = json.loads(Path(path).read_text(encoding="utf-8"))

            # Create state from dict
            state = AgentState.from_dict(state_dict)

            # Validate loaded state
            state.validate()

            # Set as current state
            self._state = state
            self._states[state.agent_id] = state
        except Exception as e:
            msg = f"Failed to load state: {e}"
            raise ConfigError(msg) from e
        else:
            return state

    def _raise_file_not_found(self, path: str) -> None:
        """Raise ConfigError for file not found.

        Args:
            path: Path to state file.

        Raises:
            ConfigError: Always raised.

        """
        msg = f"State file not found: {path}"
        raise ConfigError(msg)

    def list_states(self) -> list[str]:
        """List available states.

        Returns:
            List of state IDs.

        """
        return list(self._states.keys())

    def delete_state(self, agent_id: str) -> None:
        """Delete state.

        Args:
            agent_id: Agent ID.

        """
        if agent_id in self._states:
            del self._states[agent_id]
            state_path = Path("./state") / f"{agent_id}.json"
            if state_path.exists():
                state_path.unlink()


class FileStateManager(StateManager):
    """File-based state manager."""

    def __init__(self, base_dir: str | None = None) -> None:
        """Initialize state manager.

        Args:
            base_dir: Base directory for state files.

        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "state"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.state = None

    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: State to set.

        """
        self.state = state

    def get_state(self) -> AgentState | None:
        """Get current state.

        Returns:
            Current state or None if no state exists.

        """
        if not self.state:
            self.state = AgentState()
        return self.state

    def save_state(self) -> str:
        """Save current state.

        Returns:
            Path to saved state file.

        Raises:
            ConfigError: If no state to save.

        """
        if not self.state:
            msg = "No state to save"
            raise ConfigError(msg)

        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Get path for state file
        state_path = self.get_state_path()

        # Save state to file
        Path(state_path).write_text(json.dumps(self.state.to_dict()), encoding="utf-8")

        return state_path

    def load_state(self, path: str) -> None:
        """Load state from file.

        Args:
            path: Path to state file.

        """
        state_path = Path(path)
        if not state_path.exists():
            msg = f"State file not found: {path}"
            raise ConfigError(msg)

        data = json.loads(state_path.read_text(encoding="utf-8"))
        self.state = AgentState.from_dict(data)

    def list_states(self) -> list[str]:
        """List available states.

        Returns:
            List of state IDs.

        """
        if not self.base_dir.exists():
            return []

        return [file.stem for file in self.base_dir.glob("*.json")]

    def delete_state(self, agent_id: str) -> None:
        """Delete state file.

        Args:
            agent_id: Agent ID.

        """
        path = Path(self.get_state_path(agent_id))
        if path.exists():
            path.unlink()

    def get_state_path(self) -> str:
        """Get path for state file.

        Returns:
            Path to state file.

        Raises:
            ConfigError: If no state to save.

        """
        if not self.state:
            msg = "No state to save"
            raise ConfigError(msg)

        return str(self.base_dir / f"{self.state.agent_id}.json")

    def get_state_by_id(self, agent_id: str) -> AgentState:
        """Get state by agent ID.

        Args:
            agent_id: Agent ID.

        Returns:
            State if found.

        Raises:
            ConfigError: If state not found.

        """
        state_path = self.base_dir / f"{agent_id}.json"
        if not state_path.exists():
            msg = f"State not found: {agent_id}"
            raise ConfigError(msg)

        state_dict = json.loads(state_path.read_text(encoding="utf-8"))
        return AgentState.from_dict(state_dict)
