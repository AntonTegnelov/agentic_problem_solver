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
from src.common_types.enums import AgentStep
from src.common_types.message_types import Message
from src.common_types.result_types import Result
from src.common_types.result_types import Result as StepResult
from src.common_types.task_types import Task, TaskDependency, TaskStatus
from src.messages.creation import create_structured_message
from src.messages.utils import (
    get_message_at_index,
    get_metadata_at_index,
    set_metadata_at_index,
)

if TYPE_CHECKING:
    from src.agent.agent_types.agent_types import Agent, Message, Result, StepResult

T = TypeVar("T")


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

    def get_agent_for_step(self, step: AgentStep) -> Agent:
        """Get agent for step.

        Args:
            step: Step to get agent for.

        Returns:
            Agent for step.

        Raises:
            ConfigError: If no agent found for step.

        """
        if not self._agents:
            msg = "No agents registered"
            raise ConfigError(msg)

        # In the future, we could have different agents for different steps
        # For now, log the step and return the first agent
        step.name if hasattr(step, "name") else str(step)

        # Return the first agent regardless of step
        return next(iter(self._agents.values()))

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
