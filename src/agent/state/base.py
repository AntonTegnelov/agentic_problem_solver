"""Agent state module."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.common_types.enums import AgentStep
from src.exceptions import ConfigError
from src.messages import (
    get_message_at_index,
    get_metadata_at_index,
    set_metadata_at_index,
)

if TYPE_CHECKING:
    from src.agent.agent_types.agent_types import Message, StepResult

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
        Path(path).write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

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
            from src.messages import create_structured_message

            messages = [
                create_structured_message(msg["role"], msg["content"])
                for msg in data.get("messages", [])
            ]

            # Handle step results deserialization
            from src.agent.agent_types.agent_types import Result

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


class InMemoryStateManager(StateManager):
    """In-memory state manager."""

    def __init__(self) -> None:
        """Initialize manager."""
        self._state = AgentState()

    def get_state(self) -> AgentState:
        """Get current state.

        Returns:
            Current state.

        """
        return self._state

    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: New state.

        """
        self._state = state

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
        Path(path).write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

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


class FileStateManager(StateManager):
    """File-based state manager."""

    def __init__(self, base_dir: str = "./state") -> None:
        """Initialize manager.

        Args:
            base_dir: Base directory for state files.

        """
        self._state = AgentState()
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(exist_ok=True)

    def get_state(self) -> AgentState:
        """Get current state.

        Returns:
            Current state.

        """
        return self._state

    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: New state.

        """
        self._state = state

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
        Path(path).write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

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

    def list_states(self) -> list[str]:
        """List available state files.

        Returns:
            List of state file paths.

        """
        return [str(path) for path in self._base_dir.glob("*.json")]

    def get_state_by_id(self, agent_id: str) -> AgentState:
        """Get state by agent ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent state.

        Raises:
            ConfigError: If state not found.

        """
        path = self._base_dir / f"{agent_id}.json"
        if not path.exists():
            msg = f"State not found for agent ID: {agent_id}"
            raise ConfigError(msg)

        return self.load_state(str(path))
