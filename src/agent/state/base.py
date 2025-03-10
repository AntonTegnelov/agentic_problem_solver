"""Agent state module."""

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from src.agent.agent_types.agent_types import Message
from src.common_types.enums import AgentStep

T = TypeVar("T")


@dataclass
class Context:
    """Agent context."""

    data: dict[str, Any] = field(default_factory=dict)


class StateManager(Protocol):
    """State manager protocol."""

    def get_state(self) -> "AgentState":
        """Get current state.

        Returns:
            Current state.

        """
        ...

    def set_state(self, state: "AgentState") -> None:
        """Set current state.

        Args:
            state: New state.

        """
        ...

    def clear_state(self) -> None:
        """Clear current state."""
        ...


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

    def add_message(self, message: Message) -> None:
        """Add message to state.

        Args:
            message: Message to add.

        """
        self.messages.append(message)

    def get_message(self, index: int) -> Message:
        """Get message at index.

        Args:
            index: Message index.

        Returns:
            Message at index.

        Raises:
            IndexError: If index is out of range.

        """
        return self.messages[index]

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
        message = self.get_message(index)
        if hasattr(message, "metadata"):
            metadata = getattr(message, "metadata", {})
            return metadata.get(key, default)
        return default

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
        message = self.get_message(index)
        if not hasattr(message, "metadata"):
            message.metadata = {}
        metadata = message.metadata
        metadata[key] = value

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

    def clear(self) -> None:
        """Clear state."""
        self.messages.clear()
        self.context.data.clear()
        self.execution_result = ""
        self.step_count = 0
        self.task_completed = False
        self.error = None
        self.current_step = AgentStep.UNDERSTAND


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
