"""Base state management classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from src.agent.types import Context, Message, StateKwargs
from src.config import AgentConfig


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class AgentState:
    """Base agent state class."""

    config: AgentConfig
    status: AgentStatus = AgentStatus.IDLE
    step_count: int = 0
    retry_count: int = 0
    error: Exception | None = None
    messages: list[Message] = field(default_factory=list)
    context: Context = field(default_factory=dict)

    def update(self, **kwargs: StateKwargs) -> None:
        """Update state with new values.

        Args:
            **kwargs: Key-value pairs to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_message(self, message: Message) -> None:
        """Add a message to the state history.

        Args:
            message: Message to add.
        """
        self.messages.append(message)

    def clear_messages(self) -> None:
        """Clear message history."""
        self.messages.clear()

    def reset(self) -> None:
        """Reset state to initial values."""
        self.status = AgentStatus.IDLE
        self.step_count = 0
        self.retry_count = 0
        self.error = None
        self.clear_messages()
        self.context.clear()


class StateManager(ABC):
    """Abstract base class for state managers."""

    @abstractmethod
    def get_state(self) -> AgentState:
        """Get current state.

        Returns:
            Current agent state.
        """

    @abstractmethod
    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: New agent state.
        """

    @abstractmethod
    def update_state(self, **kwargs: StateKwargs) -> None:
        """Update state with new values.

        Args:
            **kwargs: Key-value pairs to update.
        """

    @abstractmethod
    def reset_state(self) -> None:
        """Reset state to initial values."""


@dataclass
class InMemoryStateManager(StateManager):
    """In-memory state manager implementation."""

    state: AgentState

    def get_state(self) -> AgentState:
        """Get current state.

        Returns:
            Current agent state.
        """
        return self.state

    def set_state(self, state: AgentState) -> None:
        """Set current state.

        Args:
            state: New agent state.
        """
        self.state = state

    def update_state(self, **kwargs: StateKwargs) -> None:
        """Update state with new values.

        Args:
            **kwargs: Key-value pairs to update.
        """
        self.state.update(**kwargs)

    def reset_state(self) -> None:
        """Reset state to initial values."""
        self.state.reset()
