"""Base agent module."""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from src.agent.agent_types.agent_types import Agent
from src.agent.agent_types.agent_types import Result as StepResult
from src.agent.state.base import AgentState
from src.config.agent import AgentConfig
from src.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


class BaseAgent(Agent[T, U], ABC):
    """Base agent implementation."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize agent.

        Args:
            config: Optional agent configuration.

        """
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.step_executor = None
        self._provider = None
        self._config = None
        setup_logging()

    def add_step(self, step: StepResult[T]) -> None:
        """Add a processing step.

        Args:
            step: Step to add.

        """
        if self.step_executor is None:
            msg = "Step executor not initialized"
            raise ValueError(msg)
        self.step_executor.add_step(step)

    def clear_steps(self) -> None:
        """Clear all processing steps."""
        if self.step_executor is None:
            msg = "Step executor not initialized"
            raise ValueError(msg)
        self.step_executor.clear_steps()

    @abstractmethod
    def process(self, input_data: T) -> U:
        """Process input data.

        Args:
            input_data: Input data to process.

        Returns:
            Processed output.

        """
        ...

    @abstractmethod
    async def process_stream(self, input_data: str) -> AsyncGenerator[str, None]:
        """Process input data and stream results.

        Args:
            input_data: Input data to process.

        Yields:
            Processed output chunks.

        """
        yield ""

    def get_message_metadata(
        self,
        index: int,
        key: str,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Get metadata from a message at the specified index.

        Args:
            index: Message index.
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Message metadata value.

        """
        return self.state.get_message_metadata(index, key, default)

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
        self.state.set_message_metadata(index, key, value)

    def update_state(self, **kwargs: dict[str, Any]) -> None:
        """Update agent state.

        Args:
            **kwargs: State updates.

        """
        for key, value in kwargs.items():
            setattr(self.state, key, value)

    def clear_state(self) -> None:
        """Clear agent state."""
        self.state.clear()

    def update_config(self, **kwargs: dict[str, Any]) -> None:
        """Update agent configuration.

        Args:
            **kwargs: Configuration updates.

        """
        self.config.update(kwargs)
