"""Agent type definitions."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Message:
    """Message exchanged between agents."""

    role: str
    content: str


@dataclass
class Result(Generic[T]):
    """Step execution result."""

    success: bool
    data: T
    error: str = ""

    def __post_init__(self) -> None:
        """Validate result."""
        if not self.success and not self.error:
            msg = "Error message is required when success is False"
            raise ValueError(msg)


class Agent(Protocol[T, U]):
    """Agent protocol.

    This protocol defines the interface that all agents must implement.

    Type Parameters:
        T: Type of input data.
        U: Type of output data.
    """

    def process(self, input_data: T) -> U:
        """Process input data.

        Args:
            input_data: Input data to process.

        Returns:
            Processed output.

        """
        ...

    async def process_stream(self, input_data: T) -> AsyncGenerator[str, None]:
        """Process input data and stream results.

        Args:
            input_data: Input data to process.

        Yields:
            Processed output chunks.

        """
        if False:  # pragma: no cover
            _ = input_data
            yield ""


StepResult = Result[T]
