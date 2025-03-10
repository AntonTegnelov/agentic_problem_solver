"""Result types for agent operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from src.agent.errors import AgentError

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """Result of an agent operation.

    Attributes:
        success: Whether the operation was successful.
        value: The result value if successful.
        error: The error if unsuccessful.
        metadata: Additional metadata about the result.

    """

    success: bool
    value: T | None = None
    error: Exception | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def success_result(cls, value: T, metadata: dict[str, Any] | None = None) -> Result[T]:
        """Create a successful result.

        Args:
            value: The result value.
            metadata: Additional metadata.

        Returns:
            A successful result.

        """
        return cls(success=True, value=value, metadata=metadata)

    @classmethod
    def error_result(
        cls,
        error: Exception | str,
        metadata: dict[str, Any] | None = None,
    ) -> Result[T]:
        """Create an error result.

        Args:
            error: The error that occurred.
            metadata: Additional metadata.

        Returns:
            An error result.

        """
        if isinstance(error, str):
            error = AgentError(error)
        return cls(success=False, error=error, metadata=metadata)

    def unwrap(self) -> T:
        """Unwrap the result value.

        Returns:
            The result value.

        Raises:
            AgentError: If the result is not successful.

        """
        if not self.success or self.value is None:
            msg = f"Cannot unwrap unsuccessful result: {self.error}"
            raise AgentError(msg)
        return self.value

    def map(self, func: callable[[T], Any]) -> Result[Any]:
        """Map the result value.

        Args:
            func: The function to apply to the result value.

        Returns:
            A new result with the mapped value.

        """
        if not self.success or self.value is None:
            return Result[Any](success=False, error=self.error, metadata=self.metadata)
        try:
            return Result.success_result(func(self.value), self.metadata)
        except Exception as e:
            return Result.error_result(e, self.metadata)
