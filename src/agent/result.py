"""Result types for agent operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from src.agent.errors import AgentError

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """Result type for operations that can fail."""

    success: bool
    error: Exception | None = None
    data: T | None = None
    message: str | None = None

    def __bool__(self) -> bool:
        """Convert to bool.

        Returns:
            True if successful, False otherwise.

        """
        return self.success

    def __str__(self) -> str:
        """Convert to string.

        Returns:
            String representation.

        """
        if self.success:
            return f"Success: {self.message or 'No message'}"
        return f"Error: {self.error}"

    @classmethod
    def ok(cls, data: T | None = None, message: str | None = None) -> Result[T]:
        """Create successful result.

        Args:
            data: Optional result data.
            message: Optional success message.

        Returns:
            Successful result.

        """
        return cls(success=True, data=data, message=message)

    @classmethod
    def error(cls, error: Exception | str) -> Result[T]:
        """Create error result.

        Args:
            error: Error that occurred.

        Returns:
            Error result.

        """
        if isinstance(error, str):
            error = Exception(error)
        return cls(success=False, error=error)

    def unwrap(self) -> T:
        """Unwrap the result value.

        Returns:
            The result value.

        Raises:
            AgentError: If the result is not successful.

        """
        if not self.success or self.data is None:
            msg = f"Cannot unwrap unsuccessful result: {self.error}"
            raise AgentError(msg)
        return self.data

    def map(self, func: callable[[T], Any]) -> Result[Any]:
        """Map the result value.

        Args:
            func: The function to apply to the result value.

        Returns:
            A new result with the mapped value.

        """
        if not self.success or self.data is None:
            return Result[Any](success=False, error=self.error, data=None, message=self.message)
        try:
            return Result.ok(func(self.data), self.message)
        except Exception as e:
            return Result.error(e)

    @classmethod
    def success(cls, data: Any = None, message: str | None = None) -> Result:
        """Create a success result.

        Args:
            data: Result data.
            message: Optional message.

        Returns:
            Success result.

        """
        return cls(success=True, data=data, message=message)

    @classmethod
    def failure(cls, message: str, data: Any = None) -> Result:
        """Create a failure result.

        Args:
            message: Error message.
            data: Optional result data.

        Returns:
            Failure result.

        """
        return cls(success=False, error=message, data=data)
