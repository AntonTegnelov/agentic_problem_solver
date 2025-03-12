"""Result types for operations.

This module contains the Result class, which is the preferred implementation
for operation results throughout the codebase. It is a generic utility for operation results
and is similar to a "Result" or "Either" monad pattern used in many languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from src.common_types.error_types import AgentError

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Result(Generic[T]):
    """Result type for operations that can fail.

    This is the preferred implementation for operation results throughout the codebase.
    It is used by both the Agent ABC and the Agent Protocol.
    """

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
    def create_error(cls, error: Exception, message: str | None = None) -> Result[T]:
        """Create error result.

        Args:
            error: Error that occurred.
            message: Optional error message.

        Returns:
            Error result.

        """
        return cls(success=False, error=error, message=message)

    def unwrap(self) -> T:
        """Unwrap result data.

        Returns:
            Result data.

        Raises:
            AgentError: If result is not successful or data is None.

        """
        if not self.success:
            msg = f"Cannot unwrap unsuccessful result: {self.error}"
            raise self.error.__class__(msg) if self.error else AgentError(msg)
        if self.data is None:
            msg = "Cannot unwrap None data"
            raise AgentError(msg)
        return self.data

    def map(self, func: Callable[[T], U]) -> Result[U]:
        """Map result data.

        Args:
            func: Function to apply to data.

        Returns:
            New result with mapped data.

        """
        if not self.success:
            return Result(success=False, error=self.error, message=self.message)
        return Result(
            success=True,
            data=func(self.data) if self.data is not None else None,
            error=self.error,
            message=self.message,
        )

    @classmethod
    def success(cls, data: T | None = None, message: str | None = None) -> Result[T]:
        """Create successful result.

        Args:
            data: Optional result data.
            message: Optional success message.

        Returns:
            Successful result.

        """
        return cls(success=True, data=data, message=message)

    @classmethod
    def failure(cls, error: Exception, message: str | None = None) -> Result[T]:
        """Create failure result.

        Args:
            error: Error that occurred.
            message: Optional error message.

        Returns:
            Failure result.

        """
        return cls(success=False, error=error, message=message)
