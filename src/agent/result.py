"""Result types for agent operations.

DEPRECATED: This module is deprecated and will be removed in a future version.
The Result class has been moved to src.common_types.result_types.
Please update your imports to use the new location:

    from src.common_types.result_types import Result

Known places that need to be updated before this file is deleted:
1. tests/unit/test_utils.py
2. tests/unit/test_solver_agent.py
3. tests/unit/test_provider_selection.py
4. tests/unit/test_message_processor.py
5. tests/unit/test_agent_types.py
6. tests/unit/test_agent_steps.py
7. tests/unit/test_agent_result.py
8. tests/integration/test_provider_factory.py
9. tests/integration/test_agent_steps.py
10. src/messages/processor.py
11. src/llm_providers/providers/base.py
12. src/agent/__init__.py (imports as StepResult)
13. src/agent/steps.py
14. src/agent/solver.py
15. src/agent/agent_types/__init__.py
16. src/agent/agent_types/agent_types.py
17. src/agent/state/base.py

The move process:
1. Create src/common_types/result_types.py with the Result class (DONE)
2. Update this file with deprecation warnings (DONE)
3. Update imports in the codebase to use the new location
4. Remove this file once all imports have been updated

This module contains the Result class, which is the preferred implementation
for operation results throughout the codebase. It is used by both the Agent ABC
and the Agent Protocol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from src.common_types import AgentError

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Result(Generic[T]):
    """Result type for operations that can fail.

    DEPRECATED: This class has been moved to src.common_types.result_types.
    Please update your imports to use the new location:

        from src.common_types.result_types import Result

    This is the preferred implementation for operation results throughout the codebase.
    It is used by both the Agent ABC and the Agent Protocol.
    """

    success: bool
    error: Exception | None = None
    data: T | None = None
    message: str | None = None

    def __init__(
        self,
        *,
        success: bool,
        error: Exception | None = None,
        data: T | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize Result.

        Args:
            success: Whether the operation was successful.
            error: Optional error that occurred.
            data: Optional result data.
            message: Optional message.

        """
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.success = success
        self.error = error
        self.data = data
        self.message = message

    def __bool__(self) -> bool:
        """Convert to bool.

        Returns:
            True if successful, False otherwise.

        """
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.success

    def __str__(self) -> str:
        """Convert to string.

        Returns:
            String representation.

        """
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(success=False, error=error, message=message)

    def unwrap(self) -> T:
        """Unwrap result data.

        Returns:
            Result data.

        Raises:
            AgentError: If result is not successful or data is None.

        """
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.success:
            msg = f"Cannot unwrap unsuccessful result: {self.error}"
            raise AgentError(msg)
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
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "The Result class has been moved to src.common_types.result_types. "
            "Please update your imports to use the new location.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(success=False, error=error, message=message)
