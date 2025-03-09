"""Agent type definitions."""

from typing import Any, TypeAlias, TypeVar

# Type aliases
StepResult: TypeAlias = dict[str, Any]
StepKwargs: TypeAlias = dict[str, Any]
StateKwargs: TypeAlias = dict[str, Any]
Message: TypeAlias = dict[str, Any]
Context: TypeAlias = dict[str, Any]

# Generic types
T = TypeVar("T")
