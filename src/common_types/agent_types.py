"""Agent type definitions for cross-cutting concerns.

This module contains agent information types that are used across the codebase
for registry and coordination purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from src.common_types.enums import AgentStatus

T = TypeVar("T")


@dataclass
class AgentInfo:
    """Agent information.

    This class represents metadata about an agent in the system.
    It is used for registry and coordination purposes.
    """

    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    status: str = field(default_factory=lambda: AgentStatus.IDLE.value)


@dataclass
class AgentEntry:
    """Agent registry entry.

    This class represents an entry in the agent registry.
    It contains both the agent metadata and the agent instance.
    """

    info: AgentInfo
    agent: Any  # Using Any to avoid circular imports
