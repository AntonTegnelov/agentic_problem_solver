"""Agent module.

This module provides the core agent functionality for the Agentic Problem Solver.

MIGRATION NOTICE:
----------------
The Agent abstract base class (ABC) is being deprecated in favor of the Protocol-based
approach in src.agent.agent_types.agent_types. See src/agent/base.py for the detailed
migration plan.

For new code, please use:
    from src.agent.agent_types import Agent

For existing code using the ABC, you can continue using:
    from src.agent import Agent  # Will show deprecation warnings

To help with migration, adapter classes are provided:
    from src.agent.adapters import ABCToProtocolAdapter, ProtocolToABCAdapter
"""

import warnings

from src.agent.adapters import ABCToProtocolAdapter, ProtocolToABCAdapter
from src.agent.base import Agent
from src.agent.result import Result as StepResult

# Emit a deprecation warning when the module is imported
warnings.warn(
    "The Agent ABC from src.agent.base is deprecated. Use src.agent.agent_types.Agent Protocol instead.",
    category=DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ABCToProtocolAdapter",
    "Agent",  # Deprecated but maintained for backward compatibility
    "ProtocolToABCAdapter",
    "StepResult",
]
