"""Agent module.

This module provides the core agent functionality for the Agentic Problem Solver.

The Agent Protocol from src.agent.agent_types.agent_types is the only way
to define and use agents in the codebase.

please use:
    from src.agent.agent_types import Agent
"""

from src.agent.agent_types import Agent
from src.agent.result import Result as StepResult

__all__ = [
    "Agent",
    "StepResult",
]
