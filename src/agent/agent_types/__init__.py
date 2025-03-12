"""Agent types module.

This module provides the Protocol-based Agent definition and related types.
It is the preferred way to define and use agents in the codebase.
"""

# Re-export the Agent Protocol and related types
from src.agent.agent_types.agent_types import Agent, StepResult

# Re-export the Result class from src.agent.result to avoid deprecation warnings
from src.agent.result import Result
from src.common_types.message_types import Message

__all__ = ["Agent", "Message", "Result", "StepResult"]
