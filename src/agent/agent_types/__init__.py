"""Agent types package."""

from typing import Any, TypeVar, cast

from src.agent.agent_types.agent_types import (
    Agent,
    AgentRegistry,
    MockAgent,
    SimpleAgentCoordinator,
)
from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.executor import ExecutorAgent
from src.agent.agent_types.planner import PlannerAgent
from src.agent.state.base import AgentState, StateManager
from src.common_types import AgentEntry as CommonAgentEntry
from src.common_types import AgentInfo as CommonAgentInfo
from src.common_types.enums import AgentRole
from src.common_types.result_types import Result

T = TypeVar("T")

__all__ = [
    "Agent",
    "AgentRegistry",
    "CommonAgentEntry",
    "CommonAgentInfo",
    "MockAgent",
    "Result",
    "SimpleAgentCoordinator",
    "create_agent",
    "create_architect_agent",
    "create_executor_agent",
    "create_planner_agent",
]


def create_agent(
    role: AgentRole,
    provider: object = None,
    state_manager: AgentState | StateManager | None = None,
    config: dict | None = None,
) -> Agent[Any]:
    """Create an agent based on the specified role.

    Args:
        role: The role of the agent to create.
        provider: LLM provider.
        state_manager: State manager or agent state.
        config: Agent configuration.

    Returns:
        An agent instance of the specified role.

    Raises:
        ValueError: If the role is not supported.

    """
    if role == AgentRole.ARCHITECT:
        return create_architect_agent(provider, state_manager, config)
    if role == AgentRole.PLANNER:
        return create_planner_agent(provider, state_manager, config)
    if role == AgentRole.EXECUTOR:
        return create_executor_agent(provider, state_manager, config)
    msg = f"Unsupported agent role: {role}"
    raise ValueError(msg)


def create_architect_agent(
    provider: object = None,
    state_manager: AgentState | StateManager | None = None,
    config: dict | None = None,
) -> Agent[Any]:
    """Create an architect agent.

    Args:
        provider: LLM provider.
        state_manager: State manager or agent state.
        config: Agent configuration.

    Returns:
        An architect agent instance.

    """
    agent = ArchitectAgent(provider=provider, state_manager=state_manager, config=config)
    return cast(Agent[Any], agent)


def create_planner_agent(
    provider: object = None,
    state_manager: AgentState | StateManager | None = None,
    config: dict | None = None,
) -> Agent[Any]:
    """Create a planner agent.

    Args:
        provider: LLM provider.
        state_manager: State manager or agent state.
        config: Agent configuration.

    Returns:
        A planner agent instance.

    """
    agent = PlannerAgent(provider=provider, state_manager=state_manager, config=config)
    return cast(Agent[Any], agent)


def create_executor_agent(
    provider: object = None,
    state_manager: AgentState | StateManager | None = None,
    config: dict | None = None,
) -> Agent[Any]:
    """Create an executor agent.

    Args:
        provider: LLM provider.
        state_manager: State manager or agent state.
        config: Agent configuration.

    Returns:
        An executor agent instance.

    """
    agent = ExecutorAgent(provider=provider, state_manager=state_manager, config=config)
    return cast(Agent[Any], agent)
