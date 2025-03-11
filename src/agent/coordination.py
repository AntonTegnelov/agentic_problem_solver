"""Agent coordination module."""

from __future__ import annotations

from typing import Any, Callable

from src.agent.agent_types.agent_types import Agent, AgentInfo, Result, StepResult
from src.common_types.message_types import Message
from src.exceptions import AgentNotFoundError


class AgentRegistry:
    """Agent registry for managing agents."""

    def __init__(self) -> None:
        """Initialize agent registry."""
        self._agents: dict[str, Agent] = {}
        self._agent_info: dict[str, AgentInfo] = {}

    def register_agent(self, agent: Agent, info: AgentInfo | None = None) -> None:
        """Register agent.

        Args:
            agent: Agent instance.
            info: Optional agent info.

        """
        agent_id = agent.get_agent_id()
        self._agents[agent_id] = agent
        if info:
            self._agent_info[agent_id] = info

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent.

        Args:
            agent_id: Agent ID.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        self._agents.pop(agent_id)
        if agent_id in self._agent_info:
            self._agent_info.pop(agent_id)

    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent instance.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)
        return self._agents[agent_id]

    def get_agent_info(self, agent_id: str) -> AgentInfo | None:
        """Get agent info.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent info or None if not found.

        """
        return self._agent_info.get(agent_id)

    def list_agents(self) -> list[Agent]:
        """List registered agents.

        Returns:
            List of agents.

        """
        return list(self._agents.values())

    def get_agents(self) -> dict[str, Agent]:
        """Get all agents.

        Returns:
            Dictionary of agent IDs to agents.

        """
        return self._agents.copy()

    def find_agents_by_capability(self, capability: str) -> list[Agent]:
        """Find agents by capability.

        Args:
            capability: Capability to search for.

        Returns:
            List of agents with the capability.

        """
        result = []
        for agent_id, info in self._agent_info.items():
            if info and capability in info.capabilities:
                result.append(self._agents[agent_id])
        return result

    def find_agents_by_parent(self, parent_id: str) -> list[Agent]:
        """Find agents by parent ID.

        Args:
            parent_id: Parent ID to search for.

        Returns:
            List of agents with the parent ID.

        """
        result = []
        for agent_id, info in self._agent_info.items():
            if info and info.parent_id == parent_id:
                result.append(self._agents[agent_id])
        return result


class AgentCoordinator:
    """Agent coordinator for delegating tasks to agents."""

    def __init__(self, registry: AgentRegistry) -> None:
        """Initialize agent coordinator.

        Args:
            registry: Agent registry.

        """
        self._registry = registry
        self._factories: dict[str, Callable[..., Agent]] = {}

    def register_agent_factory(self, agent_type: str, factory: Callable[..., Agent]) -> None:
        """Register agent factory.

        Args:
            agent_type: Agent type.
            factory: Agent factory function.

        """
        self._factories[agent_type] = factory

    async def delegate_task(self, agent_id: str, task: str) -> Result:
        """Delegate task to agent.

        Args:
            agent_id: Agent ID.
            task: Task to delegate.

        Returns:
            Task result.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        agent = self._registry.get_agent(agent_id)
        message = Message(content=task, role="human")
        return await agent.process(message)

    async def broadcast_task(self, task: str) -> dict[str, Result]:
        """Broadcast task to all agents.

        Args:
            task: Task to broadcast.

        Returns:
            Dictionary of agent IDs to results.

        """
        results = {}
        message = Message(content=task, role="human")

        for agent_id, agent in self._registry.get_agents().items():
            results[agent_id] = await agent.process(message)

        return results

    def find_agent_for_task(self, task: str) -> str | None:
        """Find agent for task.

        Args:
            task: Task to find agent for.

        Returns:
            Agent ID or None if no suitable agent found.

        """
        for agent_id, info in self._registry._agent_info.items():
            if info and info.can_handle_task(task):
                return agent_id
        return None

    def create_agent(self, agent_type: str, config: dict, **kwargs: Any) -> Agent:
        """Create agent.

        Args:
            agent_type: Agent type.
            config: Agent configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Created agent.

        Raises:
            ValueError: If agent type not registered.

        """
        if agent_type not in self._factories:
            msg = f"Invalid agent type: {agent_type}"
            raise ValueError(msg)

        agent = self._factories[agent_type](config=config, **kwargs)
        config.get("agent_id", agent.get_agent_id())
        self._registry.register_agent(agent, None)
        return agent

    async def route_message(self, message: Message, target_agent_id: str) -> StepResult:
        """Route message to target agent.

        Args:
            message: Message to route.
            target_agent_id: Target agent ID.

        Returns:
            Result of message processing.

        Raises:
            AgentNotFoundError: If target agent not found.

        """
        agent = self._registry.get_agent(target_agent_id)
        return await agent.process(message)


class AgentFactory:
    """Agent factory for creating agents."""

    def __init__(self, registry: AgentRegistry) -> None:
        """Initialize agent factory.

        Args:
            registry: Agent registry.

        """
        self._registry = registry
        self._factories: dict[str, Callable[..., Agent]] = {}

    def register_factory(self, agent_type: str, factory: Callable[..., Agent]) -> None:
        """Register agent factory.

        Args:
            agent_type: Agent type.
            factory: Agent factory function.

        """
        self._factories[agent_type] = factory

    def create_agent(self, agent_type: str, agent_id: str, **kwargs: Any) -> Agent:
        """Create agent.

        Args:
            agent_type: Agent type.
            agent_id: Agent ID.
            **kwargs: Additional keyword arguments.

        Returns:
            Created agent.

        Raises:
            ValueError: If agent type not registered.

        """
        if agent_type not in self._factories:
            msg = f"Agent type not registered: {agent_type}"
            raise ValueError(msg)

        agent = self._factories[agent_type](**kwargs)
        self._registry.register_agent(agent, None)
        return agent


# Aliases for backward compatibility
InMemoryAgentRegistry = AgentRegistry
SimpleAgentCoordinator = AgentCoordinator

# Export symbols
__all__ = [
    "Agent",
    "AgentCoordinator",
    "AgentFactory",
    "AgentInfo",
    "AgentRegistry",
    "InMemoryAgentRegistry",
    "SimpleAgentCoordinator",
]
