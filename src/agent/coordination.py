"""Agent coordination module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agent.agent_types.agent_types import Agent, AgentRegistry
from src.common_types import AgentInfo, AgentNotFoundError
from src.common_types.message_types import Message

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.common_types.result_types import Result
    from src.common_types.result_types import Result as StepResult


class InMemoryAgentRegistry(AgentRegistry):
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
        else:
            # Create default agent info if not provided
            agent_type = agent_id.split("_")[0] if "_" in agent_id else "unknown"
            self._agent_info[agent_id] = AgentInfo(
                agent_id=agent_id,
                name=f"{agent_type.capitalize()} Agent",
                description=f"A {agent_type} agent with ID {agent_id}",
                capabilities=agent.get_capabilities(),
                status="idle",
                parent_id=agent.get_parent_id(),
                child_ids=agent.get_child_ids(),
            )

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

        # Remove parent-child relationships
        agent = self._agents[agent_id]
        parent_id = agent.get_parent_id()
        if parent_id and parent_id in self._agents:
            parent_agent = self._agents[parent_id]
            parent_agent.remove_child(agent_id)

        # Remove this agent as parent from all its children
        for child_id in agent.get_child_ids():
            if child_id in self._agents:
                child_agent = self._agents[child_id]
                child_agent.clear_parent()

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

    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """Get agent info.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent info.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agent_info:
            msg = f"Agent info not found: {agent_id}"
            raise AgentNotFoundError(msg)
        return self._agent_info[agent_id]

    def list_agents(self) -> list[AgentInfo]:
        """List registered agents.

        Returns:
            List of agent information.

        """
        return list(self._agent_info.values())

    def get_agents(self) -> dict[str, Agent]:
        """Get all agents.

        Returns:
            Dictionary of agent IDs to agents.

        """
        return self._agents.copy()

    def find_agents_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents by capability.

        Args:
            capability: Capability to search for.

        Returns:
            List of agent information.

        """
        result = []
        for agent_id, agent in self._agents.items():
            if capability in agent.get_capabilities():
                result.append(self._agent_info[agent_id])
        return result

    def find_agents_by_role(self, role: str) -> list[AgentInfo]:
        """Find agents by role.

        Args:
            role: Role to search for.

        Returns:
            List of agent information.

        """
        result = []
        for info in self._agent_info.values():
            if hasattr(info, "role") and info.role == role:
                result.append(info)
        return result

    def find_agents_by_parent(self, parent_id: str) -> list[AgentInfo]:
        """Find agents by parent ID.

        Args:
            parent_id: Parent ID to search for.

        Returns:
            List of agent information with the specified parent.

        """
        result = []
        for info in self._agent_info.values():
            if info.parent_id == parent_id:
                result.append(info)
        return result

    def get_parent_agent(self, agent_id: str) -> Agent | None:
        """Get parent agent of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Parent agent instance or None if no parent.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        agent = self._agents[agent_id]
        parent_id = agent.get_parent_id()

        if parent_id and parent_id in self._agents:
            return self._agents[parent_id]

        return None

    def get_child_agents(self, agent_id: str) -> list[Agent]:
        """Get child agents of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of child agent instances.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        agent = self._agents[agent_id]
        child_ids = agent.get_child_ids()

        return [self._agents[child_id] for child_id in child_ids if child_id in self._agents]

    def get_sibling_agents(self, agent_id: str) -> list[Agent]:
        """Get sibling agents of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of sibling agent instances.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        agent = self._agents[agent_id]
        parent_id = agent.get_parent_id()

        if not parent_id or parent_id not in self._agents:
            return []

        parent_agent = self._agents[parent_id]
        sibling_ids = [
            child_id for child_id in parent_agent.get_child_ids() if child_id != agent_id and child_id in self._agents
        ]

        return [self._agents[sibling_id] for sibling_id in sibling_ids]

    def register_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Register parent-child relationship between agents.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        Raises:
            AgentNotFoundError: If parent or child agent not found.

        """
        if parent_id not in self._agents:
            msg = f"Parent agent not found: {parent_id}"
            raise AgentNotFoundError(msg)

        if child_id not in self._agents:
            msg = f"Child agent not found: {child_id}"
            raise AgentNotFoundError(msg)

        parent_agent = self._agents[parent_id]
        child_agent = self._agents[child_id]

        # Remove existing parent-child relationship if any
        existing_parent_id = child_agent.get_parent_id()
        if existing_parent_id and existing_parent_id in self._agents:
            existing_parent = self._agents[existing_parent_id]
            existing_parent.remove_child(child_id)

        # Set new parent-child relationship
        parent_agent.add_child(child_id)
        child_agent.set_parent(parent_id)

        # Update agent info
        if parent_id in self._agent_info:
            self._agent_info[parent_id].child_ids = parent_agent.get_child_ids()

        if child_id in self._agent_info:
            self._agent_info[child_id].parent_id = parent_id

    def remove_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Remove parent-child relationship between agents.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        Raises:
            AgentNotFoundError: If parent or child agent not found.

        """
        if parent_id not in self._agents:
            msg = f"Parent agent not found: {parent_id}"
            raise AgentNotFoundError(msg)

        if child_id not in self._agents:
            msg = f"Child agent not found: {child_id}"
            raise AgentNotFoundError(msg)

        parent_agent = self._agents[parent_id]
        child_agent = self._agents[child_id]

        # Only remove if the relationship exists
        if child_agent.get_parent_id() == parent_id and child_id in parent_agent.get_child_ids():
            parent_agent.remove_child(child_id)
            child_agent.clear_parent()

            # Update agent info
            if parent_id in self._agent_info:
                self._agent_info[parent_id].child_ids = parent_agent.get_child_ids()

            if child_id in self._agent_info:
                self._agent_info[child_id].parent_id = None

    def get_agent_hierarchy(self, root_agent_id: str) -> dict[str, list[str]]:
        """Get the agent hierarchy starting from the specified root agent.

        Args:
            root_agent_id: Root agent ID.

        Returns:
            Dictionary mapping agent IDs to lists of child agent IDs.

        Raises:
            AgentNotFoundError: If root agent not found.

        """
        if root_agent_id not in self._agents:
            msg = f"Root agent not found: {root_agent_id}"
            raise AgentNotFoundError(msg)

        hierarchy = {}
        self._build_hierarchy(root_agent_id, hierarchy)
        return hierarchy

    def _build_hierarchy(self, agent_id: str, hierarchy: dict[str, list[str]]) -> None:
        """Recursively build the agent hierarchy.

        Args:
            agent_id: Current agent ID.
            hierarchy: Dictionary to populate with the hierarchy.

        """
        if agent_id not in self._agents:
            return

        agent = self._agents[agent_id]
        child_ids = agent.get_child_ids()
        valid_child_ids = [child_id for child_id in child_ids if child_id in self._agents]

        hierarchy[agent_id] = valid_child_ids

        for child_id in valid_child_ids:
            self._build_hierarchy(child_id, hierarchy)


class AgentCoordinator:
    """Agent coordinator for delegating tasks to agents."""

    def __init__(self, registry: InMemoryAgentRegistry) -> None:
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
        for agent_id in self._registry.get_agents():
            info = self._registry.get_agent_info(agent_id)
            if info and hasattr(info, "can_handle_task") and info.can_handle_task(task):
                return agent_id
        return None

    def create_agent(self, agent_type: str, config: dict, **kwargs: dict[str, Any]) -> Agent:
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

    def __init__(self, registry: InMemoryAgentRegistry) -> None:
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

    def create_agent(self, agent_type: str, **kwargs: dict[str, Any]) -> Agent:
        """Create agent.

        Args:
            agent_type: Agent type.
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

# For backward compatibility
SimpleAgentCoordinator = AgentCoordinator
