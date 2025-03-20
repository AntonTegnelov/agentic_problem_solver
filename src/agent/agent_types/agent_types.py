"""Agent type definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from src.common_types.result_types import Result
from src.messages.creation import create_human_message
from src.messages.utils import set_message_metadata, set_receiver_id, set_sender_id

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.common_types.agent_types import AgentInfo
    from src.common_types.message_types import Message


class Agent(Protocol[T]):
    """Agent protocol."""

    def process(self, message: Message) -> Result[T]:
        """Process a message.

        Args:
            message: Message to process

        Returns:
            Result containing the processed message

        Raises:
            AgentError: If processing fails

        """
        ...

    async def process_stream(self, input_data: T) -> AsyncGenerator[str, None]:
        """Process input data and stream results.

        Args:
            input_data: Input data to process.

        Yields:
            Processed output chunks.

        """
        if False:  # pragma: no cover
            _ = input_data
            yield ""

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        ...

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        ...

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        ...

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        ...

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        ...

    def get_parent_id(self) -> str | None:
        """Get parent agent ID.

        Returns:
            Parent agent ID or None if no parent.

        """
        ...

    def get_child_ids(self) -> list[str]:
        """Get child agent IDs.

        Returns:
            List of child agent IDs.

        """
        ...

    def add_child(self, child_agent_id: str) -> None:
        """Add a child agent.

        Args:
            child_agent_id: Child agent ID to add.

        """
        ...

    def remove_child(self, child_agent_id: str) -> None:
        """Remove a child agent.

        Args:
            child_agent_id: Child agent ID to remove.

        """
        ...

    def set_parent(self, parent_agent_id: str) -> None:
        """Set parent agent.

        Args:
            parent_agent_id: Parent agent ID.

        """
        ...

    def clear_parent(self) -> None:
        """Clear parent agent reference."""
        ...

    def delegate_to_child(self, child_agent_id: str, task: str) -> Result[Any]:
        """Delegate a task to a specific child agent.

        Args:
            child_agent_id: Child agent ID.
            task: Task to delegate.

        Returns:
            Result of task processing.

        Raises:
            AgentError: If child agent not found or delegation fails.

        """
        if child_agent_id not in self._child_ids:
            msg = f"Child agent {child_agent_id} not found"
            raise ValueError(msg)

        message = create_human_message(task)
        set_sender_id(message, self.agent_id)
        set_receiver_id(message, child_agent_id)
        return Result(success=True, data="Mock result")

    def collect_results_from_children(self) -> dict[str, Result[Any]]:
        """Collect results from all child agents.

        Returns:
            Dictionary mapping child agent IDs to their results.

        """
        ...


# For backward compatibility, StepResult is now imported from src.common_types.result_types


class AgentRegistry(Protocol):
    """Agent registry protocol."""

    def register_agent(self, agent: Agent[Any], info: AgentInfo) -> None:
        """Register agent.

        Args:
            agent: Agent to register.
            info: Agent information.

        """
        ...

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent.

        Args:
            agent_id: Agent ID.

        """
        ...

    def get_agent(self, agent_id: str) -> Agent[Any]:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent instance.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        ...

    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """Get agent information.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent information.

        Raises:
            AgentError: If agent not found.

        """
        ...

    def list_agents(self) -> list[AgentInfo]:
        """List all registered agents.

        Returns:
            List of agent information.

        """
        ...

    def find_agents_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents by capability.

        Args:
            capability: Capability to search for.

        Returns:
            List of agent information.

        """
        ...

    def find_agents_by_role(self, role: str) -> list[AgentInfo]:
        """Find agents by role.

        Args:
            role: Role to search for.

        Returns:
            List of agent information.

        """
        ...

    def get_parent_agent(self, agent_id: str) -> Agent[Any] | None:
        """Get parent agent of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Parent agent instance or None if no parent.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        ...

    def get_child_agents(self, agent_id: str) -> list[Agent[Any]]:
        """Get child agents of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of child agent instances.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        ...

    def get_sibling_agents(self, agent_id: str) -> list[Agent[Any]]:
        """Get sibling agents of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of sibling agent instances.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        ...

    def register_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Register parent-child relationship between agents.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        Raises:
            AgentNotFoundError: If parent or child agent not found.

        """
        ...

    def remove_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Remove parent-child relationship between agents.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        Raises:
            AgentNotFoundError: If parent or child agent not found.

        """
        ...

    def get_agent_hierarchy(self, root_agent_id: str) -> dict[str, list[str]]:
        """Get the agent hierarchy starting from the specified root agent.

        Args:
            root_agent_id: Root agent ID.

        Returns:
            Dictionary mapping agent IDs to lists of child agent IDs.

        Raises:
            AgentNotFoundError: If root agent not found.

        """
        ...


class AgentCoordinator(Protocol):
    """Agent coordinator protocol."""

    def create_agent(self, agent_type: str, config: dict[str, Any]) -> Agent[Any]:
        """Create a new agent.

        Args:
            agent_type: Type of agent to create.
            config: Agent configuration.

        Returns:
            New agent instance.

        Raises:
            ValueError: If agent type is invalid.

        """
        ...

    async def delegate_task(self, agent_id: str, task: str) -> Result:
        """Delegate task to agent.

        Args:
            agent_id: Agent ID.
            task: Task to delegate.

        Returns:
            Task result.

        Raises:
            ValueError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise ValueError(msg)

        # Create task message
        message = create_human_message(task)
        set_message_metadata(message, "receiver_id", agent_id)

        return await self._agents[agent_id].process(message)

    def delegate_task_sync(self, agent_id: str, task: str) -> Result:
        """Delegate task to agent synchronously.

        Args:
            agent_id: Agent ID.
            task: Task to delegate.

        Returns:
            Task result.

        Raises:
            ValueError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise ValueError(msg)

        message = create_human_message(task)
        return self._agents[agent_id].process(message)

    async def broadcast_task(self, task: str) -> dict[str, Result]:
        """Broadcast task to all agents.

        Args:
            task: Task to broadcast.

        Returns:
            Dictionary of agent IDs to results.

        """
        self._update_agents()  # Ensure we have the latest agents
        results = {}
        message = create_human_message(task)

        for agent_id, agent in self._agents.items():
            results[agent_id] = await agent.process(message)

        return results

    def route_message(self, message: Message) -> Result[Any]:
        """Route message to target agent.

        Args:
            message: Message to route.

        Returns:
            Result of message routing.

        Raises:
            ValueError: If target agent not found.

        """
        receiver_id = message.metadata.get("receiver_id")
        if not receiver_id:
            msg = "No receiver_id in message metadata"
            raise ValueError(msg)

        if receiver_id not in self._agents:
            msg = f"Agent not found: {receiver_id}"
            raise ValueError(msg)

        return self._agents[receiver_id].process(message)

    def get_agent_status(self, agent_id: str) -> str:
        """Get agent status.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent status.

        Raises:
            ValueError: If agent not found.

        """
        ...

    def set_agent_status(self, agent_id: str, status: str) -> None:
        """Set agent status.

        Args:
            agent_id: Agent ID.
            status: New status.

        Raises:
            ValueError: If agent not found.

        """
        ...


class SimpleAgentCoordinator:
    """Simple agent coordinator implementation."""

    def __init__(self, registry: AgentRegistry) -> None:
        """Initialize coordinator.

        Args:
            registry: Agent registry.

        """
        self.registry = registry
        self._agents = {}
        self._agent_factories = {}
        self._update_agents()

    def _update_agents(self) -> None:
        """Update agents from registry."""
        for agent_info in self.registry.list_agents():
            self._agents[agent_info.agent_id] = self.registry.get_agent(agent_info.agent_id)

    def register_agent_factory(self, agent_type: str, factory: callable) -> None:
        """Register agent factory.

        Args:
            agent_type: Agent type.
            factory: Factory function.

        """
        self._agent_factories[agent_type] = factory

    def create_agent(self, agent_type: str, config: dict[str, Any]) -> Agent:
        """Create a new agent.

        Args:
            agent_type: Type of agent to create.
            config: Agent configuration.

        Returns:
            New agent instance.

        Raises:
            ValueError: If agent type is invalid.

        """
        if agent_type not in self._agent_factories:
            msg = f"Invalid agent type: {agent_type}"
            raise ValueError(msg)

        factory = self._agent_factories[agent_type]
        return factory(config)

    def delegate_task(self, task: str, agent_id: str) -> Result[Any]:
        """Delegate task to agent.

        Args:
            task: Task to delegate.
            agent_id: Target agent ID.

        Returns:
            Task result.

        Raises:
            ValueError: If agent not found.

        """
        self._update_agents()  # Ensure we have the latest agents
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise ValueError(msg)

        message = create_human_message(task)
        return self._agents[agent_id].process(message)

    def broadcast_task(self, task: str, capability: str) -> dict[str, Result[Any]]:
        """Broadcast task to agents with capability.

        Args:
            task: Task to broadcast.
            capability: Required capability.

        Returns:
            Results by agent ID.

        """
        self._update_agents()  # Ensure we have the latest agents
        results = {}
        message = create_human_message(task)

        for agent_id, agent in self._agents.items():
            if capability in agent.capabilities:
                results[agent_id] = agent.process(message)

        return results

    def route_message(self, message: Message) -> Result[Any]:
        """Route message to target agent.

        Args:
            message: Message to route.

        Returns:
            Result of message routing.

        Raises:
            ValueError: If target agent not found.

        """
        self._update_agents()  # Ensure we have the latest agents
        receiver_id = message.metadata.get("receiver_id")
        if not receiver_id:
            msg = "No receiver_id in message metadata"
            raise ValueError(msg)

        if receiver_id not in self._agents:
            msg = f"Agent not found: {receiver_id}"
            raise ValueError(msg)

        return self._agents[receiver_id].process(message)

    def get_agent_status(self, agent_id: str) -> str:
        """Get agent status.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent status.

        Raises:
            ValueError: If agent not found.

        """
        info = self.registry.get_agent_info(agent_id)
        return info.status

    def set_agent_status(self, agent_id: str, status: str) -> None:
        """Set agent status.

        Args:
            agent_id: Agent ID.
            status: New status.

        Raises:
            ValueError: If agent not found.

        """
        info = self.registry.get_agent_info(agent_id)
        info.status = status


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, agent_id: str, capabilities: list[str]) -> None:
        """Initialize agent.

        Args:
            agent_id: Agent ID.
            capabilities: List of capabilities.

        """
        super().__init__()
        self.agent_id = agent_id
        self._capabilities = capabilities
        self.processed_messages: list[Message] = []
        self._parent_id: str | None = None
        self._child_ids: list[str] = []

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        return self.agent_id

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        return self._capabilities

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        return any(cap in task.lower() for cap in self._capabilities)

    def process(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        self.processed_messages.append(message)
        return Result(success=True, data="Mock result")

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message asynchronously.

        Args:
            message: Message to process.

        Yields:
            Processing results.

        """
        self.processed_messages.append(message)
        yield "Mock result"

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def get_parent_id(self) -> str | None:
        """Get parent agent ID.

        Returns:
            Parent agent ID or None if no parent.

        """
        return self._parent_id

    def get_child_ids(self) -> list[str]:
        """Get child agent IDs.

        Returns:
            List of child agent IDs.

        """
        return self._child_ids

    def add_child(self, child_agent_id: str) -> None:
        """Add a child agent.

        Args:
            child_agent_id: Child agent ID to add.

        """
        if child_agent_id not in self._child_ids:
            self._child_ids.append(child_agent_id)

    def remove_child(self, child_agent_id: str) -> None:
        """Remove a child agent.

        Args:
            child_agent_id: Child agent ID to remove.

        """
        if child_agent_id in self._child_ids:
            self._child_ids.remove(child_agent_id)

    def set_parent(self, parent_agent_id: str) -> None:
        """Set parent agent.

        Args:
            parent_agent_id: Parent agent ID.

        """
        self._parent_id = parent_agent_id

    def clear_parent(self) -> None:
        """Clear parent agent reference."""
        self._parent_id = None

    def delegate_to_child(self, child_agent_id: str, task: str) -> Result[Any]:
        """Delegate a task to a specific child agent.

        Args:
            child_agent_id: Child agent ID.
            task: Task to delegate.

        Returns:
            Result of task processing.

        Raises:
            AgentError: If child agent not found or delegation fails.

        """
        if child_agent_id not in self._child_ids:
            msg = f"Child agent {child_agent_id} not found"
            raise ValueError(msg)

        message = create_human_message(task)
        set_sender_id(message, self.agent_id)
        set_receiver_id(message, child_agent_id)
        return Result(success=True, data="Mock result")

    def collect_results_from_children(self) -> dict[str, Result[Any]]:
        """Collect results from all child agents.

        Returns:
            Dictionary mapping child agent IDs to their results.

        """
        return {child_id: Result(success=True, data="Mock result") for child_id in self._child_ids}
