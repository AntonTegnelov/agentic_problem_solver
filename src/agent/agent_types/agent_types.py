"""Agent type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from src.agent.base import Agent
from src.agent.errors import AgentError, AgentNotFoundError
from src.agent.result import Result
from src.common_types.enums import AgentStatus
from src.messages import create_human_message, set_message_metadata

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.common_types.message_types import Message


@dataclass
class AgentInfo:
    """Agent information."""

    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    parent_id: str | None = None
    status: str = field(default_factory=lambda: AgentStatus.IDLE.value)


@dataclass
class AgentEntry:
    """Agent registry entry."""

    info: AgentInfo
    agent: Agent


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


StepResult = Result[T]


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
            List of matching agent information.

        """
        ...

    def find_agents_by_parent(self, parent_id: str) -> list[AgentInfo]:
        """Find agents by parent ID.

        Args:
            parent_id: Parent agent ID.

        Returns:
            List of child agent information.

        """
        ...


class InMemoryAgentRegistry(AgentRegistry):
    """In-memory agent registry implementation."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._agents: dict[str, AgentEntry] = {}

    def register_agent(self, agent: Agent[Any], info: AgentInfo) -> None:
        """Register agent.

        Args:
            agent: Agent to register.
            info: Agent information.

        """
        self._agents[info.agent_id] = AgentEntry(info=info, agent=agent)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent.

        Args:
            agent_id: Agent ID.

        """
        if agent_id in self._agents:
            del self._agents[agent_id]

    def get_agent(self, agent_id: str) -> Agent[Any]:
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
        return self._agents[agent_id].agent

    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """Get agent information.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent information.

        Raises:
            AgentError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentError(msg)
        return self._agents[agent_id].info

    def list_agents(self) -> list[AgentInfo]:
        """List all registered agents.

        Returns:
            List of agent information.

        """
        return [entry.info for entry in self._agents.values()]

    def find_agents_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents by capability.

        Args:
            capability: Capability to search for.

        Returns:
            List of matching agent information.

        """
        return [
            entry.info for entry in self._agents.values() if capability in entry.info.capabilities
        ]

    def find_agents_by_parent(self, parent_id: str) -> list[AgentInfo]:
        """Find agents by parent ID.

        Args:
            parent_id: Parent agent ID.

        Returns:
            List of child agent information.

        """
        return [entry.info for entry in self._agents.values() if entry.info.parent_id == parent_id]


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

    def delegate_task(self, task: str, agent_id: str) -> Result[Any]:
        """Delegate task to agent.

        Args:
            task: Task to delegate.
            agent_id: Target agent ID.

        Returns:
            Result of task delegation.

        Raises:
            ValueError: If agent not found.

        """
        agent = self._registry.get_agent(agent_id)

        # Create task message
        message = create_human_message(task)
        set_message_metadata(message, "receiver_id", agent_id)

        # Send message to agent
        return agent.receive_message(message)

    def broadcast_task(self, task: str, capability: str) -> dict[str, Result[Any]]:
        """Broadcast task to all agents with capability.

        Args:
            task: Task to broadcast.
            capability: Required capability.

        Returns:
            Dictionary of agent IDs to results.

        """
        ...

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

    async def process(self, message: Message) -> Result:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        """
        self.processed_messages.append(message)
        return Result(success=True, data=f"Processed by {self.agent_id}")

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        """
        self.processed_messages.append(message)
        yield f"Processed by {self.agent_id}"

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
