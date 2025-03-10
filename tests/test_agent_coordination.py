"""Test agent coordination functionality."""

import pytest

from src.agent.agent_types.agent_types import (
    Agent,
    AgentInfo,
    AgentNotFoundError,
    InMemoryAgentRegistry,
    Message,
    Result,
    SimpleAgentCoordinator,
)
from src.agent.coordination import AgentCoordinator, AgentRegistry


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        agent_id: str,
        capabilities: list[str],
        should_fail: bool = False,
    ) -> None:
        """Initialize mock agent.

        Args:
            agent_id: Agent ID.
            capabilities: List of agent capabilities.
            should_fail: Whether agent should fail processing.

        """
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.should_fail = should_fail
        self.processed_messages: list[Message] = []

    def process(self, message: Message) -> Result[str]:
        """Process message.

        Args:
            message: Message to process.

        Returns:
            Result of processing.

        Raises:
            TestProcessingError: If should_fail is True.

        """
        if self.should_fail:
            msg = "Processing failed"
            raise TestProcessingError(msg)
        self.processed_messages.append(message)
        return Result(success=True, data=f"Processed by {self.agent_id}", error="")

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
        return self.capabilities

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        return any(capability in task.lower() for capability in self.capabilities)

    def send_message(self, message: Message) -> Result[str]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def receive_message(self, message: Message) -> Result[str]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        return self.process(message)


class TestProcessingError(Exception):
    """Error raised when test processing fails."""


class TestAgent(Agent):
    """Test agent implementation."""

    def __init__(self, agent_id: str, should_fail: bool = False) -> None:
        """Initialize test agent.

        Args:
            agent_id: Agent ID
            should_fail: Whether the agent should fail processing

        """
        self.agent_id = agent_id
        self.should_fail = should_fail
        self.processed_messages = []

    def process(self, message: Message) -> Result:
        """Process a message.

        Args:
            message: Message to process

        Returns:
            Processing result

        Raises:
            TestProcessingError: If processing should fail

        """
        if self.should_fail:
            msg = "Processing failed"
            raise TestProcessingError(msg)
        self.processed_messages.append(message)
        return Result(success=True, data=f"Processed by {self.agent_id}", error="")


def test_agent_registry() -> None:
    """Test agent registry functionality."""
    registry = InMemoryAgentRegistry()

    # Test registering agents
    agent1 = MockAgent("agent1", ["math", "logic"])
    agent2 = MockAgent("agent2", ["text", "nlp"])

    info1 = AgentInfo(
        agent_id="agent1",
        name="Math Agent",
        description="Handles math tasks",
        capabilities=["math", "logic"],
    )

    info2 = AgentInfo(
        agent_id="agent2",
        name="Text Agent",
        description="Handles text tasks",
        capabilities=["text", "nlp"],
        parent_id="agent1",
    )

    registry.register_agent(agent1, info1)
    registry.register_agent(agent2, info2)

    # Test getting agent
    retrieved_agent = registry.get_agent("agent1")
    assert retrieved_agent.get_agent_id() == "agent1"

    # Test getting agent info
    retrieved_info = registry.get_agent_info("agent2")
    assert retrieved_info.name == "Text Agent"
    assert retrieved_info.parent_id == "agent1"

    # Test listing agents
    agents = registry.list_agents()
    assert len(agents) == 2
    assert any(agent.agent_id == "agent1" for agent in agents)
    assert any(agent.agent_id == "agent2" for agent in agents)

    # Test finding agents by capability
    math_agents = registry.find_agents_by_capability("math")
    assert len(math_agents) == 1
    assert math_agents[0].agent_id == "agent1"

    # Test finding agents by parent
    child_agents = registry.find_agents_by_parent("agent1")
    assert len(child_agents) == 1
    assert child_agents[0].agent_id == "agent2"

    # Test unregistering agent
    registry.unregister_agent("agent1")
    with pytest.raises(ValueError, match="Agent not found: agent1"):
        registry.get_agent("agent1")


def test_agent_coordinator() -> None:
    """Test agent coordinator functionality."""
    registry = InMemoryAgentRegistry()
    coordinator = SimpleAgentCoordinator(registry)

    # Register agents
    agent1 = MockAgent("agent1", ["math"])
    agent2 = MockAgent("agent2", ["text"])
    agent3 = MockAgent("agent3", ["logic"], should_fail=True)

    info1 = AgentInfo(
        agent_id="agent1",
        name="Math Agent",
        description="Handles math tasks",
        capabilities=["math"],
    )

    info2 = AgentInfo(
        agent_id="agent2",
        name="Text Agent",
        description="Handles text tasks",
        capabilities=["text"],
    )

    info3 = AgentInfo(
        agent_id="agent3",
        name="Logic Agent",
        description="Handles logic tasks",
        capabilities=["logic"],
        status="busy",
    )

    registry.register_agent(agent1, info1)
    registry.register_agent(agent2, info2)
    registry.register_agent(agent3, info3)

    # Test delegating task
    result = coordinator.delegate_task("Solve math problem", "agent1")
    assert result.success
    assert result.data == "Processed by agent1"
    assert len(agent1.processed_messages) == 1
    assert agent1.processed_messages[0].content == "Solve math problem"

    # Test broadcasting task
    results = coordinator.broadcast_task("Process text", "text")
    assert len(results) == 1
    assert "agent2" in results
    assert results["agent2"].success
    assert results["agent2"].data == "Processed by agent2"

    # Test routing message
    message = Message(
        role="user",
        content="Test message",
        sender_id="user",
        receiver_id="agent2",
    )
    result = coordinator.route_message(message)
    assert result.success
    assert result.data == "Processed by agent2"
    assert message in agent2.processed_messages

    # Test getting agent status
    status = coordinator.get_agent_status("agent3")
    assert status == "busy"

    # Test setting agent status
    coordinator.set_agent_status("agent3", "idle")
    assert registry.get_agent_info("agent3").status == "idle"


def test_agent_factory() -> None:
    """Test agent factory functionality."""
    registry = InMemoryAgentRegistry()
    coordinator = SimpleAgentCoordinator(registry)

    # Register agent factories
    def create_math_agent(config: dict) -> Agent:
        """Create math agent."""
        return MockAgent(
            agent_id=config.get("agent_id", "math_agent"),
            capabilities=["math"],
        )

    def create_text_agent(config: dict) -> Agent:
        """Create text agent."""
        return MockAgent(
            agent_id=config.get("agent_id", "text_agent"),
            capabilities=["text"],
        )

    coordinator.register_agent_factory("math", create_math_agent)
    coordinator.register_agent_factory("text", create_text_agent)

    # Test creating agents
    math_agent = coordinator.create_agent("math", {"agent_id": "custom_math"})
    assert math_agent.get_agent_id() == "custom_math"
    assert "math" in math_agent.get_capabilities()

    text_agent = coordinator.create_agent("text", {})
    assert text_agent.get_agent_id() == "text_agent"
    assert "text" in text_agent.get_capabilities()

    # Test invalid agent type
    with pytest.raises(ValueError, match="Invalid agent type: invalid"):
        coordinator.create_agent("invalid", {})


def test_agent_communication() -> None:
    """Test agent communication."""
    registry = InMemoryAgentRegistry()

    # Create agents
    agent1 = MockAgent("agent1", ["math"])
    agent2 = MockAgent("agent2", ["text"])

    info1 = AgentInfo(
        agent_id="agent1",
        name="Math Agent",
        description="Handles math tasks",
        capabilities=["math"],
    )

    info2 = AgentInfo(
        agent_id="agent2",
        name="Text Agent",
        description="Handles text tasks",
        capabilities=["text"],
    )

    registry.register_agent(agent1, info1)
    registry.register_agent(agent2, info2)

    # Test direct communication
    message = Message(
        role="user",
        content="Process this",
        sender_id="agent1",
        receiver_id="agent2",
    )

    result = agent2.receive_message(message)
    assert result.success
    assert result.data == "Processed by agent2"
    assert message in agent2.processed_messages

    # Test communication through coordinator
    coordinator = SimpleAgentCoordinator(registry)

    result = coordinator.delegate_task("Math task", "agent1")
    assert result.success
    assert result.data == "Processed by agent1"

    # Test message routing
    message = Message(
        role="system",
        content="Route this",
        sender_id="agent1",
        receiver_id="agent2",
    )

    result = coordinator.route_message(message)
    assert result.success
    assert result.data == "Processed by agent2"
    assert message in agent2.processed_messages


def test_agent_registry_new() -> None:
    """Test agent registry."""
    registry = AgentRegistry()

    # Test registering agent
    agent1 = TestAgent("agent1")
    registry.register_agent("agent1", agent1)
    assert "agent1" in registry.list_agents()
    assert registry.get_agent("agent1") == agent1

    # Test unregistering agent
    registry.unregister_agent("agent1")
    with pytest.raises(AgentNotFoundError, match="Agent not found: agent1"):
        registry.get_agent("agent1")


def test_agent_coordinator_new() -> None:
    """Test agent coordinator."""
    coordinator = AgentCoordinator()

    # Test creating agent
    agent1 = coordinator.create_agent("test", {"agent_id": "agent1"})
    assert isinstance(agent1, TestAgent)
    assert agent1.agent_id == "agent1"

    # Test invalid agent type
    with pytest.raises(AgentNotFoundError, match="Agent type not found: invalid"):
        coordinator.create_agent("invalid", {})
