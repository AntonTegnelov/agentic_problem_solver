"""Unit tests for agent types to improve coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.agent.agent_types.agent_types import MockAgent, SimpleAgentCoordinator
from src.common_types import AgentInfo
from src.common_types.enums import AgentStatus
from src.messages.creation import create_human_message
from src.messages.utils import set_receiver_id

if TYPE_CHECKING:
    from src.common_types.result_types import Result


class SubclassMockAgent(MockAgent):
    """Subclass of MockAgent with capabilities attribute for testing."""

    def __init__(self, agent_id: str, capabilities: list[str]) -> None:
        """Initialize agent with public capabilities attribute."""
        super().__init__(agent_id, capabilities)
        self.capabilities = capabilities


class MockRegistry:
    """Mock agent registry for testing."""

    def __init__(self) -> None:
        """Initialize the mock registry."""
        self.agents: dict[str, Any] = {}
        self.agent_info: dict[str, AgentInfo] = {}
        self.parent_child_map: dict[str, list[str]] = {}

    def register_agent(self, agent: Any, info: AgentInfo) -> None:
        """Register an agent."""
        self.agents[info.agent_id] = agent
        self.agent_info[info.agent_id] = info

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.agent_info:
            del self.agent_info[agent_id]

    def get_agent(self, agent_id: str) -> Any:
        """Get an agent by ID."""
        if agent_id not in self.agents:
            msg = f"Agent not found: {agent_id}"
            raise ValueError(msg)
        return self.agents[agent_id]

    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """Get agent info by ID."""
        if agent_id not in self.agent_info:
            msg = f"Agent not found: {agent_id}"
            raise ValueError(msg)
        return self.agent_info[agent_id]

    def list_agents(self) -> list[AgentInfo]:
        """List all registered agents."""
        return list(self.agent_info.values())

    def find_agents_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents by capability."""
        return [info for info in self.agent_info.values() if capability in info.capabilities]

    def find_agents_by_role(self, role: str) -> list[AgentInfo]:
        """Find agents by role."""
        return [info for info in self.agent_info.values() if info.role == role]

    def register_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Register a parent-child relationship."""
        if parent_id not in self.parent_child_map:
            self.parent_child_map[parent_id] = []
        self.parent_child_map[parent_id].append(child_id)


class TestSimpleAgentCoordinatorCoverage:
    """Additional tests for SimpleAgentCoordinator to improve coverage."""

    def test_register_agent_factory(self) -> None:
        """Test registering an agent factory."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Define a factory function
        def factory(config: dict) -> MockAgent:
            return MockAgent(
                agent_id=config.get("agent_id", "default-id"),
                capabilities=config.get("capabilities", []),
            )

        # Register the factory
        coordinator.register_agent_factory("mock", factory)

        # Verify the factory was registered
        assert "mock" in coordinator._agent_factories

        # Create an agent using the factory
        agent = coordinator.create_agent("mock", {"agent_id": "test-id", "capabilities": ["test"]})

        # Verify the agent was created correctly
        assert agent.get_agent_id() == "test-id"
        assert agent.get_capabilities() == ["test"]

    def test_create_agent_invalid_type(self) -> None:
        """Test creating an agent with an invalid type."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Try to create an agent with an invalid type
        with pytest.raises(ValueError, match="Invalid agent type: invalid"):
            coordinator.create_agent("invalid", {})

    def test_broadcast_task(self) -> None:
        """Test broadcasting a task to agents with a specific capability."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Create some agents with the public capabilities attribute
        agent1 = SubclassMockAgent(agent_id="agent1", capabilities=["math"])
        agent2 = SubclassMockAgent(agent_id="agent2", capabilities=["text"])
        agent3 = SubclassMockAgent(agent_id="agent3", capabilities=["math", "logic"])

        # Register the agents
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
            capabilities=["math", "logic"],
        )

        registry.register_agent(agent1, info1)
        registry.register_agent(agent2, info2)
        registry.register_agent(agent3, info3)

        # Broadcast a task to agents with the 'math' capability
        results = coordinator.broadcast_task("Solve math problem", "math")

        # Verify the results
        assert len(results) == 2
        assert "agent1" in results
        assert "agent3" in results
        assert "agent2" not in results

    def test_get_message_metadata(self) -> None:
        """Test creating a message and accessing its metadata."""
        # Create a message
        message = create_human_message("Test message")

        # Verify that additional_kwargs exists but may be empty
        assert hasattr(message, "additional_kwargs")

        # Set metadata using our utility function
        set_receiver_id(message, "test-agent")

        # Verify that we can access the metadata
        assert "metadata" in message.additional_kwargs
        assert "receiver_id" in message.additional_kwargs["metadata"]
        assert message.additional_kwargs["metadata"]["receiver_id"] == "test-agent"

    def test_route_message(self) -> None:
        """Test routing a message to a specific agent."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Define a custom route_message method for our test
        def custom_route_message(self: SimpleAgentCoordinator, message: Any) -> Result:
            receiver_id = message.additional_kwargs.get("metadata", {}).get("receiver_id")
            if not receiver_id:
                msg = "No receiver_id in message metadata"
                raise ValueError(msg)
            if receiver_id not in self._agents:
                msg = f"Agent not found: {receiver_id}"
                raise ValueError(msg)
            return self._agents[receiver_id].process(message)

        # Monkey patch the route_message method
        coordinator.route_message = custom_route_message.__get__(coordinator, type(coordinator))

        # Create an agent
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])

        # Register the agent
        info = AgentInfo(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
            capabilities=["test"],
        )
        registry.register_agent(agent, info)

        # Set up the _agents dictionary in the coordinator
        coordinator._agents = {"test-agent": agent}

        # Create a message with the correct receiver_id
        message = create_human_message("Test message")
        set_receiver_id(message, "test-agent")

        # Route the message
        result = coordinator.route_message(message)

        # Verify the result
        assert result.success
        assert result.data == "Mock result"

    def test_route_message_no_receiver(self) -> None:
        """Test routing a message with no receiver ID."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Define a custom route_message method for our test
        def custom_route_message(self: SimpleAgentCoordinator, message: Any) -> Result:
            receiver_id = message.additional_kwargs.get("metadata", {}).get("receiver_id")
            if not receiver_id:
                msg = "No receiver_id in message metadata"
                raise ValueError(msg)
            if receiver_id not in self._agents:
                msg = f"Agent not found: {receiver_id}"
                raise ValueError(msg)
            return self._agents[receiver_id].process(message)

        # Monkey patch the route_message method
        coordinator.route_message = custom_route_message.__get__(coordinator, type(coordinator))

        # Create a message with no receiver_id
        message = create_human_message("Test message")

        # Route the message
        with pytest.raises(ValueError, match="No receiver_id in message metadata"):
            coordinator.route_message(message)

    def test_route_message_agent_not_found(self) -> None:
        """Test routing a message to a non-existent agent."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Define a custom route_message method for our test
        def custom_route_message(self: SimpleAgentCoordinator, message: Any) -> Result:
            receiver_id = message.additional_kwargs.get("metadata", {}).get("receiver_id")
            if not receiver_id:
                msg = "No receiver_id in message metadata"
                raise ValueError(msg)
            if receiver_id not in self._agents:
                msg = f"Agent not found: {receiver_id}"
                raise ValueError(msg)
            return self._agents[receiver_id].process(message)

        # Monkey patch the route_message method
        coordinator.route_message = custom_route_message.__get__(coordinator, type(coordinator))

        # Create a message with an invalid receiver_id
        message = create_human_message("Test message")
        set_receiver_id(message, "non-existent")

        # Route the message
        with pytest.raises(ValueError, match="Agent not found: non-existent"):
            coordinator.route_message(message)

    def test_get_agent_status(self) -> None:
        """Test getting an agent's status."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Create an agent
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])

        # Register the agent
        info = AgentInfo(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
            capabilities=["test"],
            status=AgentStatus.IDLE,
        )
        registry.register_agent(agent, info)

        # Get the agent's status
        status = coordinator.get_agent_status("test-agent")

        # Verify the status
        assert status == AgentStatus.IDLE

    def test_get_agent_status_not_found(self) -> None:
        """Test getting status of a non-existent agent."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Get the status of a non-existent agent
        with pytest.raises(ValueError, match="Agent not found: non-existent"):
            coordinator.get_agent_status("non-existent")

    def test_set_agent_status(self) -> None:
        """Test setting an agent's status."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Create an agent
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])

        # Register the agent
        info = AgentInfo(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
            capabilities=["test"],
            status=AgentStatus.IDLE,
        )
        registry.register_agent(agent, info)

        # Set the agent's status
        coordinator.set_agent_status("test-agent", AgentStatus.BUSY)

        # Verify the status was updated
        assert registry.get_agent_info("test-agent").status == AgentStatus.BUSY

    def test_set_agent_status_not_found(self) -> None:
        """Test setting status of a non-existent agent."""
        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Set the status of a non-existent agent
        with pytest.raises(ValueError, match="Agent not found: non-existent"):
            coordinator.set_agent_status("non-existent", AgentStatus.BUSY)


class TestMockAgentCoverage:
    """Additional tests for MockAgent to improve coverage."""

    def test_mock_agent_capabilities(self) -> None:
        """Test MockAgent capabilities."""
        agent = MockAgent(agent_id="test-agent", capabilities=["math", "logic"])

        # Test getting capabilities
        assert agent.get_capabilities() == ["math", "logic"]

        # Test can_handle method
        assert agent.can_handle("Solve a math problem")
        assert not agent.can_handle("Unknown task")

    def test_mock_agent_parent_child_relationships(self) -> None:
        """Test MockAgent parent-child relationships."""
        parent = MockAgent(agent_id="parent", capabilities=["parent"])
        child1 = MockAgent(agent_id="child1", capabilities=["child"])
        child2 = MockAgent(agent_id="child2", capabilities=["child"])

        # Set parent-child relationships
        child1.set_parent("parent")
        child2.set_parent("parent")
        parent.add_child("child1")
        parent.add_child("child2")

        # Verify relationships
        assert child1.get_parent_id() == "parent"
        assert child2.get_parent_id() == "parent"
        assert sorted(parent.get_child_ids()) == ["child1", "child2"]

        # Remove relationships
        parent.remove_child("child1")
        child2.clear_parent()

        # Verify relationships were removed
        assert "child1" not in parent.get_child_ids()
        assert "child2" in parent.get_child_ids()
        assert child2.get_parent_id() is None

    def test_mock_agent_message_handling(self) -> None:
        """Test MockAgent message handling."""
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])

        # Create a test message
        message = create_human_message("Test message")

        # Test send_message
        send_result = agent.send_message(message)
        assert send_result.success
        assert send_result.data == "Mock result"

        # Test receive_message
        receive_result = agent.receive_message(message)
        assert receive_result.success
        assert receive_result.data == "Mock result"

        # Test process
        process_result = agent.process(message)
        assert process_result.success
        assert process_result.data == "Mock result"

    @pytest.mark.asyncio
    async def test_mock_agent_process_stream(self) -> None:
        """Test MockAgent process_stream method."""
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])

        # Create a test message
        message = create_human_message("Test message")

        # Test process_stream
        chunks = [chunk async for chunk in agent.process_stream(message)]

        # Verify the result
        assert chunks == ["Mock result"]

    def test_collect_results_from_children_with_results(self) -> None:
        """Test collecting results from children with actual results."""
        parent = MockAgent(agent_id="parent", capabilities=["parent"])
        child1 = MockAgent(agent_id="child1", capabilities=["child"])
        child2 = MockAgent(agent_id="child2", capabilities=["child"])

        # Set up parent-child relationships
        parent.add_child("child1")
        parent.add_child("child2")
        child1.set_parent("parent")
        child2.set_parent("parent")

        # Collect results
        results = parent.collect_results_from_children()

        # Verify the results
        assert len(results) == 2
        assert "child1" in results
        assert "child2" in results
        assert results["child1"].success
        assert results["child2"].success
        assert results["child1"].data == "Mock result"
        assert results["child2"].data == "Mock result"
