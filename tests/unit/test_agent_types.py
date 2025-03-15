"""Unit tests for agent types."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import HumanMessage

from src.agent.agent_types import (
    Agent,
    MockAgent,
    SimpleAgentCoordinator,
)
from src.common_types import AgentEntry, AgentInfo
from src.common_types.enums import AgentStatus
from src.common_types.result_types import Result
from src.messages.creation import create_human_message

if TYPE_CHECKING:
    from src.config.agent import AgentConfig


def test_agent_info_initialization() -> None:
    """Test AgentInfo initialization."""
    # Test with minimal parameters
    info = AgentInfo(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["test"],
    )
    assert info.agent_id == "test-agent"
    assert info.name == "Test Agent"
    assert info.description == "A test agent"
    assert info.capabilities == ["test"]
    assert info.parent_id is None
    assert info.child_ids == []
    assert info.status == AgentStatus.IDLE.value

    # Test with all parameters
    info = AgentInfo(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["test"],
        parent_id="parent-agent",
        child_ids=["child-agent-1", "child-agent-2"],
        status=AgentStatus.BUSY.value,
    )
    assert info.agent_id == "test-agent"
    assert info.name == "Test Agent"
    assert info.description == "A test agent"
    assert info.capabilities == ["test"]
    assert info.parent_id == "parent-agent"
    assert info.child_ids == ["child-agent-1", "child-agent-2"]
    assert info.status == AgentStatus.BUSY.value


def test_agent_entry_initialization() -> None:
    """Test AgentEntry initialization."""
    agent = MockAgent(agent_id="test-agent", capabilities=["test"])
    info = AgentInfo(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["test"],
    )
    entry = AgentEntry(info=info, agent=agent)
    assert entry.info == info
    assert entry.agent == agent


def test_mock_agent_initialization() -> None:
    """Test MockAgent initialization."""
    agent = MockAgent(agent_id="test-agent", capabilities=["test"])
    assert agent.get_agent_id() == "test-agent"
    assert agent.get_capabilities() == ["test"]
    assert agent.can_handle("test task")


def test_mock_agent_process() -> None:
    """Test MockAgent process method."""
    agent = MockAgent(agent_id="test-agent", capabilities=["test"])
    message = HumanMessage(content="test message")
    result = agent.process(message)
    assert isinstance(result, Result)
    assert result.success
    assert result.data == "Mock result"
    assert message in agent.processed_messages


def test_mock_agent_process_stream() -> None:
    """Test MockAgent process_stream method."""
    agent = MockAgent(agent_id="test-agent", capabilities=["test"])
    message = HumanMessage(content="test message")

    # Test the async generator
    async def test_stream() -> list[str]:
        return [chunk async for chunk in agent.process_stream(message)]

    chunks = asyncio.run(test_stream())
    assert chunks == ["Mock result"]
    assert message in agent.processed_messages


def test_mock_agent_send_message() -> None:
    """Test MockAgent send_message method."""
    agent = MockAgent(agent_id="test-agent", capabilities=["test"])
    message = HumanMessage(content="test message")
    result = agent.send_message(message)
    assert isinstance(result, Result)
    assert result.success
    assert result.data == "Mock result"
    assert message in agent.processed_messages


def test_mock_agent_receive_message() -> None:
    """Test MockAgent receive_message method."""
    agent = MockAgent(agent_id="test-agent", capabilities=["test"])
    message = HumanMessage(content="test message")
    result = agent.receive_message(message)
    assert isinstance(result, Result)
    assert result.success
    assert result.data == "Mock result"
    assert message in agent.processed_messages


def test_simple_agent_coordinator_initialization() -> None:
    """Test SimpleAgentCoordinator initialization."""

    # Create a mock registry
    class MockRegistry:
        def __init__(self) -> None:
            self.agents: dict[str, AgentEntry] = {}

        def register_agent(self, agent: Agent, info: AgentInfo) -> None:
            self.agents[info.agent_id] = AgentEntry(info=info, agent=agent)

        def get_agent(self, agent_id: str) -> Agent:
            return self.agents[agent_id].agent

        def get_agent_info(self, agent_id: str) -> AgentInfo:
            return self.agents[agent_id].info

        def list_agents(self) -> list[AgentInfo]:
            return [entry.info for entry in self.agents.values()]

    registry = MockRegistry()
    coordinator = SimpleAgentCoordinator(registry)
    assert hasattr(coordinator, "registry")
    # Access to private member is acceptable in tests
    assert hasattr(coordinator, "_agent_factories")


def test_simple_agent_coordinator_register_agent_factory() -> None:
    """Test SimpleAgentCoordinator register_agent_factory method."""

    # Create a mock registry
    class MockRegistry:
        def __init__(self) -> None:
            self.agents: dict[str, AgentEntry] = {}

        def register_agent(self, agent: Agent, info: AgentInfo) -> None:
            self.agents[info.agent_id] = AgentEntry(info=info, agent=agent)

        def get_agent(self, agent_id: str) -> Agent:
            return self.agents[agent_id].agent

        def get_agent_info(self, agent_id: str) -> AgentInfo:
            return self.agents[agent_id].info

        def list_agents(self) -> list[AgentInfo]:
            return [entry.info for entry in self.agents.values()]

    registry = MockRegistry()
    coordinator = SimpleAgentCoordinator(registry)

    def factory(config: AgentConfig) -> MockAgent:
        return MockAgent(config)

    coordinator.register_agent_factory("test_type", factory)
    # Access to private member is acceptable in tests
    assert "test_type" in coordinator._agent_factories
    assert coordinator._agent_factories["test_type"] == factory


def test_simple_agent_coordinator_create_agent() -> None:
    """Test SimpleAgentCoordinator create_agent method."""

    # Create a mock registry
    class MockRegistry:
        def __init__(self) -> None:
            self.agents: dict[str, AgentEntry] = {}

        def register_agent(self, agent: Agent, info: AgentInfo) -> None:
            self.agents[info.agent_id] = AgentEntry(info=info, agent=agent)

        def get_agent(self, agent_id: str) -> Agent:
            return self.agents[agent_id].agent

        def get_agent_info(self, agent_id: str) -> AgentInfo:
            return self.agents[agent_id].info

        def list_agents(self) -> list[AgentInfo]:
            return [entry.info for entry in self.agents.values()]

    registry = MockRegistry()
    coordinator = SimpleAgentCoordinator(registry)

    def factory(config: dict[str, Any]) -> Agent:
        return MockAgent(agent_id=config.get("agent_id", "test"), capabilities=["test"])

    coordinator.register_agent_factory("test_type", factory)

    # Test creating an agent
    config = {"agent_id": "new-agent", "capabilities": ["test"]}
    agent = coordinator.create_agent("test_type", config)
    assert isinstance(agent, MockAgent)
    assert agent.get_agent_id() == "new-agent"

    # Test with invalid agent type
    with pytest.raises(ValueError, match="Invalid agent type"):
        coordinator.create_agent("invalid_type", {})


class TestMockAgent:
    """Tests for the MockAgent implementation."""

    def test_initialization(self) -> None:
        """Test agent initialization."""
        agent_id = "test-agent"
        capabilities = ["test", "code"]
        agent = MockAgent(agent_id, capabilities)

        assert agent.get_agent_id() == agent_id
        assert agent.get_capabilities() == capabilities
        assert agent.processed_messages == []

    def test_can_handle(self) -> None:
        """Test can_handle method."""
        agent = MockAgent("test-agent", ["test", "code"])

        # Test with capability in task
        assert agent.can_handle("I need help with some test")
        assert agent.can_handle("Can you write some code for me?")

        # Test with capability not in task
        assert not agent.can_handle("I need help with math")

    def test_process(self) -> None:
        """Test process method."""
        agent = MockAgent("test-agent", ["test"])

        # Test process
        message = create_human_message("Hello")
        result = agent.process(message)
        assert result.success is True
        assert result.data == "Mock result"

        # Test that message was stored
        assert len(agent.processed_messages) == 1
        assert agent.processed_messages[0] == message

    @pytest.mark.asyncio
    async def test_process_stream(self) -> None:
        """Test process_stream method."""
        agent = MockAgent("test-agent", ["test"])
        message = create_human_message("Hello")

        # Test process_stream
        chunks = [chunk async for chunk in agent.process_stream(message)]
        assert chunks == ["Mock result"]

        # Test that message was stored
        assert len(agent.processed_messages) == 1
        assert agent.processed_messages[0] == message

    def test_send_message(self) -> None:
        """Test send_message method."""
        agent = MockAgent("test-agent", ["test"])
        message = create_human_message("Hello")

        # Test send_message
        result = agent.send_message(message)
        assert result.success is True
        assert result.data == "Mock result"

        # Test that message was stored
        assert len(agent.processed_messages) == 1
        assert agent.processed_messages[0] == message

    def test_receive_message(self) -> None:
        """Test receive_message method."""
        agent = MockAgent("test-agent", ["test"])
        message = create_human_message("Hello")

        # Test receive_message
        result = agent.receive_message(message)
        assert result.success is True
        assert result.data == "Mock result"

        # Test that message was stored
        assert len(agent.processed_messages) == 1
        assert agent.processed_messages[0] == message


class TestAgentProtocol:
    """Tests for the Agent protocol implementation."""

    @pytest.mark.asyncio
    async def test_process_stream_implementation(self) -> None:
        """Test the implementation of process_stream in a concrete Agent class."""
        agent = MockAgent("test-agent", ["test"])
        message = create_human_message("Hello")

        # Test process_stream
        chunks = [chunk async for chunk in agent.process_stream(message)]
        assert chunks == ["Mock result"]
