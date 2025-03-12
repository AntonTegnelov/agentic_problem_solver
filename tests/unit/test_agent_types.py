"""Unit tests for agent types."""

import asyncio
from typing import Any

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
    assert info.status == AgentStatus.IDLE.value

    # Test with all parameters
    info = AgentInfo(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["test"],
        parent_id="parent-agent",
        status=AgentStatus.BUSY.value,
    )
    assert info.agent_id == "test-agent"
    assert info.name == "Test Agent"
    assert info.description == "A test agent"
    assert info.capabilities == ["test"]
    assert info.parent_id == "parent-agent"
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
    factory = lambda config: MockAgent(config)  # noqa: E731
    coordinator.register_agent_factory("test_type", factory)
    # Access to private member is acceptable in tests
    assert "test_type" in coordinator._agent_factories  # noqa: SLF001
    assert coordinator._agent_factories["test_type"] == factory  # noqa: SLF001


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
