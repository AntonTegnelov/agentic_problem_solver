"""Extended unit tests for agent types."""

from typing import Any

import pytest

from src.agent.agent_types import (
    Agent,
    MockAgent,
    SimpleAgentCoordinator,
)
from src.common_types import AgentInfo
from src.common_types.result_types import Result
from src.messages.creation import create_human_message


class TestMockAgent:
    """Test cases for MockAgent."""

    def test_mock_agent_extended_capabilities(self) -> None:
        """Test MockAgent capabilities handling."""
        agent = MockAgent(agent_id="test-agent", capabilities=["test", "search"])
        assert agent.get_capabilities() == ["test", "search"]
        assert agent.can_handle("test task")
        assert agent.can_handle("search task")

    def test_mock_agent_multiple_messages(self) -> None:
        """Test MockAgent handling multiple messages."""
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])

        # Process multiple messages
        messages = [
            create_human_message("message 1"),
            create_human_message("message 2"),
            create_human_message("message 3"),
        ]

        results = [agent.process(msg) for msg in messages]

        # Check all results are successful
        for result in results:
            assert isinstance(result, Result)
            assert result.success
            assert result.data == "Mock result"

        # Check all messages were processed
        for msg in messages:
            assert msg in agent.processed_messages

    @pytest.mark.asyncio
    async def test_mock_agent_process_stream_multiple(self) -> None:
        """Test MockAgent process_stream with multiple messages."""
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])

        # Process multiple messages
        messages = [
            create_human_message("message 1"),
            create_human_message("message 2"),
        ]

        # Test the async generator for each message
        for msg in messages:
            chunks = [chunk async for chunk in agent.process_stream(msg)]
            assert chunks == ["Mock result"]
            assert msg in agent.processed_messages


class TestSimpleAgentCoordinatorExtended:
    """Extended test cases for SimpleAgentCoordinator."""

    def test_create_agent_with_config(self) -> None:
        """Test creating an agent with configuration."""

        # Create a mock registry
        class MockRegistry:
            def __init__(self) -> None:
                self.agents: dict[str, Agent] = {}
                self.agent_info: dict[str, AgentInfo] = {}

            def register_agent(self, agent: Agent, info: AgentInfo) -> None:
                self.agents[info.agent_id] = agent
                self.agent_info[info.agent_id] = info

            def get_agent(self, agent_id: str) -> Agent:
                return self.agents[agent_id]

            def get_agent_info(self, agent_id: str) -> AgentInfo:
                return self.agent_info[agent_id]

            def list_agents(self) -> list[AgentInfo]:
                return list(self.agent_info.values())

        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Define a factory function that uses the config
        def factory(config: dict[str, Any]) -> Agent:
            agent_id = config.get("agent_id", "default-id")
            capabilities = config.get("capabilities", [])
            return MockAgent(agent_id=agent_id, capabilities=capabilities)

        # Register the factory
        coordinator.register_agent_factory("custom_agent", factory)

        # Create an agent with custom config
        config = {
            "agent_id": "custom-agent-1",
            "capabilities": ["search", "analyze"],
            "name": "Custom Agent",
            "description": "A custom agent for testing",
        }

        agent = coordinator.create_agent("custom_agent", config)

        assert isinstance(agent, MockAgent)
        assert agent.get_agent_id() == "custom-agent-1"
        assert agent.get_capabilities() == ["search", "analyze"]

    def test_create_agent_with_invalid_type(self) -> None:
        """Test creating an agent with an invalid type."""

        # Create a mock registry
        class MockRegistry:
            def __init__(self) -> None:
                self.agents = {}
                self.agent_info = {}

            def register_agent(self, agent: Agent, info: AgentInfo) -> None:
                self.agents[info.agent_id] = agent
                self.agent_info[info.agent_id] = info

            def list_agents(self) -> list[AgentInfo]:
                return list(self.agent_info.values())

        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Try to create an agent with an invalid type
        with pytest.raises(ValueError, match="Invalid agent type"):
            coordinator.create_agent("invalid_type", {})

    def test_delegate_task_extended(self) -> None:
        """Test delegating tasks with different configurations."""

        # Create a mock registry
        class MockRegistry:
            def __init__(self) -> None:
                self.agents = {}
                self.agent_info = {}

            def register_agent(self, agent: Agent, info: AgentInfo) -> None:
                self.agents[info.agent_id] = agent
                self.agent_info[info.agent_id] = info

            def get_agent(self, agent_id: str) -> Agent:
                if agent_id not in self.agents:
                    msg = f"Agent not found: {agent_id}"
                    raise ValueError(msg)
                return self.agents[agent_id]

            def get_agent_info(self, agent_id: str) -> AgentInfo:
                if agent_id not in self.agent_info:
                    msg = f"Agent info not found: {agent_id}"
                    raise ValueError(msg)
                return self.agent_info[agent_id]

            def list_agents(self) -> list[AgentInfo]:
                return list(self.agent_info.values())

        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Register multiple agents
        agents = []
        for i in range(3):
            agent = MockAgent(agent_id=f"agent-{i}", capabilities=[f"capability-{i}"])
            info = AgentInfo(
                agent_id=f"agent-{i}",
                name=f"Agent {i}",
                description=f"Test agent {i}",
                capabilities=[f"capability-{i}"],
            )
            registry.register_agent(agent, info)
            agents.append(agent)

        # Delegate tasks to different agents
        for i, agent in enumerate(agents):
            task = f"Task for agent {i}"
            result = coordinator.delegate_task(task, f"agent-{i}")

            assert isinstance(result, Result)
            assert result.success
            assert result.data == "Mock result"

            # Verify the task was processed by the correct agent
            assert any(task in msg.content for msg in agent.processed_messages)

    def test_update_agents(self) -> None:
        """Test the _update_agents method."""

        # Create a mock registry
        class MockRegistry:
            def __init__(self) -> None:
                self.agents = {}
                self.agent_info = {}

            def register_agent(self, agent: Agent, info: AgentInfo) -> None:
                self.agents[info.agent_id] = agent
                self.agent_info[info.agent_id] = info

            def get_agent(self, agent_id: str) -> Agent:
                return self.agents[agent_id]

            def list_agents(self) -> list[AgentInfo]:
                return list(self.agent_info.values())

        registry = MockRegistry()
        coordinator = SimpleAgentCoordinator(registry)

        # Initially, no agents
        coordinator._update_agents()  # noqa: SLF001
        assert not hasattr(coordinator, "_agents") or not coordinator._agents  # noqa: SLF001

        # Register an agent
        agent = MockAgent(agent_id="test-agent", capabilities=["test"])
        info = AgentInfo(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent",
            capabilities=["test"],
        )
        registry.register_agent(agent, info)

        # Update agents and check
        coordinator._update_agents()  # noqa: SLF001
        assert hasattr(coordinator, "_agents")
        assert "test-agent" in coordinator._agents  # noqa: SLF001
        assert coordinator._agents["test-agent"] == agent  # noqa: SLF001
