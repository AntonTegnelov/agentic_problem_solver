"""Tests for agent creation functionality."""

import pytest

from src.agent.coordination import AgentCoordinator, InMemoryAgentRegistry
from src.common_types.enums import AgentRole


class TestAgentCreation:
    """Test agent creation functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.registry = InMemoryAgentRegistry()
        self.coordinator = AgentCoordinator(self.registry)

    def test_create_agent_by_role_architect(self) -> None:
        """Test creating an architect agent by role."""
        agent = self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        assert agent is not None
        assert "architect" in agent.get_agent_id().lower()
        assert "architecture" in agent.get_capabilities()
        assert "design" in agent.get_capabilities()

    def test_create_agent_by_role_planner(self) -> None:
        """Test creating a planner agent by role."""
        agent = self.coordinator.create_agent_by_role(AgentRole.PLANNER, {})
        assert agent is not None
        assert "planner" in agent.get_agent_id().lower()
        assert "planning" in agent.get_capabilities()
        assert "task-breakdown" in agent.get_capabilities()

    def test_create_agent_by_role_executor(self) -> None:
        """Test creating an executor agent by role."""
        agent = self.coordinator.create_agent_by_role(AgentRole.EXECUTOR, {})
        assert agent is not None
        assert "executor" in agent.get_agent_id().lower()
        assert "execution" in agent.get_capabilities()
        assert "implementation" in agent.get_capabilities()

    def test_create_agent_by_role_with_parent_id(self) -> None:
        """Test creating an agent with a parent ID."""
        parent = self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        parent_id = parent.get_agent_id()

        child = self.coordinator.create_agent_by_role(
            AgentRole.PLANNER,
            {"parent_id": parent_id},
        )
        assert child.get_parent_id() == parent_id

        self.registry.register_parent_child_relationship(parent_id, child.get_agent_id())

        assert child.get_agent_id() in parent.get_child_ids()

    def test_create_agent_by_role_with_config(self) -> None:
        """Test creating an agent with custom configuration."""
        custom_id = "custom_architect_123"
        agent = self.coordinator.create_agent_by_role(
            AgentRole.ARCHITECT,
            {"agent_id": custom_id},
        )

        assert "architect" in agent.get_agent_id().lower()

    def test_create_agent_with_role_string(self) -> None:
        """Test creating an agent using a role string in the create_agent method."""
        agent = self.coordinator.create_agent("architect", {})
        assert agent is not None
        assert "architect" in agent.get_agent_id().lower()
        assert "architecture" in agent.get_capabilities()
        assert "design" in agent.get_capabilities()

    def test_create_agent_with_invalid_role_string(self) -> None:
        """Test creating an agent with an invalid role string."""
        with pytest.raises(ValueError, match="Invalid agent type"):
            self.coordinator.create_agent("invalid_role", {})
