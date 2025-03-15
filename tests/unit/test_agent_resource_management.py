"""Unit tests for agent resource management."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.coordination import AgentCoordinator, InMemoryAgentRegistry
from src.common_types.enums import AgentRole


def test_resource_limits_max_agents() -> None:
    """Test that agent creation is limited by max_agents."""
    registry = InMemoryAgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Register a factory for test_agent
    coordinator.register_agent_factory("test_agent", lambda **_: MagicMock())

    # Mock the registry to return a large number of agents and test agent creation
    with (
        patch.object(registry, "get_agents", return_value={f"agent_{i}": MagicMock() for i in range(50)}),
        pytest.raises(ValueError, match="Maximum number of agents"),
    ):
        coordinator.create_agent("test_agent", {})


def test_resource_limits_max_agents_per_role() -> None:
    """Test that agent creation is limited by max_agents_per_role."""
    registry = InMemoryAgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Mock find_agents_by_role to return max number of agents for a specific role
    role = AgentRole.ARCHITECT.value
    max_role_agents = coordinator._resource_limits["max_agents_per_role"][role]

    with (
        patch.object(registry, "find_agents_by_role", return_value=[MagicMock() for _ in range(max_role_agents)]),
        pytest.raises(ValueError, match=f"Maximum number of {role} agents"),
    ):
        # Try to create a new agent with that role
        coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})


def test_resource_limits_max_children_per_agent() -> None:
    """Test that agent creation is limited by max_children_per_agent."""
    registry = InMemoryAgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Register a factory for test_agent
    coordinator.register_agent_factory("test_agent", lambda **_: MagicMock())

    # Create a parent agent
    parent_id = "parent_agent"
    parent_agent = MagicMock()
    parent_agent.get_agent_id.return_value = parent_id
    parent_agent.get_child_ids.return_value = [
        f"child_{i}" for i in range(coordinator._resource_limits["max_children_per_agent"])
    ]

    # Register the parent agent
    registry.register_agent(parent_agent)

    # Try to create a new child agent
    with pytest.raises(ValueError, match="Maximum number of children"):
        coordinator.create_agent("test_agent", {"parent_id": parent_id})


def test_resource_limits_max_hierarchy_depth() -> None:
    """Test that agent creation is limited by max_hierarchy_depth."""
    registry = InMemoryAgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Register a factory for test_agent
    coordinator.register_agent_factory("test_agent", lambda **_: MagicMock())

    # Create a parent agent
    parent_id = "deep_parent"
    parent_agent = MagicMock()
    parent_agent.get_agent_id.return_value = parent_id
    parent_agent.get_child_ids.return_value = []

    # Register the parent agent
    registry.register_agent(parent_agent)

    # Create a chain of agents at maximum depth
    max_depth = coordinator._resource_limits["max_hierarchy_depth"]

    # Mock get_ancestors to return a list of ancestors that would make the hierarchy too deep
    ancestors = [MagicMock() for _ in range(max_depth - 1)]

    with (
        patch.object(registry, "get_ancestors", return_value=ancestors),
        pytest.raises(ValueError, match="Maximum hierarchy depth"),
    ):
        # Try to create a new agent that would exceed the max depth
        coordinator.create_agent("test_agent", {"parent_id": parent_id})


def test_resource_limits_not_exceeded() -> None:
    """Test that agent creation succeeds when resource limits are not exceeded."""
    registry = InMemoryAgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Register a factory for test_agent
    coordinator.register_agent_factory("test_agent", lambda **_: MagicMock())

    # Mock methods to return values below the limits
    with (
        patch.object(registry, "get_agents", return_value={}),
        patch.object(registry, "find_agents_by_role", return_value=[]),
        patch.object(registry, "get_ancestors", return_value=[]),
    ):
        # Create an agent - should succeed
        agent = coordinator.create_agent("test_agent", {})
        assert agent is not None


def test_resource_limits_with_role_creation() -> None:
    """Test resource limits with role-based agent creation."""
    registry = InMemoryAgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Mock the create_agent function that would be imported and test agent creation
    with (
        patch("src.agent.agent_types.create_agent", return_value=MagicMock()),
        patch.object(registry, "get_agents", return_value={}),
        patch.object(registry, "find_agents_by_role", return_value=[]),
        patch.object(registry, "get_ancestors", return_value=[]),
    ):
        # Create an agent by role - should succeed
        agent = coordinator.create_agent_by_role(AgentRole.EXECUTOR, {})
        assert agent is not None
