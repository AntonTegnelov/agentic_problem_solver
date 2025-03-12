"""Unit tests for agent hierarchy functionality."""

from unittest.mock import MagicMock

import pytest

from src.agent.agent_types import (
    create_architect_agent,
    create_executor_agent,
    create_planner_agent,
)
from src.agent.coordination import InMemoryAgentRegistry


@pytest.fixture
def registry() -> InMemoryAgentRegistry:
    """Create an InMemoryAgentRegistry instance."""
    return InMemoryAgentRegistry()


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate.return_value = "Mock response"
    return provider


@pytest.fixture
def agent_hierarchy(registry: InMemoryAgentRegistry, mock_provider: MagicMock) -> dict[str, str]:
    """Create a hierarchy of agents for testing.

    Structure:
    architect_agent
    ├── planner_agent1
    │   ├── executor_agent1
    │   └── executor_agent2
    └── planner_agent2
        └── executor_agent3

    Returns:
        Dictionary mapping agent roles to agent IDs.

    """
    # Create agents
    architect = create_architect_agent(provider=mock_provider)
    planner1 = create_planner_agent(provider=mock_provider)
    planner2 = create_planner_agent(provider=mock_provider)
    executor1 = create_executor_agent(provider=mock_provider)
    executor2 = create_executor_agent(provider=mock_provider)
    executor3 = create_executor_agent(provider=mock_provider)

    # Register agents
    registry.register_agent(architect)
    registry.register_agent(planner1)
    registry.register_agent(planner2)
    registry.register_agent(executor1)
    registry.register_agent(executor2)
    registry.register_agent(executor3)

    # Set up hierarchy
    registry.register_parent_child_relationship(architect.get_agent_id(), planner1.get_agent_id())
    registry.register_parent_child_relationship(architect.get_agent_id(), planner2.get_agent_id())
    registry.register_parent_child_relationship(planner1.get_agent_id(), executor1.get_agent_id())
    registry.register_parent_child_relationship(planner1.get_agent_id(), executor2.get_agent_id())
    registry.register_parent_child_relationship(planner2.get_agent_id(), executor3.get_agent_id())

    return {
        "architect": architect.get_agent_id(),
        "planner1": planner1.get_agent_id(),
        "planner2": planner2.get_agent_id(),
        "executor1": executor1.get_agent_id(),
        "executor2": executor2.get_agent_id(),
        "executor3": executor3.get_agent_id(),
    }


class TestAgentHierarchy:
    """Tests for agent hierarchy functionality."""

    def test_register_parent_child_relationship(
        self,
        registry: InMemoryAgentRegistry,
        mock_provider: MagicMock,
    ) -> None:
        """Test register_parent_child_relationship method."""
        # Create agents
        parent = create_architect_agent(provider=mock_provider)
        child = create_planner_agent(provider=mock_provider)

        # Register agents
        registry.register_agent(parent)
        registry.register_agent(child)

        # Register parent-child relationship
        registry.register_parent_child_relationship(parent.get_agent_id(), child.get_agent_id())

        # Verify relationship
        assert child.get_parent_id() == parent.get_agent_id()
        assert child.get_agent_id() in parent.get_child_ids()

    def test_remove_parent_child_relationship(
        self,
        registry: InMemoryAgentRegistry,
        mock_provider: MagicMock,
    ) -> None:
        """Test remove_parent_child_relationship method."""
        # Create agents
        parent = create_architect_agent(provider=mock_provider)
        child = create_planner_agent(provider=mock_provider)

        # Register agents
        registry.register_agent(parent)
        registry.register_agent(child)

        # Register parent-child relationship
        registry.register_parent_child_relationship(parent.get_agent_id(), child.get_agent_id())

        # Verify relationship
        assert child.get_parent_id() == parent.get_agent_id()
        assert child.get_agent_id() in parent.get_child_ids()

        # Remove relationship
        registry.remove_parent_child_relationship(parent.get_agent_id(), child.get_agent_id())

        # Verify relationship is removed
        assert child.get_parent_id() is None
        assert child.get_agent_id() not in parent.get_child_ids()

    def test_get_parent_agent(
        self,
        registry: InMemoryAgentRegistry,
        agent_hierarchy: dict[str, str],
    ) -> None:
        """Test get_parent_agent method."""
        # Get parent of planner1
        parent = registry.get_parent_agent(agent_hierarchy["planner1"])
        assert parent is not None
        assert parent.get_agent_id() == agent_hierarchy["architect"]

        # Get parent of executor1
        parent = registry.get_parent_agent(agent_hierarchy["executor1"])
        assert parent is not None
        assert parent.get_agent_id() == agent_hierarchy["planner1"]

        # Get parent of architect (should be None)
        parent = registry.get_parent_agent(agent_hierarchy["architect"])
        assert parent is None

    def test_get_child_agents(
        self,
        registry: InMemoryAgentRegistry,
        agent_hierarchy: dict[str, str],
    ) -> None:
        """Test get_child_agents method."""
        # Get children of architect
        children = registry.get_child_agents(agent_hierarchy["architect"])
        assert len(children) == 2
        child_ids = [child.get_agent_id() for child in children]
        assert agent_hierarchy["planner1"] in child_ids
        assert agent_hierarchy["planner2"] in child_ids

        # Get children of planner1
        children = registry.get_child_agents(agent_hierarchy["planner1"])
        assert len(children) == 2
        child_ids = [child.get_agent_id() for child in children]
        assert agent_hierarchy["executor1"] in child_ids
        assert agent_hierarchy["executor2"] in child_ids

        # Get children of executor1 (should be empty)
        children = registry.get_child_agents(agent_hierarchy["executor1"])
        assert len(children) == 0

    def test_get_sibling_agents(
        self,
        registry: InMemoryAgentRegistry,
        agent_hierarchy: dict[str, str],
    ) -> None:
        """Test get_sibling_agents method."""
        # Get siblings of planner1
        siblings = registry.get_sibling_agents(agent_hierarchy["planner1"])
        assert len(siblings) == 1
        assert siblings[0].get_agent_id() == agent_hierarchy["planner2"]

        # Get siblings of executor1
        siblings = registry.get_sibling_agents(agent_hierarchy["executor1"])
        assert len(siblings) == 1
        assert siblings[0].get_agent_id() == agent_hierarchy["executor2"]

        # Get siblings of architect (should be empty)
        siblings = registry.get_sibling_agents(agent_hierarchy["architect"])
        assert len(siblings) == 0

    def test_get_agent_hierarchy(
        self,
        registry: InMemoryAgentRegistry,
        agent_hierarchy: dict[str, str],
    ) -> None:
        """Test get_agent_hierarchy method."""
        # Get hierarchy starting from architect
        hierarchy = registry.get_agent_hierarchy(agent_hierarchy["architect"])

        # Verify hierarchy structure
        assert len(hierarchy) == 6  # All agents should be in the hierarchy
        assert agent_hierarchy["architect"] in hierarchy
        assert agent_hierarchy["planner1"] in hierarchy
        assert agent_hierarchy["planner2"] in hierarchy
        assert agent_hierarchy["executor1"] in hierarchy
        assert agent_hierarchy["executor2"] in hierarchy
        assert agent_hierarchy["executor3"] in hierarchy

        # Verify parent-child relationships
        assert len(hierarchy[agent_hierarchy["architect"]]) == 2
        assert agent_hierarchy["planner1"] in hierarchy[agent_hierarchy["architect"]]
        assert agent_hierarchy["planner2"] in hierarchy[agent_hierarchy["architect"]]

        assert len(hierarchy[agent_hierarchy["planner1"]]) == 2
        assert agent_hierarchy["executor1"] in hierarchy[agent_hierarchy["planner1"]]
        assert agent_hierarchy["executor2"] in hierarchy[agent_hierarchy["planner1"]]

        assert len(hierarchy[agent_hierarchy["planner2"]]) == 1
        assert agent_hierarchy["executor3"] in hierarchy[agent_hierarchy["planner2"]]

        # Leaf nodes should have empty lists
        assert len(hierarchy[agent_hierarchy["executor1"]]) == 0
        assert len(hierarchy[agent_hierarchy["executor2"]]) == 0
        assert len(hierarchy[agent_hierarchy["executor3"]]) == 0

    def test_unregister_agent_updates_relationships(
        self,
        registry: InMemoryAgentRegistry,
        agent_hierarchy: dict[str, str],
    ) -> None:
        """Test that unregistering an agent updates parent-child relationships."""
        # Unregister planner1
        registry.unregister_agent(agent_hierarchy["planner1"])

        # Verify that architect no longer has planner1 as a child
        architect = registry.get_agent(agent_hierarchy["architect"])
        assert agent_hierarchy["planner1"] not in architect.get_child_ids()

        # Verify that executor1 and executor2 no longer have a parent
        executor1 = registry.get_agent(agent_hierarchy["executor1"])
        executor2 = registry.get_agent(agent_hierarchy["executor2"])
        assert executor1.get_parent_id() is None
        assert executor2.get_parent_id() is None
