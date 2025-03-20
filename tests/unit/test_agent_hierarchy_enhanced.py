"""Unit tests for enhanced agent hierarchy functionality."""

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


class TestEnhancedAgentHierarchy:
    """Tests for enhanced agent hierarchy functionality."""

    def test_get_root_agents(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test get_root_agents method."""
        root_agents = registry.get_root_agents()
        assert len(root_agents) == 1
        assert root_agents[0].get_agent_id() == agent_hierarchy["architect"]

    def test_get_leaf_agents(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test get_leaf_agents method."""
        leaf_agents = registry.get_leaf_agents()
        assert len(leaf_agents) == 3
        leaf_agent_ids = [agent.get_agent_id() for agent in leaf_agents]
        assert agent_hierarchy["executor1"] in leaf_agent_ids
        assert agent_hierarchy["executor2"] in leaf_agent_ids
        assert agent_hierarchy["executor3"] in leaf_agent_ids

    def test_get_ancestors(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test get_ancestors method."""
        # Test executor1's ancestors
        ancestors = registry.get_ancestors(agent_hierarchy["executor1"])
        assert len(ancestors) == 2
        assert ancestors[0].get_agent_id() == agent_hierarchy["planner1"]
        assert ancestors[1].get_agent_id() == agent_hierarchy["architect"]

        # Test planner1's ancestors
        ancestors = registry.get_ancestors(agent_hierarchy["planner1"])
        assert len(ancestors) == 1
        assert ancestors[0].get_agent_id() == agent_hierarchy["architect"]

        # Test architect's ancestors (should be empty)
        ancestors = registry.get_ancestors(agent_hierarchy["architect"])
        assert len(ancestors) == 0

    def test_get_descendants(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test get_descendants method."""
        # Test architect's descendants
        descendants = registry.get_descendants(agent_hierarchy["architect"])
        assert len(descendants) == 5
        descendant_ids = [agent.get_agent_id() for agent in descendants]
        assert agent_hierarchy["planner1"] in descendant_ids
        assert agent_hierarchy["planner2"] in descendant_ids
        assert agent_hierarchy["executor1"] in descendant_ids
        assert agent_hierarchy["executor2"] in descendant_ids
        assert agent_hierarchy["executor3"] in descendant_ids

        # Test planner1's descendants
        descendants = registry.get_descendants(agent_hierarchy["planner1"])
        assert len(descendants) == 2
        descendant_ids = [agent.get_agent_id() for agent in descendants]
        assert agent_hierarchy["executor1"] in descendant_ids
        assert agent_hierarchy["executor2"] in descendant_ids

        # Test executor1's descendants (should be empty)
        descendants = registry.get_descendants(agent_hierarchy["executor1"])
        assert len(descendants) == 0

    def test_validate_hierarchy(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test validate_hierarchy method."""
        # Valid hierarchy should have no inconsistencies
        inconsistencies = registry.validate_hierarchy()
        assert len(inconsistencies) == 0

        # Create an inconsistency by manually breaking a relationship
        planner1 = registry.get_agent(agent_hierarchy["planner1"])
        planner1.set_parent("non-existent-id")

        # Verify the inconsistency exists
        inconsistencies = registry.validate_hierarchy()
        assert len(inconsistencies) > 0
        assert any("non-existent" in msg for msg in inconsistencies)

    def test_repair_hierarchy(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test repair_hierarchy method."""
        # Create an inconsistency by manually breaking a relationship
        planner1 = registry.get_agent(agent_hierarchy["planner1"])
        planner1.set_parent("non-existent-id")

        # Verify the inconsistency exists
        inconsistencies = registry.validate_hierarchy()
        assert len(inconsistencies) > 0

        # Repair the hierarchy
        repairs_count = registry.repair_hierarchy()
        assert repairs_count > 0

        # Verify the inconsistency is fixed
        inconsistencies = registry.validate_hierarchy()
        assert len(inconsistencies) == 0

    def test_would_create_cycle(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test _would_create_cycle method."""
        # Trying to make executor1 the parent of architect would create a cycle
        assert registry._would_create_cycle(agent_hierarchy["executor1"], agent_hierarchy["architect"])

        # Trying to make executor1 the parent of planner2 would not create a cycle
        assert not registry._would_create_cycle(agent_hierarchy["executor1"], agent_hierarchy["planner2"])

        # Trying to make an agent its own parent would create a cycle
        assert registry._would_create_cycle(agent_hierarchy["executor1"], agent_hierarchy["executor1"])

    def test_cycle_prevention(self, registry: InMemoryAgentRegistry, agent_hierarchy: dict[str, str]) -> None:
        """Test that the registry prevents creating cycles."""
        # Trying to make executor1 the parent of architect should raise an error
        with pytest.raises(ValueError, match="circular relationship"):
            registry.register_parent_child_relationship(
                agent_hierarchy["executor1"],
                agent_hierarchy["architect"],
            )

        # Trying to make an agent its own parent should raise an error
        with pytest.raises(ValueError, match="circular relationship"):
            registry.register_parent_child_relationship(
                agent_hierarchy["executor1"],
                agent_hierarchy["executor1"],
            )

    def test_hierarchy_cache_invalidation(
        self,
        registry: InMemoryAgentRegistry,
        agent_hierarchy: dict[str, str],
        mock_provider: MagicMock,
    ) -> None:
        """Test that the hierarchy cache is invalidated when relationships change."""
        # Get the hierarchy to populate the cache
        hierarchy1 = registry.get_agent_hierarchy(agent_hierarchy["architect"])

        # Add a new agent and relationship
        new_executor = create_executor_agent(provider=mock_provider)
        registry.register_agent(new_executor)
        registry.register_parent_child_relationship(agent_hierarchy["planner1"], new_executor.get_agent_id())

        # Get the hierarchy again, should include the new agent
        hierarchy2 = registry.get_agent_hierarchy(agent_hierarchy["architect"])
        assert len(hierarchy2) > len(hierarchy1)
        assert new_executor.get_agent_id() in hierarchy2
