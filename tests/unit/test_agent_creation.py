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

    def test_resource_limits_max_agents(self) -> None:
        """Test that agent creation respects the maximum agent limit."""
        # Set a low maximum agent limit for testing
        self.coordinator._resource_limits["max_agents"] = 2

        # Create agents up to the limit
        self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        self.coordinator.create_agent_by_role(AgentRole.PLANNER, {})

        # Attempting to create one more should raise an error
        with pytest.raises(ValueError, match="Maximum number of agents"):
            self.coordinator.create_agent_by_role(AgentRole.EXECUTOR, {})

    def test_resource_limits_max_agents_per_role(self) -> None:
        """Test that agent creation respects the maximum agents per role limit."""
        # Mock the find_agents_by_role method to return a list of the specified length
        original_find_agents_by_role = self.registry.find_agents_by_role

        def mock_find_agents_by_role(role: str) -> list:
            if role == AgentRole.PLANNER.value:
                # Return a list with one item to simulate having one planner agent
                return [1]  # The actual content doesn't matter, just the length
            return original_find_agents_by_role(role)

        # Apply the mock
        self.registry.find_agents_by_role = mock_find_agents_by_role

        # Set a low maximum for the planner role
        self.coordinator._resource_limits["max_agents_per_role"] = {AgentRole.PLANNER.value: 1}

        # Attempting to create a planner should raise an error since our mock says we already have one
        with pytest.raises(ValueError, match="Maximum number of planner agents"):
            self.coordinator.create_agent_by_role(AgentRole.PLANNER, {})

        # Restore the original method
        self.registry.find_agents_by_role = original_find_agents_by_role

    def test_resource_limits_max_children_per_agent(self) -> None:
        """Test that agent creation respects the maximum children per agent limit."""
        # Set a low maximum children limit
        self.coordinator._resource_limits["max_children_per_agent"] = 2

        # Create a parent agent
        parent = self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        parent_id = parent.get_agent_id()

        # Create children up to the limit
        self.coordinator.create_agent_by_role(AgentRole.PLANNER, {"parent_id": parent_id})
        self.coordinator.create_agent_by_role(AgentRole.PLANNER, {"parent_id": parent_id})

        # Attempting to create one more child should raise an error
        with pytest.raises(ValueError, match="Maximum number of children"):
            self.coordinator.create_agent_by_role(AgentRole.EXECUTOR, {"parent_id": parent_id})

    def test_resource_limits_max_hierarchy_depth(self) -> None:
        """Test that agent creation respects the maximum hierarchy depth limit."""
        # Set a low maximum hierarchy depth
        self.coordinator._resource_limits["max_hierarchy_depth"] = 2

        # Create a hierarchy up to the limit
        architect = self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        planner = self.coordinator.create_agent_by_role(
            AgentRole.PLANNER,
            {"parent_id": architect.get_agent_id()},
        )

        # Attempting to create another level should raise an error
        with pytest.raises(ValueError, match="Maximum hierarchy depth"):
            self.coordinator.create_agent_by_role(
                AgentRole.EXECUTOR,
                {"parent_id": planner.get_agent_id()},
            )

    def test_capability_discovery(self) -> None:
        """Test the capability discovery mechanism."""
        # Create agents with different capabilities
        self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        self.coordinator.create_agent_by_role(AgentRole.PLANNER, {})
        self.coordinator.create_agent_by_role(AgentRole.EXECUTOR, {})

        # Discover capabilities
        capabilities = self.coordinator.discover_capabilities()

        # Verify that capabilities are properly categorized
        assert "design" in capabilities.get("design", [])
        assert "planning" in capabilities.get("planning", [])
        assert "implementation" in capabilities.get("development", [])

    def test_hierarchical_agent_creation(self) -> None:
        """Test creating a complete agent hierarchy."""
        # Create the root architect
        architect = self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        architect_id = architect.get_agent_id()

        # Create planners under the architect
        planner1 = self.coordinator.create_agent_by_role(
            AgentRole.PLANNER,
            {"parent_id": architect_id},
        )
        planner2 = self.coordinator.create_agent_by_role(
            AgentRole.PLANNER,
            {"parent_id": architect_id},
        )

        # Create executors under the planners
        executor1 = self.coordinator.create_agent_by_role(
            AgentRole.EXECUTOR,
            {"parent_id": planner1.get_agent_id()},
        )
        executor2 = self.coordinator.create_agent_by_role(
            AgentRole.EXECUTOR,
            {"parent_id": planner2.get_agent_id()},
        )

        # Verify the hierarchy
        assert len(architect.get_child_ids()) == 2
        assert planner1.get_agent_id() in architect.get_child_ids()
        assert planner2.get_agent_id() in architect.get_child_ids()

        assert len(planner1.get_child_ids()) == 1
        assert executor1.get_agent_id() in planner1.get_child_ids()

        assert len(planner2.get_child_ids()) == 1
        assert executor2.get_agent_id() in planner2.get_child_ids()

        # Verify parent-child relationships
        assert planner1.get_parent_id() == architect_id
        assert planner2.get_parent_id() == architect_id
        assert executor1.get_parent_id() == planner1.get_agent_id()
        assert executor2.get_parent_id() == planner2.get_agent_id()

        # Verify hierarchy traversal
        hierarchy = self.registry.get_agent_hierarchy(architect_id)
        assert planner1.get_agent_id() in hierarchy[architect_id]
        assert planner2.get_agent_id() in hierarchy[architect_id]
        assert executor1.get_agent_id() in hierarchy[planner1.get_agent_id()]
        assert executor2.get_agent_id() in hierarchy[planner2.get_agent_id()]

    def test_find_agents_by_capability(self) -> None:
        """Test finding agents by capability."""
        # Create agents with different capabilities
        architect = self.coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
        planner = self.coordinator.create_agent_by_role(AgentRole.PLANNER, {})
        executor = self.coordinator.create_agent_by_role(AgentRole.EXECUTOR, {})

        # Find agents by capability
        design_agents = self.registry.find_agents_by_capability("design")
        planning_agents = self.registry.find_agents_by_capability("planning")
        implementation_agents = self.registry.find_agents_by_capability("implementation")

        # Verify results
        assert any(agent.agent_id == architect.get_agent_id() for agent in design_agents)
        assert any(agent.agent_id == planner.get_agent_id() for agent in planning_agents)
        assert any(agent.agent_id == executor.get_agent_id() for agent in implementation_agents)
