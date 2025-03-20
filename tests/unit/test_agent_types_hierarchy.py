"""Unit tests for agent hierarchy and relationships."""

from __future__ import annotations

from typing import Any

import pytest

from src.agent.agent_types import Agent, MockAgent
from src.common_types import AgentInfo
from src.common_types.result_types import Result


class HierarchicalMockAgent(MockAgent):
    """Mock agent with hierarchy support."""

    def __init__(self, agent_id: str, capabilities: list[str]) -> None:
        """Initialize the agent.

        Args:
            agent_id: Agent ID
            capabilities: List of capabilities

        """
        super().__init__(agent_id, capabilities)
        self.parent_id: str | None = None
        self.child_ids: list[str] = []
        self.child_results: dict[str, Result[Any]] = {}

    def get_parent_id(self) -> str | None:
        """Get parent agent ID.

        Returns:
            Parent agent ID or None if no parent.

        """
        return self.parent_id

    def get_child_ids(self) -> list[str]:
        """Get child agent IDs.

        Returns:
            List of child agent IDs.

        """
        return self.child_ids.copy()

    def add_child(self, child_agent_id: str) -> None:
        """Add a child agent.

        Args:
            child_agent_id: Child agent ID to add.

        """
        if child_agent_id not in self.child_ids:
            self.child_ids.append(child_agent_id)

    def remove_child(self, child_agent_id: str) -> None:
        """Remove a child agent.

        Args:
            child_agent_id: Child agent ID to remove.

        """
        if child_agent_id in self.child_ids:
            self.child_ids.remove(child_agent_id)

    def set_parent(self, parent_agent_id: str) -> None:
        """Set parent agent.

        Args:
            parent_agent_id: Parent agent ID.

        """
        self.parent_id = parent_agent_id

    def clear_parent(self) -> None:
        """Clear parent agent."""
        self.parent_id = None

    def delegate_to_child(self, child_agent_id: str, task: str) -> Result[Any]:
        """Delegate task to child agent.

        Args:
            child_agent_id: Child agent ID.
            task: Task to delegate.

        Returns:
            Result of task delegation.

        """
        if child_agent_id not in self.child_ids:
            return Result(success=False, error=f"Child agent {child_agent_id} not found")
        result = Result(success=True, data=f"Delegated {task} to {child_agent_id}")
        self.child_results[child_agent_id] = result
        return result

    def collect_results_from_children(self) -> dict[str, Result[Any]]:
        """Collect results from child agents.

        Returns:
            Dictionary mapping child agent IDs to their results.

        """
        return self.child_results.copy()


class MockAgentRegistry:
    """Mock agent registry for testing."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self.agents: dict[str, Agent[Any]] = {}
        self.agent_info: dict[str, AgentInfo] = {}

    def register_agent(self, agent: Agent[Any], info: AgentInfo) -> None:
        """Register an agent.

        Args:
            agent: Agent to register.
            info: Agent info.

        """
        self.agents[info.agent_id] = agent
        self.agent_info[info.agent_id] = info

    def get_agent(self, agent_id: str) -> Agent[Any]:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent instance.

        Raises:
            KeyError: If agent not found.

        """
        return self.agents[agent_id]

    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """Get agent info by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent info.

        Raises:
            KeyError: If agent not found.

        """
        return self.agent_info[agent_id]

    def get_parent_agent(self, agent_id: str) -> Agent[Any] | None:
        """Get parent agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Parent agent or None if no parent.

        """
        info = self.agent_info.get(agent_id)
        if info and info.parent_id:
            return self.agents.get(info.parent_id)
        return None

    def get_child_agents(self, agent_id: str) -> list[Agent[Any]]:
        """Get child agents.

        Args:
            agent_id: Agent ID.

        Returns:
            List of child agents.

        """
        info = self.agent_info.get(agent_id)
        if info:
            return [self.agents[child_id] for child_id in info.child_ids]
        return []

    def get_sibling_agents(self, agent_id: str) -> list[Agent[Any]]:
        """Get sibling agents.

        Args:
            agent_id: Agent ID.

        Returns:
            List of sibling agents.

        """
        info = self.agent_info.get(agent_id)
        if info and info.parent_id:
            parent_info = self.agent_info.get(info.parent_id)
            if parent_info:
                return [self.agents[child_id] for child_id in parent_info.child_ids if child_id != agent_id]
        return []

    def register_parent_child_relationship(
        self,
        parent_id: str,
        child_id: str,
    ) -> None:
        """Register parent-child relationship.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        """
        parent_info = self.agent_info.get(parent_id)
        child_info = self.agent_info.get(child_id)
        if parent_info and child_info:
            if child_id not in parent_info.child_ids:
                parent_info.child_ids.append(child_id)
            child_info.parent_id = parent_id

    def remove_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Remove parent-child relationship.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        """
        parent_info = self.agent_info.get(parent_id)
        child_info = self.agent_info.get(child_id)
        if parent_info and child_info:
            if child_id in parent_info.child_ids:
                parent_info.child_ids.remove(child_id)
            if child_info.parent_id == parent_id:
                child_info.parent_id = None

    def get_agent_hierarchy(self, root_agent_id: str) -> dict[str, list[str]]:
        """Get agent hierarchy.

        Args:
            root_agent_id: Root agent ID.

        Returns:
            Dictionary mapping agent IDs to their child IDs.

        Raises:
            KeyError: If root agent not found.

        """
        if root_agent_id not in self.agent_info:
            msg = f"Agent not found: {root_agent_id}"
            raise KeyError(msg)

        hierarchy: dict[str, list[str]] = {}
        queue = [root_agent_id]
        visited = set()

        while queue:
            agent_id = queue.pop(0)
            if agent_id in visited:
                continue
            visited.add(agent_id)

            info = self.agent_info.get(agent_id)
            if info:
                hierarchy[agent_id] = info.child_ids.copy()
                queue.extend(info.child_ids)

        return hierarchy


def test_hierarchical_mock_agent_initialization() -> None:
    """Test HierarchicalMockAgent initialization."""
    agent = HierarchicalMockAgent(agent_id="test-agent", capabilities=["test"])
    assert agent.get_agent_id() == "test-agent"
    assert agent.get_capabilities() == ["test"]
    assert agent.get_parent_id() is None
    assert agent.get_child_ids() == []


def test_hierarchical_mock_agent_parent_child_operations() -> None:
    """Test HierarchicalMockAgent parent-child operations."""
    agent = HierarchicalMockAgent(agent_id="test-agent", capabilities=["test"])

    # Test parent operations
    agent.set_parent("parent-agent")
    assert agent.get_parent_id() == "parent-agent"

    agent.clear_parent()
    assert agent.get_parent_id() is None

    # Test child operations
    agent.add_child("child-1")
    agent.add_child("child-2")
    assert set(agent.get_child_ids()) == {"child-1", "child-2"}

    agent.remove_child("child-1")
    assert agent.get_child_ids() == ["child-2"]


def test_hierarchical_mock_agent_task_delegation() -> None:
    """Test HierarchicalMockAgent task delegation."""
    agent = HierarchicalMockAgent(agent_id="test-agent", capabilities=["test"])

    # Test delegation to non-existent child
    result = agent.delegate_to_child("non-existent", "test task")
    assert not result.success
    assert "not found" in str(result.error)

    # Test delegation to existing child
    agent.add_child("child-1")
    result = agent.delegate_to_child("child-1", "test task")
    assert result.success
    assert result.data == "Delegated test task to child-1"

    # Test collecting results
    results = agent.collect_results_from_children()
    assert len(results) == 1
    assert "child-1" in results
    assert results["child-1"].success


def test_mock_agent_registry_initialization() -> None:
    """Test MockAgentRegistry initialization."""
    registry = MockAgentRegistry()
    assert not registry.agents
    assert not registry.agent_info


def test_mock_agent_registry_registration() -> None:
    """Test MockAgentRegistry agent registration."""
    registry = MockAgentRegistry()
    agent = HierarchicalMockAgent(agent_id="test-agent", capabilities=["test"])
    info = AgentInfo(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["test"],
    )

    registry.register_agent(agent, info)
    assert registry.get_agent("test-agent") == agent
    assert registry.get_agent_info("test-agent") == info


def test_mock_agent_registry_relationships() -> None:
    """Test MockAgentRegistry relationship management."""
    registry = MockAgentRegistry()

    # Create agents
    parent = HierarchicalMockAgent(agent_id="parent", capabilities=["test"])
    child1 = HierarchicalMockAgent(agent_id="child1", capabilities=["test"])
    child2 = HierarchicalMockAgent(agent_id="child2", capabilities=["test"])

    # Create agent info
    parent_info = AgentInfo(
        agent_id="parent",
        name="Parent Agent",
        description="Parent agent",
        capabilities=["test"],
    )
    child1_info = AgentInfo(
        agent_id="child1",
        name="Child Agent 1",
        description="Child agent 1",
        capabilities=["test"],
    )
    child2_info = AgentInfo(
        agent_id="child2",
        name="Child Agent 2",
        description="Child agent 2",
        capabilities=["test"],
    )

    # Register agents
    registry.register_agent(parent, parent_info)
    registry.register_agent(child1, child1_info)
    registry.register_agent(child2, child2_info)

    # Test relationship registration
    registry.register_parent_child_relationship("parent", "child1")
    registry.register_parent_child_relationship("parent", "child2")

    # Test parent-child relationships
    assert registry.get_parent_agent("child1") == parent
    assert registry.get_parent_agent("child2") == parent
    assert set(registry.get_child_agents("parent")) == {child1, child2}

    # Test sibling relationships
    assert set(registry.get_sibling_agents("child1")) == {child2}
    assert set(registry.get_sibling_agents("child2")) == {child1}

    # Test hierarchy
    hierarchy = registry.get_agent_hierarchy("parent")
    assert hierarchy == {"parent": ["child1", "child2"], "child1": [], "child2": []}

    # Test relationship removal
    registry.remove_parent_child_relationship("parent", "child1")
    assert registry.get_parent_agent("child1") is None
    assert registry.get_child_agents("parent") == [child2]
    assert not registry.get_sibling_agents("child2")


def test_mock_agent_registry_edge_cases() -> None:
    """Test MockAgentRegistry edge cases."""
    registry = MockAgentRegistry()

    # Test getting non-existent agents
    with pytest.raises(KeyError):
        registry.get_agent("non-existent")
    with pytest.raises(KeyError):
        registry.get_agent_info("non-existent")

    # Test relationships with non-existent agents
    assert registry.get_parent_agent("non-existent") is None
    assert not registry.get_child_agents("non-existent")
    assert not registry.get_sibling_agents("non-existent")

    # Test empty hierarchy
    with pytest.raises(KeyError):
        registry.get_agent_hierarchy("non-existent")

    # Test relationship operations with non-existent agents
    registry.register_parent_child_relationship("non-existent", "also-non-existent")
    registry.remove_parent_child_relationship("non-existent", "also-non-existent")


def test_mock_agent_registry_get_agent_hierarchy() -> None:
    """Test MockAgentRegistry get_agent_hierarchy method."""
    registry = MockAgentRegistry()

    # Create a hierarchy of agents
    root = HierarchicalMockAgent(agent_id="root", capabilities=["test"])
    child1 = HierarchicalMockAgent(agent_id="child1", capabilities=["test"])
    child2 = HierarchicalMockAgent(agent_id="child2", capabilities=["test"])
    grandchild1 = HierarchicalMockAgent(agent_id="grandchild1", capabilities=["test"])
    grandchild2 = HierarchicalMockAgent(agent_id="grandchild2", capabilities=["test"])

    # Set up relationships
    child1.set_parent(root.get_agent_id())
    child2.set_parent(root.get_agent_id())
    grandchild1.set_parent(child1.get_agent_id())
    grandchild2.set_parent(child1.get_agent_id())

    root.add_child(child1.get_agent_id())
    root.add_child(child2.get_agent_id())
    child1.add_child(grandchild1.get_agent_id())
    child1.add_child(grandchild2.get_agent_id())

    # Register all agents
    registry.register_agent(
        root,
        AgentInfo(
            agent_id="root",
            name="Root Agent",
            description="Root test agent",
            capabilities=["test"],
            child_ids=["child1", "child2"],
        ),
    )

    registry.register_agent(
        child1,
        AgentInfo(
            agent_id="child1",
            name="Child Agent 1",
            description="First child test agent",
            capabilities=["test"],
            parent_id="root",
            child_ids=["grandchild1", "grandchild2"],
        ),
    )

    registry.register_agent(
        child2,
        AgentInfo(
            agent_id="child2",
            name="Child Agent 2",
            description="Second child test agent",
            capabilities=["test"],
            parent_id="root",
        ),
    )

    registry.register_agent(
        grandchild1,
        AgentInfo(
            agent_id="grandchild1",
            name="Grandchild Agent 1",
            description="First grandchild test agent",
            capabilities=["test"],
            parent_id="child1",
        ),
    )

    registry.register_agent(
        grandchild2,
        AgentInfo(
            agent_id="grandchild2",
            name="Grandchild Agent 2",
            description="Second grandchild test agent",
            capabilities=["test"],
            parent_id="child1",
        ),
    )

    # Test getting hierarchy from root
    hierarchy = registry.get_agent_hierarchy("root")
    assert "root" in hierarchy
    assert set(hierarchy["root"]) == {"child1", "child2"}
    assert set(hierarchy["child1"]) == {"grandchild1", "grandchild2"}
    assert hierarchy["child2"] == []
    assert hierarchy["grandchild1"] == []
    assert hierarchy["grandchild2"] == []

    # Test getting hierarchy from middle of tree
    hierarchy = registry.get_agent_hierarchy("child1")
    assert "child1" in hierarchy
    assert set(hierarchy["child1"]) == {"grandchild1", "grandchild2"}
    assert hierarchy["grandchild1"] == []
    assert hierarchy["grandchild2"] == []
    assert "root" not in hierarchy
    assert "child2" not in hierarchy

    # Test getting hierarchy from leaf
    hierarchy = registry.get_agent_hierarchy("grandchild1")
    assert "grandchild1" in hierarchy
    assert hierarchy["grandchild1"] == []

    # Test getting hierarchy for non-existent agent
    with pytest.raises(KeyError):
        registry.get_agent_hierarchy("non-existent")


def test_mock_agent_registry_parent_child_relationships() -> None:
    """Test MockAgentRegistry parent-child relationship methods."""
    registry = MockAgentRegistry()

    # Create and register parent agent
    parent = HierarchicalMockAgent(agent_id="parent", capabilities=["test"])
    registry.register_agent(
        parent,
        AgentInfo(
            agent_id="parent",
            name="Parent Agent",
            description="Parent test agent",
            capabilities=["test"],
            child_ids=["child1", "child2"],
        ),
    )

    # Create and register child agents
    child1 = HierarchicalMockAgent(agent_id="child1", capabilities=["test"])
    registry.register_agent(
        child1,
        AgentInfo(
            agent_id="child1",
            name="Child Agent 1",
            description="First child test agent",
            capabilities=["test"],
            parent_id="parent",
        ),
    )

    child2 = HierarchicalMockAgent(agent_id="child2", capabilities=["test"])
    registry.register_agent(
        child2,
        AgentInfo(
            agent_id="child2",
            name="Child Agent 2",
            description="Second child test agent",
            capabilities=["test"],
            parent_id="parent",
        ),
    )

    # Test getting parent agent
    parent_agent = registry.get_parent_agent("child1")
    assert parent_agent is not None
    assert parent_agent.get_agent_id() == "parent"

    # Test getting child agents
    child_agents = registry.get_child_agents("parent")
    assert len(child_agents) == 2
    child_ids = {agent.get_agent_id() for agent in child_agents}
    assert child_ids == {"child1", "child2"}

    # Test getting parent of root agent
    assert registry.get_parent_agent("parent") is None

    # Test getting children of leaf agent
    assert registry.get_child_agents("child1") == []

    # Test with non-existent agents
    assert registry.get_parent_agent("non-existent") is None
    assert registry.get_child_agents("non-existent") == []
