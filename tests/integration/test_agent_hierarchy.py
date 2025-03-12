"""Integration tests for hierarchical agent system.

This module contains integration tests for the hierarchical agent system,
testing how different agent types work together in a multi-tier workflow.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.agent.agent_types import (
    create_architect_agent,
    create_executor_agent,
    create_planner_agent,
)
from src.agent.coordination import InMemoryAgentRegistry


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate.return_value = "Mock response"
    return provider


@pytest.fixture
def registry() -> InMemoryAgentRegistry:
    """Create an InMemoryAgentRegistry instance."""
    return InMemoryAgentRegistry()


@pytest.fixture
def hierarchical_system(registry: InMemoryAgentRegistry, mock_provider: MagicMock) -> dict[str, str]:
    """Create a hierarchical agent system for testing.

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


class TestHierarchicalAgentSystem:
    """Integration tests for hierarchical agent system."""

    def test_task_delegation_from_architect_to_planner(
        self,
        registry: InMemoryAgentRegistry,
        hierarchical_system: dict[str, str],
        mock_provider: MagicMock,
    ) -> None:
        """Test task delegation from architect to planner."""
        # Set up mock responses
        mock_provider.generate.side_effect = [
            "Task delegated to planner1",  # Architect response
            "Task received by planner1",  # Planner response
        ]

        # Get agents
        architect = registry.get_agent(hierarchical_system["architect"])
        planner1 = registry.get_agent(hierarchical_system["planner1"])

        # Create a task message
        task_message = HumanMessage(content="Design a system for task management")

        # Process the task with the architect
        result = architect.process(task_message)
        assert result.success is True
        assert "Task delegated to planner1" in result.data

        # Verify that the planner can process a message
        planner_message = HumanMessage(content="Plan the implementation of task management system")
        result = planner1.process(planner_message)
        assert result.success is True
        assert "Task received by planner1" in result.data

    def test_task_delegation_from_planner_to_executor(
        self,
        registry: InMemoryAgentRegistry,
        hierarchical_system: dict[str, str],
        mock_provider: MagicMock,
    ) -> None:
        """Test task delegation from planner to executor."""
        # Set up mock responses
        mock_provider.generate.side_effect = [
            "Task delegated to executor1",  # Planner response
            "Task implemented by executor1",  # Executor response
        ]

        # Get agents
        planner1 = registry.get_agent(hierarchical_system["planner1"])
        executor1 = registry.get_agent(hierarchical_system["executor1"])

        # Create a task message
        task_message = HumanMessage(content="Plan the implementation of a login system")

        # Process the task with the planner
        result = planner1.process(task_message)
        assert result.success is True
        assert "Task delegated to executor1" in result.data

        # Verify that the executor can process a message
        executor_message = HumanMessage(content="Implement the login form component")
        result = executor1.process(executor_message)
        assert result.success is True
        assert "Task implemented by executor1" in result.data

    def test_multi_tier_workflow(
        self,
        registry: InMemoryAgentRegistry,
        hierarchical_system: dict[str, str],
        mock_provider: MagicMock,
    ) -> None:
        """Test a complete multi-tier workflow from architect to executor."""
        # Set up mock responses for each agent in the workflow
        mock_provider.generate.side_effect = [
            "Breaking down task into components: UI, API, Database",  # Architect response
            "Planning UI implementation with components: Login, Dashboard, Settings",  # Planner1 response
            "Implementing Login component with username and password fields",  # Executor1 response
        ]

        # Get agents
        architect = registry.get_agent(hierarchical_system["architect"])
        planner1 = registry.get_agent(hierarchical_system["planner1"])
        executor1 = registry.get_agent(hierarchical_system["executor1"])

        # Step 1: Architect breaks down the task
        architect_message = HumanMessage(content="Design a web application for task management")
        architect_result = architect.process(architect_message)
        assert architect_result.success is True
        assert "Breaking down task into components" in architect_result.data

        # Step 2: Planner creates detailed implementation plan
        planner_message = HumanMessage(content="Plan the UI implementation for the task management app")
        planner_result = planner1.process(planner_message)
        assert planner_result.success is True
        assert "Planning UI implementation" in planner_result.data

        # Step 3: Executor implements a specific component
        executor_message = HumanMessage(content="Implement the Login component for the UI")
        executor_result = executor1.process(executor_message)
        assert executor_result.success is True
        assert "Implementing Login component" in executor_result.data

    def test_result_collection_from_children(
        self,
        registry: InMemoryAgentRegistry,
        hierarchical_system: dict[str, str],
    ) -> None:
        """Test collecting results from child agents."""
        # Get agents
        architect = registry.get_agent(hierarchical_system["architect"])
        planner1 = registry.get_agent(hierarchical_system["planner1"])

        # Collect results from children
        architect_results = architect.collect_results_from_children()
        planner_results = planner1.collect_results_from_children()

        # Verify architect results
        assert len(architect_results) == 2
        assert hierarchical_system["planner1"] in architect_results
        assert hierarchical_system["planner2"] in architect_results

        # Verify planner results
        assert len(planner_results) == 2
        assert hierarchical_system["executor1"] in planner_results
        assert hierarchical_system["executor2"] in planner_results

    def test_capability_based_task_routing(
        self,
        registry: InMemoryAgentRegistry,
        hierarchical_system: dict[str, str],
        mock_provider: MagicMock,
    ) -> None:
        """Test routing tasks based on agent capabilities."""
        # Get agents
        architect = registry.get_agent(hierarchical_system["architect"])
        planner1 = registry.get_agent(hierarchical_system["planner1"])
        executor1 = registry.get_agent(hierarchical_system["executor1"])

        # Test architect capabilities
        assert architect.can_handle("Design a system architecture")
        assert not architect.can_handle("Implement this function")

        # Test planner capabilities
        assert planner1.can_handle("Plan the implementation steps")
        assert not planner1.can_handle("Design a system architecture")

        # Test executor capabilities
        assert executor1.can_handle("Implement this function")
        assert not executor1.can_handle("Design a system architecture")

        # Find agents by capability
        design_agents = registry.find_agents_by_capability("design")
        planning_agents = registry.find_agents_by_capability("planning")
        coding_agents = registry.find_agents_by_capability("coding")

        # Verify capability-based agent discovery
        assert any(agent.agent_id == hierarchical_system["architect"] for agent in design_agents)
        assert any(agent.agent_id == hierarchical_system["planner1"] for agent in planning_agents)
        assert any(agent.agent_id == hierarchical_system["executor1"] for agent in coding_agents)
