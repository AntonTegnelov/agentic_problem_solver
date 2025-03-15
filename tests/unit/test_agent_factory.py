"""Unit tests for agent factory methods."""

import pytest

from src.agent.agent_types import (
    create_agent,
    create_architect_agent,
    create_executor_agent,
    create_planner_agent,
)
from src.agent.agent_types.architect import ArchitectAgent
from src.agent.agent_types.executor import ExecutorAgent
from src.agent.agent_types.planner import PlannerAgent
from src.common_types.enums import AgentRole


class TestAgentFactory:
    """Tests for agent factory methods."""

    def test_create_architect_agent(self) -> None:
        """Test create_architect_agent function."""
        agent = create_architect_agent()
        assert isinstance(agent, ArchitectAgent)
        assert agent.get_agent_id().startswith("architect_")

    def test_create_planner_agent(self) -> None:
        """Test create_planner_agent function."""
        agent = create_planner_agent()
        assert isinstance(agent, PlannerAgent)
        assert agent.get_agent_id().startswith("planner_")

    def test_create_executor_agent(self) -> None:
        """Test create_executor_agent function."""
        agent = create_executor_agent()
        assert isinstance(agent, ExecutorAgent)
        assert agent.get_agent_id().startswith("executor_")

    def test_create_agent_by_role(self) -> None:
        """Test create_agent function with different roles."""
        architect = create_agent(AgentRole.ARCHITECT)
        assert isinstance(architect, ArchitectAgent)
        assert architect.get_agent_id().startswith("architect_")

        planner = create_agent(AgentRole.PLANNER)
        assert isinstance(planner, PlannerAgent)
        assert planner.get_agent_id().startswith("planner_")

        executor = create_agent(AgentRole.EXECUTOR)
        assert isinstance(executor, ExecutorAgent)
        assert executor.get_agent_id().startswith("executor_")

    def test_create_agent_with_invalid_role(self) -> None:
        """Test create_agent function with an invalid role."""
        with pytest.raises(ValueError, match="Unsupported agent role"):
            create_agent(AgentRole.SOLVER)

    def test_create_agent_with_parent_id(self) -> None:
        """Test creating agents with parent-child relationships."""
        # Create a parent architect agent
        architect = create_architect_agent()
        architect_id = architect.get_agent_id()

        # Create a planner agent with the architect as parent
        planner = create_planner_agent(parent_id=architect_id)
        assert planner.get_parent_id() == architect_id

        # Create an executor agent with the planner as parent
        planner_id = planner.get_agent_id()
        executor = create_executor_agent(parent_id=planner_id)
        assert executor.get_parent_id() == planner_id

        # Test with create_agent function
        executor2 = create_agent(AgentRole.EXECUTOR, parent_id=planner_id)
        assert executor2.get_parent_id() == planner_id
