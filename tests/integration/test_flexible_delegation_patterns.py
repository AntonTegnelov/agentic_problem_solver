"""Integration tests for flexible delegation patterns.

This module contains integration tests that verify direct and recursive delegation patterns
between different agent types in the hierarchical agent system.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages.base import BaseMessage

from src.agent.agent_types import (
    create_architect_agent,
    create_executor_agent,
    create_planner_agent,
)
from src.agent.coordination import InMemoryAgentRegistry
from src.common_types.message_types import HumanMessage
from src.common_types.task_types import TaskComplexity


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider."""
    provider = MagicMock()

    async def mock_generate(_messages: list[BaseMessage]) -> str:
        return "Test response"

    async def mock_stream(_messages: list[BaseMessage]) -> AsyncGenerator[str, None]:
        chunks = ["Mock", " stream", " response"]
        for chunk in chunks:
            yield chunk

    # Set up the generate method
    provider.generate = AsyncMock(side_effect=mock_generate)
    provider.generate_stream = mock_stream
    provider.__bool__.return_value = True
    return provider


@pytest.fixture
def registry() -> InMemoryAgentRegistry:
    """Create an InMemoryAgentRegistry instance."""
    return InMemoryAgentRegistry()


@pytest.fixture
def flexible_delegation_system(registry: InMemoryAgentRegistry, mock_provider: MagicMock) -> dict[str, str]:
    """Create a hierarchical agent system with flexible delegation capabilities."""
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

    # Set up parent-child relationships using the registry
    registry.register_parent_child_relationship(architect.get_agent_id(), planner1.get_agent_id())
    registry.register_parent_child_relationship(architect.get_agent_id(), executor1.get_agent_id())  # Direct delegation
    registry.register_parent_child_relationship(
        planner1.get_agent_id(),
        planner2.get_agent_id(),
    )  # Recursive delegation
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


class TestFlexibleDelegationPatterns:
    """Integration tests for flexible delegation patterns."""

    @pytest.mark.asyncio
    async def test_architect_direct_delegation_to_executor(
        self,
        registry: InMemoryAgentRegistry,
        flexible_delegation_system: dict[str, str],
    ) -> None:
        """Test direct delegation from architect to executor for simple tasks."""
        # Get agents
        architect = registry.get_agent(flexible_delegation_system["architect"])

        # Test direct delegation to executor
        with patch.object(
            type(architect),
            "analyze_task_complexity",
            return_value=TaskComplexity.SIMPLE,
        ) as mock_analyze:
            # Call the delegate_to_executor method directly
            result = await architect.delegate_to_executor("Implement a simple login form")

            # Verify the task was analyzed for complexity
            mock_analyze.assert_called_once()

            # Verify the result
            assert result.success is True
            # The method now returns the executor's result directly
            assert result.data == "Test response"

    @pytest.mark.asyncio
    async def test_planner_recursive_delegation_to_planner(
        self,
        registry: InMemoryAgentRegistry,
        flexible_delegation_system: dict[str, str],
    ) -> None:
        """Test recursive delegation from planner to another planner for complex tasks."""
        # Get agents
        planner1 = registry.get_agent(flexible_delegation_system["planner1"])

        # Test recursive delegation to another planner
        with patch.object(
            type(planner1),
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.COMPLEX,
        ) as mock_evaluate:
            # Call the delegate_to_planner method directly
            result = await planner1.delegate_to_planner("Design a complex authentication system with OAuth integration")

            # Verify the task was evaluated for complexity
            mock_evaluate.assert_called_once()

            # Verify the result
            assert result.success is True
            assert "delegated to sub-planner" in result.data

    @pytest.mark.asyncio
    async def test_delegation_decision_based_on_complexity(
        self,
        registry: InMemoryAgentRegistry,
        flexible_delegation_system: dict[str, str],
    ) -> None:
        """Test that delegation decisions are correctly made based on task complexity."""
        # Get agents
        architect = registry.get_agent(flexible_delegation_system["architect"])
        planner1 = registry.get_agent(flexible_delegation_system["planner1"])

        # Test architect delegation decisions
        with patch.object(
            type(architect),
            "analyze_task_complexity",
            side_effect=[
                TaskComplexity.SIMPLE,
                TaskComplexity.COMPLEX,
            ],
        ):
            # Test delegation for simple task
            simple_result = await architect.delegate_to_executor("Implement a simple login form")
            assert simple_result.success is True
            # The method now returns the executor's result directly
            assert simple_result.data == "Test response"

            # Test delegation for complex task
            complex_result = await architect.delegate_to_executor("Design a complex distributed system")
            assert complex_result.success is False
            # Check the error message - handle the case where data might be None
            if complex_result.data is not None:
                assert "too complex" in complex_result.data or "complex" in complex_result.data
            else:
                # If data is None, we just verify that the result indicates failure
                assert complex_result.error is not None

        # Test planner delegation decisions
        with patch.object(
            type(planner1),
            "evaluate_subtask_complexity",
            side_effect=[
                TaskComplexity.MODERATE,
                TaskComplexity.VERY_COMPLEX,
            ],
        ):
            # Test delegation for moderate task
            moderate_result = await planner1.delegate_to_executor("Implement a form validation function")
            assert moderate_result.success is True
            assert "delegated directly to executor" in moderate_result.data

            # Test delegation for very complex task
            very_complex_result = await planner1.delegate_to_planner("Design a complex authentication system")
            assert very_complex_result.success is True
            assert "delegated to sub-planner" in very_complex_result.data

    @pytest.mark.asyncio
    async def test_end_to_end_flexible_delegation_workflow(
        self,
        registry: InMemoryAgentRegistry,
        flexible_delegation_system: dict[str, str],
        mock_provider: MagicMock,
    ) -> None:
        """Test an end-to-end workflow with flexible delegation paths."""
        # Get agents
        architect = registry.get_agent(flexible_delegation_system["architect"])

        # Mock complexity analysis to create a mixed delegation pattern
        with patch.object(
            type(architect),
            "analyze_task_complexity",
            side_effect=[
                TaskComplexity.COMPLEX,  # For the main task - delegate to planner
                TaskComplexity.SIMPLE,  # For a subtask - delegate directly to executor
            ],
        ):
            # Set up mock responses for the workflow
            mock_provider.generate.side_effect = [
                "Breaking down task into components: UI (simple), API (complex)",  # Architect response
                "Delegating UI component directly to executor",  # Architect delegating simple task
                "UI component implemented",  # Executor response
                "Delegating API component to planner",  # Architect delegating complex task
                "Planning API implementation",  # Planner response
            ]

            # Create a task message with mixed complexity components
            task_message = HumanMessage(content="Build a web application with simple UI and complex API")

            # Process the task with the architect
            result = await architect.process(task_message)

            # Verify the result
            assert result.success is True

            # In a real implementation, we would verify the specific delegation paths
            # and check that the right agents were used for each component

    @pytest.mark.asyncio
    async def test_complete_delegation_flow(
        self,
        registry: InMemoryAgentRegistry,
        flexible_delegation_system: dict[str, str],
    ) -> None:
        """Test the complete delegation flow from architect through planners to executors.

        This test verifies that a complex task can be properly delegated through the entire
        hierarchy: Architect -> Planner -> Sub-Planner -> Executor, as well as direct paths
        like Architect -> Executor for simple tasks.
        """
        # Get agent IDs
        architect_id = flexible_delegation_system["architect"]
        planner1_id = flexible_delegation_system["planner1"]
        planner2_id = flexible_delegation_system["planner2"]
        executor1_id = flexible_delegation_system["executor1"]
        flexible_delegation_system["executor2"]
        executor3_id = flexible_delegation_system["executor3"]

        # Get agents
        architect = registry.get_agent(architect_id)
        planner1 = registry.get_agent(planner1_id)
        planner2 = registry.get_agent(planner2_id)

        # Test direct delegation from architect to executor
        with patch.object(
            type(architect),
            "analyze_task_complexity",
            return_value=TaskComplexity.SIMPLE,
        ):
            # Delegate a simple task directly to executor
            simple_task = "Implement a simple login form"
            result = await architect.delegate_to_executor(simple_task)

            # Verify the result
            assert result.success is True
            # The method now returns the executor's result directly
            assert result.data == "Test response"

        # Test delegation from architect to planner
        # First, we need to register the planner as a child of the architect
        registry.register_parent_child_relationship(architect_id, planner1_id)

        # Now test delegation to the planner
        complex_task = "Design a complex authentication system"
        result = await architect.delegate_to_child(planner1_id, complex_task)

        # Verify the result
        assert result.success is True
        assert "delegated to" in result.data

        # Test recursive delegation from planner to another planner
        # First, register the second planner as a child of the first planner
        registry.register_parent_child_relationship(planner1_id, planner2_id)

        # Test with a very complex task that should be delegated to another planner
        with patch.object(
            type(planner1),
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.VERY_COMPLEX,
        ):
            very_complex_task = "Design a distributed microservices architecture"
            result = await planner1.delegate_to_planner(very_complex_task)

            # Verify the result
            assert result.success is True
            assert "delegated to sub-planner" in result.data

        # Test delegation from planner to executor
        # First, register the executor as a child of the planner
        registry.register_parent_child_relationship(planner2_id, executor3_id)

        # Test with a moderate task that should be delegated to an executor
        with patch.object(
            type(planner2),
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.MODERATE,
        ):
            moderate_task = "Implement a database schema"
            result = await planner2.delegate_to_executor(moderate_task)

            # Verify the result
            assert result.success is True
            assert "delegated directly to executor" in result.data

        # Verify the complete delegation path by checking the parent-child relationships
        # Architect -> Planner1 -> Planner2 -> Executor3
        architect_children = registry.get_child_agents(architect_id)
        planner1_children = registry.get_child_agents(planner1_id)
        planner2_children = registry.get_child_agents(planner2_id)

        # Check that planner1 is a child of architect
        assert any(agent.get_agent_id() == planner1_id for agent in architect_children)

        # Check that planner2 is a child of planner1
        assert any(agent.get_agent_id() == planner2_id for agent in planner1_children)

        # Check that executor3 is a child of planner2
        assert any(agent.get_agent_id() == executor3_id for agent in planner2_children)

        # Also verify direct delegation path: Architect -> Executor1
        registry.register_parent_child_relationship(architect_id, executor1_id)
        architect_children = registry.get_child_agents(architect_id)
        assert any(agent.get_agent_id() == executor1_id for agent in architect_children)
