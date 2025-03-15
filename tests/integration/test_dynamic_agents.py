"""Integration tests for dynamic agent creation.

This module contains integration tests for the dynamic agent creation workflow,
testing how agents can create other agents at runtime and delegate tasks to them.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages.base import BaseMessage

from src.agent.agent_types import create_architect_agent, create_planner_agent
from src.agent.coordination import AgentCoordinator, InMemoryAgentRegistry
from src.common_types.enums import AgentRole
from src.common_types.message_types import HumanMessage, Message
from src.common_types.result_types import Result
from src.messages.creation import create_human_message


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider."""
    provider = MagicMock()

    async def mock_generate(messages: list[BaseMessage]) -> str:
        # Simulate different responses based on the message content
        content = messages[-1].content if messages else ""
        if "create" in content.lower() and "agent" in content.lower():
            return "I'll create a new agent to handle this task."
        if "delegate" in content.lower():
            return "I've delegated the task to the appropriate agent."
        return "Task completed successfully."

    async def mock_stream(messages: list[BaseMessage]) -> AsyncGenerator[str, None]:
        content = messages[-1].content if messages else ""
        if "create" in content.lower() and "agent" in content.lower():
            chunks = ["I'll ", "create ", "a new ", "agent ", "to handle ", "this task."]
        else:
            chunks = ["Task ", "completed ", "successfully."]

        for chunk in chunks:
            yield chunk

    # Set up the methods
    provider.generate = AsyncMock(side_effect=mock_generate)
    provider.generate_stream = mock_stream
    provider.__bool__.return_value = True
    return provider


@pytest.fixture
def registry() -> InMemoryAgentRegistry:
    """Create an InMemoryAgentRegistry instance."""
    return InMemoryAgentRegistry()


@pytest.fixture
def coordinator(registry: InMemoryAgentRegistry) -> AgentCoordinator:
    """Create an AgentCoordinator instance."""
    return AgentCoordinator(registry)


class TestDynamicAgentCreation:
    """Integration tests for dynamic agent creation."""

    @pytest.mark.asyncio
    async def test_architect_creates_planner(
        self,
        registry: InMemoryAgentRegistry,
        coordinator: AgentCoordinator,
        mock_provider: MagicMock,
    ) -> None:
        """Test that an architect agent can create a planner agent."""
        # Create an architect agent
        architect = create_architect_agent(provider=mock_provider)

        # Register the agent
        registry.register_agent(architect)

        # Set up the mock to simulate the architect creating a planner
        async def architect_creates_planner(messages: list[BaseMessage]) -> str:
            content = messages[-1].content if messages else ""
            if "create" in content.lower() and "planner" in content.lower():
                # Actually create a planner agent through the coordinator
                planner = coordinator.create_agent_by_role(
                    AgentRole.PLANNER,
                    {"parent_id": architect.get_agent_id()},
                )
                registry.register_parent_child_relationship(
                    architect.get_agent_id(),
                    planner.get_agent_id(),
                )
                return f"Created planner agent with ID: {planner.get_agent_id()}"
            return "Task completed successfully."

        mock_provider.generate = AsyncMock(side_effect=architect_creates_planner)

        # Create a custom process method that doesn't delegate to executor

        async def custom_process(message: Message) -> Result[str]:
            # Skip the complexity analysis and delegation logic
            # Just prepare messages and generate a response
            messages = architect._prepare_messages([message])
            response = await architect._generate_response(messages)
            return Result(success=True, data=str(response), error=None)

        # Patch the process method
        with patch.object(architect, "process", side_effect=custom_process):
            # Send a message to the architect to create a planner
            message = HumanMessage(content="Create a planner agent to handle the UI component")
            result = await architect.process(message)

            # Verify the result
            assert result.success is True
            assert "Created planner agent" in result.data

            # Verify that a planner was created and registered as a child of the architect
            children = architect.get_child_ids()
            assert len(children) == 1

            # Verify the parent-child relationship
            child_agent = registry.get_agent(children[0])
            assert child_agent.get_parent_id() == architect.get_agent_id()
            assert child_agent.get_role() == AgentRole.PLANNER.value

    @pytest.mark.asyncio
    async def test_planner_creates_executors(
        self,
        registry: InMemoryAgentRegistry,
        coordinator: AgentCoordinator,
        mock_provider: MagicMock,
    ) -> None:
        """Test that a planner agent can create multiple executor agents."""
        # Create an architect and a planner
        architect = create_architect_agent(provider=mock_provider)
        registry.register_agent(architect)

        planner = create_planner_agent(provider=mock_provider)
        planner.set_parent(architect.get_agent_id())
        registry.register_agent(planner)
        registry.register_parent_child_relationship(
            architect.get_agent_id(),
            planner.get_agent_id(),
        )

        # Set up the mock to simulate the planner creating executors
        async def planner_creates_executors(messages: list[BaseMessage]) -> str:
            content = messages[-1].content if messages else ""
            if "create" in content.lower() and "executor" in content.lower():
                # Create two executor agents
                executor1 = coordinator.create_agent_by_role(
                    AgentRole.EXECUTOR,
                    {"parent_id": planner.get_agent_id()},
                )
                executor2 = coordinator.create_agent_by_role(
                    AgentRole.EXECUTOR,
                    {"parent_id": planner.get_agent_id()},
                )

                registry.register_parent_child_relationship(
                    planner.get_agent_id(),
                    executor1.get_agent_id(),
                )
                registry.register_parent_child_relationship(
                    planner.get_agent_id(),
                    executor2.get_agent_id(),
                )

                return f"Created executor agents with IDs: {executor1.get_agent_id()} and {executor2.get_agent_id()}"
            return "Task completed successfully."

        mock_provider.generate = AsyncMock(side_effect=planner_creates_executors)

        # Send a message to the planner to create executors
        message = HumanMessage(
            content="Create executor agents to implement the login and registration components",
        )
        result = await planner.process(message)

        # Verify the result
        assert result.success is True
        assert "Created executor agents" in result.data

        # Verify that executors were created and registered as children of the planner
        children = planner.get_child_ids()
        assert len(children) == 2

        # Verify the parent-child relationships
        for child_id in children:
            child_agent = registry.get_agent(child_id)
            assert child_agent.get_parent_id() == planner.get_agent_id()
            assert child_agent.get_role() == AgentRole.EXECUTOR.value

    @pytest.mark.asyncio
    async def test_architect_creates_planners(
        self,
        registry: InMemoryAgentRegistry,
        coordinator: AgentCoordinator,
        mock_provider: MagicMock,
    ) -> None:
        """Test that an architect agent can create multiple planner agents."""
        # Create the root architect
        architect = create_architect_agent(provider=mock_provider)

        # Register the agent
        registry.register_agent(architect)

        # Set up the mock to simulate the architect creating planners
        async def create_planners(messages: list[BaseMessage]) -> str:
            content = messages[-1].content if messages else ""
            if "create" in content.lower() and "planner" in content.lower():
                # Create two planners
                planner1 = coordinator.create_agent_by_role(
                    AgentRole.PLANNER,
                    {"parent_id": architect.get_agent_id()},
                )
                planner2 = coordinator.create_agent_by_role(
                    AgentRole.PLANNER,
                    {"parent_id": architect.get_agent_id()},
                )

                registry.register_parent_child_relationship(
                    architect.get_agent_id(),
                    planner1.get_agent_id(),
                )
                registry.register_parent_child_relationship(
                    architect.get_agent_id(),
                    planner2.get_agent_id(),
                )

                return f"Created planner agents with IDs: {planner1.get_agent_id()} and {planner2.get_agent_id()}"
            return "Task completed successfully."

        mock_provider.generate = AsyncMock(side_effect=create_planners)

        # Create a custom process method that doesn't delegate to executor
        async def custom_process(message: Message) -> Result[str]:
            # Skip the complexity analysis and delegation logic
            # Just prepare messages and generate a response
            messages = architect._prepare_messages([message])
            response = await architect._generate_response(messages)
            return Result(success=True, data=str(response), error=None)

        # Patch the process method
        with patch.object(architect, "process", side_effect=custom_process):
            # Send a message to the architect to create planners
            message = HumanMessage(
                content="Create planner agents for the frontend and backend components",
            )
            result = await architect.process(message)

            # Verify the result
            assert result.success is True
            assert "Created planner agents" in result.data

            # Verify that planners were created
            children = architect.get_child_ids()
            assert len(children) == 2

            # Verify the hierarchy
            hierarchy = registry.get_agent_hierarchy(architect.get_agent_id())
            assert len(hierarchy[architect.get_agent_id()]) == 2

    @pytest.mark.asyncio
    async def test_planner_creates_executors_in_hierarchy(
        self,
        registry: InMemoryAgentRegistry,
        coordinator: AgentCoordinator,
        mock_provider: MagicMock,
    ) -> None:
        """Test that planners in a hierarchy can create executor agents."""
        # Create the root architect and two planners
        architect = create_architect_agent(provider=mock_provider)
        registry.register_agent(architect)

        planner1 = create_planner_agent(provider=mock_provider)
        planner1.set_parent(architect.get_agent_id())
        registry.register_agent(planner1)

        planner2 = create_planner_agent(provider=mock_provider)
        planner2.set_parent(architect.get_agent_id())
        registry.register_agent(planner2)

        registry.register_parent_child_relationship(
            architect.get_agent_id(),
            planner1.get_agent_id(),
        )
        registry.register_parent_child_relationship(
            architect.get_agent_id(),
            planner2.get_agent_id(),
        )

        # Set up the mock for planner1 to create executors
        async def planner1_creates_executors(messages: list[BaseMessage]) -> str:
            content = messages[-1].content if messages else ""
            if "create" in content.lower() and "executor" in content.lower():
                # Create two executors under planner1
                executor1 = coordinator.create_agent_by_role(
                    AgentRole.EXECUTOR,
                    {"parent_id": planner1.get_agent_id()},
                )
                executor2 = coordinator.create_agent_by_role(
                    AgentRole.EXECUTOR,
                    {"parent_id": planner1.get_agent_id()},
                )

                registry.register_parent_child_relationship(
                    planner1.get_agent_id(),
                    executor1.get_agent_id(),
                )
                registry.register_parent_child_relationship(
                    planner1.get_agent_id(),
                    executor2.get_agent_id(),
                )

                return f"Created executor agents with IDs: {executor1.get_agent_id()} and {executor2.get_agent_id()}"
            return "Task completed successfully."

        # Set up the mock for planner2 to create an executor
        async def planner2_creates_executor(messages: list[BaseMessage]) -> str:
            content = messages[-1].content if messages else ""
            if "create" in content.lower() and "executor" in content.lower():
                # Create one executor under planner2
                executor = coordinator.create_agent_by_role(
                    AgentRole.EXECUTOR,
                    {"parent_id": planner2.get_agent_id()},
                )

                registry.register_parent_child_relationship(
                    planner2.get_agent_id(),
                    executor.get_agent_id(),
                )

                return f"Created executor agent with ID: {executor.get_agent_id()}"
            return "Task completed successfully."

        # Test planner1 creating executors
        mock_provider.generate = AsyncMock(side_effect=planner1_creates_executors)
        message1 = HumanMessage(
            content="Create executor agents for the UI components",
        )
        result1 = await planner1.process(message1)

        # Verify the result
        assert result1.success is True
        assert "Created executor agents" in result1.data

        # Verify that executors were created under planner1
        planner1_children = planner1.get_child_ids()
        assert len(planner1_children) == 2

        # Test planner2 creating an executor
        mock_provider.generate = AsyncMock(side_effect=planner2_creates_executor)
        message2 = HumanMessage(
            content="Create an executor agent for the API component",
        )
        result2 = await planner2.process(message2)

        # Verify the result
        assert result2.success is True
        assert "Created executor agent" in result2.data

        # Verify that an executor was created under planner2
        planner2_children = planner2.get_child_ids()
        assert len(planner2_children) == 1

        # Verify the complete hierarchy
        hierarchy = registry.get_agent_hierarchy(architect.get_agent_id())
        assert len(hierarchy[architect.get_agent_id()]) == 2
        assert len(hierarchy[planner1.get_agent_id()]) == 2
        assert len(hierarchy[planner2.get_agent_id()]) == 1

    @pytest.mark.asyncio
    async def test_resource_limits_in_dynamic_creation(
        self,
        registry: InMemoryAgentRegistry,
        coordinator: AgentCoordinator,
        mock_provider: MagicMock,
    ) -> None:
        """Test that resource limits are enforced during dynamic agent creation."""
        # Create an architect agent
        architect = create_architect_agent(provider=mock_provider)

        # Register the agent
        registry.register_agent(architect)

        # Set a low maximum for children per agent
        coordinator._resource_limits["max_children_per_agent"] = 1

        # Set up the mock to simulate the architect trying to create multiple planners
        async def architect_creates_planners(_: list[BaseMessage]) -> str:
            try:
                # Try to create two planners (should fail on the second one)
                planner1 = coordinator.create_agent_by_role(
                    AgentRole.PLANNER,
                    {"parent_id": architect.get_agent_id()},
                )
                registry.register_parent_child_relationship(
                    architect.get_agent_id(),
                    planner1.get_agent_id(),
                )

                # This should raise a ValueError due to resource limits
                planner2 = coordinator.create_agent_by_role(
                    AgentRole.PLANNER,
                    {"parent_id": architect.get_agent_id()},
                )
                registry.register_parent_child_relationship(
                    architect.get_agent_id(),
                    planner2.get_agent_id(),
                )
            except ValueError as e:
                return f"Resource limit reached: {e!s}"
            else:
                # This will only execute if no exception is raised
                return "Created two planner agents"

        mock_provider.generate = AsyncMock(side_effect=architect_creates_planners)

        # Create a custom process method that doesn't delegate to executor
        async def custom_process(message: Message) -> Result[str]:
            # Skip the complexity analysis and delegation logic
            # Just prepare messages and generate a response
            messages = architect._prepare_messages([message])
            response = await architect._generate_response(messages)
            return Result(success=True, data=str(response), error=None)

        # Patch the process method
        with patch.object(architect, "process", side_effect=custom_process):
            # Send a message to the architect to create planners
            message = HumanMessage(
                content="Create planner agents for the frontend and backend components",
            )
            result = await architect.process(message)

            # Verify the result indicates resource limits were reached
            assert result.success is True
            assert "Resource limit reached" in result.data

            # Verify that only one planner was created
            children = architect.get_child_ids()
            assert len(children) == 1

    @pytest.mark.asyncio
    async def test_end_to_end_task_delegation_with_dynamic_agents(
        self,
        registry: InMemoryAgentRegistry,
        coordinator: AgentCoordinator,
        mock_provider: MagicMock,
    ) -> None:
        """Test end-to-end workflow where an agent creates another agent and delegates a task to it."""
        # Create an architect agent
        architect = create_architect_agent(provider=mock_provider)
        registry.register_agent(architect)

        # Set up the mock to simulate the architect creating a planner
        async def architect_response(messages: list[BaseMessage]) -> str:
            content = messages[-1].content if messages else ""
            if "create" in content.lower() and "agent" in content.lower():
                return "I'll create a planner agent to handle the database design task."
            return "Task completed successfully."

        mock_provider.generate = AsyncMock(side_effect=architect_response)

        # Send a message to the architect
        message = HumanMessage(
            content="Create a planner agent and delegate the database design task to it",
        )
        result = await architect.process(message)

        # Verify the architect's response
        assert result.success is True
        assert "create" in result.data.lower()
        assert "planner agent" in result.data.lower()

        # Now manually create a planner agent as if the architect had created it
        planner = create_planner_agent(provider=mock_provider)
        planner.set_parent(architect.get_agent_id())
        registry.register_agent(planner)
        registry.register_parent_child_relationship(
            architect.get_agent_id(),
            planner.get_agent_id(),
        )

        # Set up the mock for the planner's response
        async def planner_response(messages: list[BaseMessage]) -> str:
            content = messages[-1].content if messages else ""
            if "database" in content.lower() and "schema" in content.lower():
                return "Designed database schema with tables for users, products, and orders."
            return "Task completed successfully."

        mock_provider.generate = AsyncMock(side_effect=planner_response)

        # Delegate a task to the planner
        task_message = create_human_message("Design the database schema for the application")
        delegation_result = await coordinator.route_message(task_message, planner.get_agent_id())

        # Verify the delegation result
        assert delegation_result is not None
        assert delegation_result.success is True
        assert "database schema" in delegation_result.data.lower()

        # Verify the parent-child relationship
        children = architect.get_child_ids()
        assert len(children) == 1
        assert planner.get_agent_id() in children
        assert planner.get_parent_id() == architect.get_agent_id()
