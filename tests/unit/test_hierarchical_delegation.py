"""Unit tests for hierarchical task delegation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agent.coordination import AgentCoordinator, InMemoryAgentRegistry
from src.common_types import AgentInfo
from src.common_types.error_types import AgentProcessingError
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity, TaskStatus


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        agent_id: str,
        capabilities: list[str],
        role: str | None = None,
        should_fail: bool = False,
    ) -> None:
        """Initialize mock agent.

        Args:
            agent_id: Agent ID.
            capabilities: List of agent capabilities.
            role: Optional agent role.
            should_fail: Whether agent should fail processing.

        """
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.role = role
        self.should_fail = should_fail
        self.processed_messages: list[HumanMessage] = []
        self._parent_id: str | None = None
        self._child_ids: list[str] = []
        self._state = MagicMock()
        self._state.get_agent_for_step.return_value = self

    async def process(self, message: HumanMessage) -> Result:
        """Process a message.

        Args:
            message: Message to process.

        Returns:
            Processing result.

        Raises:
            AgentProcessingError: If should_fail is True.

        """
        if self.should_fail:
            msg = f"Error processing message: {message.content}"
            raise AgentProcessingError(msg)
        self.processed_messages.append(message)
        return Result(success=True, data=f"Processed by {self.agent_id}", error="")

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        return self.agent_id

    def get_parent_id(self) -> str | None:
        """Get parent agent ID.

        Returns:
            Parent agent ID or None if no parent.

        """
        return self._parent_id

    def get_child_ids(self) -> list[str]:
        """Get child agent IDs.

        Returns:
            List of child agent IDs.

        """
        return self._child_ids.copy()

    def add_child(self, child_agent_id: str) -> None:
        """Add a child agent.

        Args:
            child_agent_id: Child agent ID to add.

        """
        if child_agent_id not in self._child_ids:
            self._child_ids.append(child_agent_id)

    def remove_child(self, child_agent_id: str) -> None:
        """Remove a child agent.

        Args:
            child_agent_id: Child agent ID to remove.

        """
        if child_agent_id in self._child_ids:
            self._child_ids.remove(child_agent_id)

    def set_parent(self, parent_agent_id: str) -> None:
        """Set parent agent.

        Args:
            parent_agent_id: Parent agent ID.

        """
        self._parent_id = parent_agent_id

    def clear_parent(self) -> None:
        """Clear parent agent reference."""
        self._parent_id = None

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        return self.capabilities

    def get_state(self) -> MagicMock:
        """Get agent state.

        Returns:
            Agent state.

        """
        return self._state


class TestHierarchicalDelegation:
    """Test hierarchical delegation functionality."""

    @pytest.mark.asyncio
    async def test_delegate_hierarchical_tasks_success(self) -> None:
        """Test successful hierarchical task delegation."""
        registry = InMemoryAgentRegistry()
        coordinator = AgentCoordinator(registry)

        # Create agents with different roles
        architect_agent = MockAgent("architect1", ["design", "architecture"], "architect")
        planner_agent = MockAgent("planner1", ["planning", "implementation"], "planner")
        executor_agent = MockAgent("executor1", ["coding", "testing"], "executor")

        # Set agent info with roles
        architect_info = AgentInfo(
            agent_id="architect1",
            name="Architect Agent",
            description="Handles high-level design",
            capabilities=["design", "architecture"],
        )
        # Add role as an attribute
        architect_info.role = "architect"

        planner_info = AgentInfo(
            agent_id="planner1",
            name="Planner Agent",
            description="Handles implementation planning",
            capabilities=["planning", "implementation"],
        )
        # Add role as an attribute
        planner_info.role = "planner"

        executor_info = AgentInfo(
            agent_id="executor1",
            name="Executor Agent",
            description="Handles code implementation",
            capabilities=["coding", "testing"],
        )
        # Add role as an attribute
        executor_info.role = "executor"

        # Register agents
        registry.register_agent(architect_agent, architect_info)
        registry.register_agent(planner_agent, planner_info)
        registry.register_agent(executor_agent, executor_info)

        # Create mock tasks
        task1 = Task(description="Task 1", complexity=TaskComplexity.SIMPLE)
        task2 = Task(description="Task 2", complexity=TaskComplexity.MODERATE)

        # Mock the TaskBreakdownStep
        with patch("src.agent.steps.TaskBreakdownStep") as mock_task_breakdown:
            # Create an AsyncMock for the instance that will be returned
            mock_instance = AsyncMock()
            # Configure the mock to return tasks when called
            mock_instance.return_value = Result(success=True, data=[task1, task2])
            # Make the mock_task_breakdown return our AsyncMock instance
            mock_task_breakdown.return_value = mock_instance

            # Mock the delegate_task_flexible method
            with patch.object(
                coordinator,
                "delegate_task_flexible",
                side_effect=[
                    Result(success=True, data={"result": "Task 1 delegated", "agent_id": "executor1"}),
                    Result(success=True, data={"result": "Task 2 delegated", "agent_id": "planner1"}),
                ],
            ):
                # Test hierarchical delegation
                result = await coordinator.delegate_hierarchical_tasks(
                    source_agent_id="architect1",
                    task="Design a system",
                )

                # Verify the result
                assert result.success
                assert "subtasks" in result.data
                assert "delegation_results" in result.data
                assert len(result.data["subtasks"]) == 2
                assert len(result.data["delegation_results"]) == 2

                # Verify that tasks were updated
                assert task1.status == TaskStatus.IN_PROGRESS
                assert task1.assigned_agent_id == "executor1"
                assert task2.status == TaskStatus.IN_PROGRESS
                assert task2.assigned_agent_id == "planner1"

    @pytest.mark.asyncio
    async def test_delegate_hierarchical_tasks_breakdown_failure(self) -> None:
        """Test hierarchical task delegation with breakdown failure."""
        registry = InMemoryAgentRegistry()
        coordinator = AgentCoordinator(registry)

        # Create and register architect agent
        architect_agent = MockAgent("architect1", ["design", "architecture"], "architect")
        architect_info = AgentInfo(
            agent_id="architect1",
            name="Architect Agent",
            description="Handles high-level design",
            capabilities=["design", "architecture"],
        )
        architect_info.role = "architect"
        registry.register_agent(architect_agent, architect_info)

        # Mock the TaskBreakdownStep
        with patch("src.agent.steps.TaskBreakdownStep") as mock_task_breakdown:
            # Create an AsyncMock for the instance that will be returned
            mock_instance = AsyncMock()
            # Configure the mock to fail when called
            mock_instance.return_value = Result(
                success=False,
                error="Failed to break down task",
            )
            # Make the mock_task_breakdown return our AsyncMock instance
            mock_task_breakdown.return_value = mock_instance

            # Test hierarchical delegation
            result = await coordinator.delegate_hierarchical_tasks(
                source_agent_id="architect1",
                task="Design a system",
            )

            # Verify the result
            assert not result.success
            assert "Failed to break down task" in result.error

    @pytest.mark.asyncio
    async def test_delegate_hierarchical_tasks_delegation_failure(self) -> None:
        """Test hierarchical task delegation with delegation failure."""
        registry = InMemoryAgentRegistry()
        coordinator = AgentCoordinator(registry)

        # Create and register architect agent
        architect_agent = MockAgent("architect1", ["design", "architecture"], "architect")
        architect_info = AgentInfo(
            agent_id="architect1",
            name="Architect Agent",
            description="Handles high-level design",
            capabilities=["design", "architecture"],
        )
        architect_info.role = "architect"
        registry.register_agent(architect_agent, architect_info)

        # Create mock tasks
        task1 = Task(description="Task 1", complexity=TaskComplexity.SIMPLE)

        # Mock the TaskBreakdownStep
        with patch("src.agent.steps.TaskBreakdownStep") as mock_task_breakdown:
            # Create an AsyncMock for the instance that will be returned
            mock_instance = AsyncMock()
            # Configure the mock to return tasks when called
            mock_instance.return_value = Result(success=True, data=[task1])
            # Make the mock_task_breakdown return our AsyncMock instance
            mock_task_breakdown.return_value = mock_instance

            # Mock the delegate_task_flexible method to fail
            with patch.object(
                coordinator,
                "delegate_task_flexible",
                return_value=Result(success=False, error="No suitable agent found"),
            ):
                # Test hierarchical delegation
                result = await coordinator.delegate_hierarchical_tasks(
                    source_agent_id="architect1",
                    task="Design a system",
                )

                # Verify the result
                assert result.success  # Overall process still succeeds
                assert "subtasks" in result.data
                assert "delegation_results" in result.data
                assert len(result.data["delegation_results"]) == 1
                assert not result.data["delegation_results"][0]["success"]
                assert "No suitable agent found" in result.data["delegation_results"][0]["error"]

                # Verify that task was updated
                assert task1.status == TaskStatus.FAILED
                assert task1.error == "No suitable agent found"

    @pytest.mark.asyncio
    async def test_delegate_hierarchical_tasks_agent_not_found(self) -> None:
        """Test hierarchical task delegation with agent not found."""
        registry = InMemoryAgentRegistry()
        coordinator = AgentCoordinator(registry)

        # Test with non-existent agent
        result = await coordinator.delegate_hierarchical_tasks(
            source_agent_id="nonexistent",
            task="Design a system",
        )

        # Verify the result
        assert not result.success
        assert "Agent not found" in result.error
