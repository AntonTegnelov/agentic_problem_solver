"""Tests for task result aggregation and agent registry functionality."""

import uuid
from unittest.mock import MagicMock

import pytest

from src.agent.coordination import InMemoryAgentRegistry, TaskResultAggregator
from src.common_types.agent_types import AgentInfo
from src.common_types.task_types import Task, TaskStatus


class MockAgentState:
    """Mock agent state for testing."""

    def __init__(self) -> None:
        """Initialize the mock agent state."""
        self.get_tasks = MagicMock(return_value=[])
        self.get_all_tasks = MagicMock(return_value=[])
        self.get_task_by_id = MagicMock(return_value=None)


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        agent_id: str | None = None,
        role: str = "tester",
        capabilities: list[str] | None = None,
    ) -> None:
        """Initialize the mock agent."""
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = role
        self.capabilities = capabilities or []
        self.state = MockAgentState()
        self.parent_id = None
        self.child_ids = []

    def get_agent_id(self) -> str:
        """Get agent ID."""
        return self.agent_id

    def get_state(self) -> MockAgentState:
        """Get agent state."""
        return self.state

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities."""
        return self.capabilities

    def get_parent_id(self) -> str | None:
        """Get parent agent ID."""
        return self.parent_id

    def get_child_ids(self) -> list[str]:
        """Get child agent IDs."""
        return self.child_ids

    def add_child(self, child_id: str) -> None:
        """Add a child agent."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)

    def remove_child(self, child_id: str) -> None:
        """Remove a child agent."""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)

    def set_parent(self, parent_id: str) -> None:
        """Set parent agent ID."""
        self.parent_id = parent_id

    def set_parent_id(self, parent_id: str) -> None:
        """Set parent agent ID (alias for compatibility)."""
        self.set_parent(parent_id)

    def clear_parent(self) -> None:
        """Clear parent agent ID."""
        self.parent_id = None


class TestTaskResultAggregator:
    """Test task result aggregator."""

    @pytest.fixture
    def registry(self) -> InMemoryAgentRegistry:
        """Create InMemoryAgentRegistry fixture."""
        return InMemoryAgentRegistry()

    @pytest.fixture
    def agent(self) -> MockAgent:
        """Create agent fixture."""
        return MockAgent()

    @pytest.fixture
    def aggregator(self, registry: InMemoryAgentRegistry) -> TaskResultAggregator:
        """Create TaskResultAggregator fixture."""
        return TaskResultAggregator(registry)

    def test_collect_results_no_state(
        self,
        aggregator: TaskResultAggregator,
        registry: InMemoryAgentRegistry,
        agent: MockAgent,
    ) -> None:
        """Test collecting results when agent has no state."""
        agent.state = None
        registry.register_agent(agent)
        results = aggregator.collect_results(agent.agent_id)
        assert not results.success
        assert "Cannot collect results without parent state" in results.message

    def test_collect_results_no_tasks(
        self,
        aggregator: TaskResultAggregator,
        registry: InMemoryAgentRegistry,
        agent: MockAgent,
    ) -> None:
        """Test collecting results when there are no tasks."""
        agent.state.get_tasks.return_value = []
        registry.register_agent(agent)
        results = aggregator.collect_results(agent.agent_id)
        assert results.success
        assert "No results to collect" in results.message

    def test_merge_text_results(self, aggregator: TaskResultAggregator) -> None:
        """Test merging text results."""
        results = [
            {"task_id": str(uuid.uuid4()), "description": "Task 1", "result": "Text 1"},
            {"task_id": str(uuid.uuid4()), "description": "Task 2", "result": "Text 2"},
        ]

        result = aggregator.merge_results(results, "text")
        assert result.success
        assert isinstance(result.data, str)

    def test_merge_code_results(self, aggregator: TaskResultAggregator) -> None:
        """Test merging code results."""
        results = [
            {"task_id": str(uuid.uuid4()), "description": "Task 1", "result": "```python\ndef func1():\n    pass```"},
            {"task_id": str(uuid.uuid4()), "description": "Task 2", "result": "```python\ndef func2():\n    pass```"},
        ]

        result = aggregator.merge_results(results, "code")
        assert result.success
        # Check if the data structure matches the actual implementation
        assert isinstance(result.data, dict)
        assert "code_sections" in result.data

    def test_merge_json_results(self, aggregator: TaskResultAggregator) -> None:
        """Test merging JSON results."""
        results = [
            {"task_id": str(uuid.uuid4()), "description": "Task 1", "result": {"key1": "value1"}},
            {"task_id": str(uuid.uuid4()), "description": "Task 2", "result": {"key2": "value2"}},
        ]

        result = aggregator.merge_results(results, "json")
        assert result.success
        assert isinstance(result.data, dict)
        assert "Task 1" in result.data
        assert result.data["Task 1"]["key1"] == "value1"

    def test_merge_mixed_results(self, aggregator: TaskResultAggregator) -> None:
        """Test merging mixed results."""
        results = [
            {"task_id": str(uuid.uuid4()), "description": "Task 1", "result": "Text result"},
            {"task_id": str(uuid.uuid4()), "description": "Task 2", "result": {"key": "value"}},
            {"task_id": str(uuid.uuid4()), "description": "Task 3", "result": "```python\ndef func():\n    pass```"},
        ]

        result = aggregator.merge_results(results, "mixed")
        assert result.success
        assert isinstance(result.data, dict)
        assert "text" in result.data
        assert "json" in result.data

    def test_track_completion_status(
        self,
        aggregator: TaskResultAggregator,
        registry: InMemoryAgentRegistry,
        agent: MockAgent,
    ) -> None:
        """Test tracking completion status."""
        # Create mock tasks
        task1 = Task(description="Task 1")
        task1.status = TaskStatus.COMPLETED
        task1.result = "Result 1"

        task2 = Task(description="Task 2")
        task2.status = TaskStatus.IN_PROGRESS

        # Set up agent state
        agent.state.get_tasks.return_value = [
            {"task_id": str(task1.task_id), "status": task1.status.value},
            {"task_id": str(task2.task_id), "status": task2.status.value},
        ]
        agent.state.get_task_by_id.side_effect = lambda task_id: task1 if str(task_id) == str(task1.task_id) else task2
        registry.register_agent(agent)

        # Track completion status
        stats = aggregator.track_completion_status(agent.agent_id)
        assert stats.success
        assert "completed_count" in stats.data
        assert "completion_percentage" in stats.data
        assert stats.data["completed_count"] == 1
        assert stats.data["total_tasks"] == 2

    def test_track_subtask_completion_status(
        self,
        aggregator: TaskResultAggregator,
        registry: InMemoryAgentRegistry,
        agent: MockAgent,
    ) -> None:
        """Test tracking subtask completion status."""
        # Create mock tasks
        parent_task = Task(description="Parent Task")
        parent_task_id = str(parent_task.task_id)

        subtask1 = Task(description="Subtask 1")
        subtask1.status = TaskStatus.COMPLETED
        subtask1.parent_task_id = parent_task.task_id

        subtask2 = Task(description="Subtask 2")
        subtask2.status = TaskStatus.IN_PROGRESS
        subtask2.parent_task_id = parent_task.task_id

        # Set up agent state
        agent.state.get_tasks.return_value = [
            {"task_id": parent_task_id, "status": parent_task.status.value},
            {"task_id": str(subtask1.task_id), "status": subtask1.status.value, "parent_task_id": parent_task_id},
            {"task_id": str(subtask2.task_id), "status": subtask2.status.value, "parent_task_id": parent_task_id},
        ]

        agent.state.get_task_by_id.side_effect = lambda task_id: {
            str(parent_task.task_id): parent_task,
            str(subtask1.task_id): subtask1,
            str(subtask2.task_id): subtask2,
        }.get(str(task_id))

        registry.register_agent(agent)

        # Track completion status for parent task
        stats = aggregator.track_subtask_completion_status(agent.agent_id, parent_task_id)
        assert stats.success
        assert "status_summary" in stats.data
        assert "completion_percentage" in stats.data
        assert stats.data["status_summary"]["completed"] == 1


class TestInMemoryAgentRegistry:
    """Test InMemoryAgentRegistry."""

    @pytest.fixture
    def registry(self) -> InMemoryAgentRegistry:
        """Create InMemoryAgentRegistry fixture."""
        return InMemoryAgentRegistry()

    def test_register_agent(self, registry: InMemoryAgentRegistry) -> None:
        """Test registering an agent."""
        agent = MockAgent()
        registry.register_agent(agent)
        assert agent.agent_id in registry._agents

    def test_unregister_agent(self, registry: InMemoryAgentRegistry) -> None:
        """Test unregistering an agent."""
        agent = MockAgent()
        registry.register_agent(agent)
        registry.unregister_agent(agent.agent_id)
        assert agent.agent_id not in registry._agents

    def test_list_agents(self, registry: InMemoryAgentRegistry) -> None:
        """Test listing agents."""
        agent1 = MockAgent(agent_id="agent1")
        agent2 = MockAgent(agent_id="agent2")

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        agents = registry.list_agents()
        assert len(agents) == 2
        assert any(a.agent_id == "agent1" for a in agents)
        assert any(a.agent_id == "agent2" for a in agents)

    def test_get_agents(self, registry: InMemoryAgentRegistry) -> None:
        """Test getting agents."""
        agent1 = MockAgent(agent_id="agent1")
        agent2 = MockAgent(agent_id="agent2")

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        agents = registry.get_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents

    def test_find_agents_by_capability(self, registry: InMemoryAgentRegistry) -> None:
        """Test finding agents by capability."""
        agent1 = MockAgent(agent_id="agent1", capabilities=["coding"])
        agent2 = MockAgent(agent_id="agent2", capabilities=["planning"])

        registry.register_agent(agent1)
        registry.register_agent(agent2)

        coding_agents = registry.find_agents_by_capability("coding")
        assert len(coding_agents) == 1
        assert any(a.agent_id == "agent1" for a in coding_agents)

    def test_find_agents_by_role(self, registry: InMemoryAgentRegistry) -> None:
        """Test finding agents by role."""
        agent1 = MockAgent(agent_id="agent1", role="executor")
        agent2 = MockAgent(agent_id="agent2", role="planner")

        # Create custom agent info to ensure role is set
        info1 = AgentInfo(
            agent_id="agent1",
            name="Agent 1",
            description="Test agent 1",
            capabilities=[],
            status="idle",
            parent_id=None,
            child_ids=[],
        )
        info1.role = "executor"

        info2 = AgentInfo(
            agent_id="agent2",
            name="Agent 2",
            description="Test agent 2",
            capabilities=[],
            status="idle",
            parent_id=None,
            child_ids=[],
        )
        info2.role = "planner"

        registry.register_agent(agent1, info1)
        registry.register_agent(agent2, info2)

        executor_agents = registry.find_agents_by_role("executor")
        assert len(executor_agents) == 1
        assert any(a.agent_id == "agent1" for a in executor_agents)

    def test_register_parent_child_relationship(self, registry: InMemoryAgentRegistry) -> None:
        """Test registering a parent-child relationship."""
        parent = MockAgent(agent_id="parent")
        child = MockAgent(agent_id="child")

        registry.register_agent(parent)
        registry.register_agent(child)

        registry.register_parent_child_relationship("parent", "child")
        assert "child" in parent.get_child_ids()
        assert child.get_parent_id() == "parent"

    def test_get_parent_agent(self, registry: InMemoryAgentRegistry) -> None:
        """Test getting a parent agent."""
        parent = MockAgent(agent_id="parent")
        child = MockAgent(agent_id="child")

        registry.register_agent(parent)
        registry.register_agent(child)

        registry.register_parent_child_relationship("parent", "child")
        parent_agent = registry.get_parent_agent("child")
        assert parent_agent.get_agent_id() == "parent"

    def test_get_child_agents(self, registry: InMemoryAgentRegistry) -> None:
        """Test getting child agents."""
        parent = MockAgent(agent_id="parent")
        child1 = MockAgent(agent_id="child1")
        child2 = MockAgent(agent_id="child2")

        registry.register_agent(parent)
        registry.register_agent(child1)
        registry.register_agent(child2)

        registry.register_parent_child_relationship("parent", "child1")
        registry.register_parent_child_relationship("parent", "child2")

        child_agents = registry.get_child_agents("parent")
        assert len(child_agents) == 2
        assert any(agent.get_agent_id() == "child1" for agent in child_agents)
        assert any(agent.get_agent_id() == "child2" for agent in child_agents)
