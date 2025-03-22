"""Tests for improving coverage of the PlannerAgent class."""

import json
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import BaseMessage as Message
from langchain_core.messages import HumanMessage

from src.agent.agent_types.planner import PlannerAgent
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity, TaskDependency
from src.config.agent import AgentConfig
from src.llm_providers.interface import LLMProvider
from src.messages.creation import create_human_message


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self) -> None:
        """Initialize mock provider."""
        self.responses: dict[str, Any] = {}

    def set_response(self, prompt: str, response: dict[str, str | list[dict[str, Any]]]) -> None:
        """Set response for a prompt."""
        self.responses[prompt] = response

    async def generate(self, messages: str | list[HumanMessage], *, config: None = None) -> str:  # noqa: ARG002
        """Generate response."""
        # Ignoring config argument as it's not used in tests but required by protocol
        prompt = messages[0].content if isinstance(messages, list) else messages
        if prompt in self.responses:
            return json.dumps(self.responses[prompt])
        return json.dumps({"error": "No response set for prompt"})


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """Create mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def planner_agent(mock_provider: MockLLMProvider) -> PlannerAgent:
    """Create planner agent with mock provider."""
    config = AgentConfig()
    return PlannerAgent(provider=mock_provider, config=config, max_delegation_depth=3)


def test_evaluate_subtask_complexity_rule_based(planner_agent: PlannerAgent) -> None:
    """Test rule-based complexity evaluation."""
    # Test simple tasks
    assert planner_agent.evaluate_subtask_complexity("Simple task to print hello") == TaskComplexity.SIMPLE
    assert planner_agent.evaluate_subtask_complexity("Basic function to add numbers") == TaskComplexity.SIMPLE

    # Test very complex tasks
    assert planner_agent.evaluate_subtask_complexity("Very complex distributed system") == TaskComplexity.VERY_COMPLEX
    assert planner_agent.evaluate_subtask_complexity("Extremely complex AI model") == TaskComplexity.VERY_COMPLEX

    # Test complex tasks
    assert planner_agent.evaluate_subtask_complexity("Complex authentication system") == TaskComplexity.COMPLEX
    assert planner_agent.evaluate_subtask_complexity("Advanced database schema") == TaskComplexity.COMPLEX

    # Test moderate tasks
    assert planner_agent.evaluate_subtask_complexity("Moderate difficulty task") == TaskComplexity.MODERATE
    assert planner_agent.evaluate_subtask_complexity("Standard implementation") == TaskComplexity.MODERATE


def test_evaluate_subtask_complexity_llm_fallback(planner_agent: PlannerAgent, mock_provider: MockLLMProvider) -> None:
    """Test LLM fallback for complexity evaluation."""
    # Set up mock response for a task that won't match rule-based patterns
    task_description = "Implement feature XYZ with consideration for future extensibility"
    mock_provider.set_response(
        f"Evaluate complexity of: {task_description}",
        {"complexity": "COMPLEX"},
    )

    # Test with task that needs LLM evaluation
    result = planner_agent.evaluate_subtask_complexity(task_description)
    assert result == TaskComplexity.COMPLEX

    # Test with invalid LLM response
    mock_provider.set_response(
        "Evaluate complexity of: Another task",
        {"invalid": "response"},
    )
    result = planner_agent.evaluate_subtask_complexity("Another task")
    assert result == TaskComplexity.MODERATE  # Default fallback


async def test_delegate_to_planner(planner_agent: PlannerAgent) -> None:
    """Test delegation to another planner."""
    # Create a mock sub-planner
    sub_planner = PlannerAgent(
        provider=planner_agent.provider,
        config=planner_agent.config,
        max_delegation_depth=planner_agent.max_delegation_depth - 1,
    )

    # Mock the process method to return a successful result
    async def mock_process(_: Message) -> Result[str]:
        return type("Result", (), {"success": True, "data": "Task processed by sub-planner", "error": None})

    sub_planner.process = mock_process

    # Mock the _create_sub_planner method to return our mock sub-planner
    async def mock_create_sub_planner() -> PlannerAgent:
        return sub_planner

    planner_agent._create_sub_planner = mock_create_sub_planner

    # Create a properly structured message
    task_message = create_human_message(content="Complex task to delegate")

    # Test with mock provider
    result = await planner_agent.delegate_to_planner(task_message)
    assert result.success
    assert "Task delegated to sub-planner" in str(result.data)

    # Test delegation depth limit
    planner_agent._current_delegation_depth = planner_agent.max_delegation_depth
    result = await planner_agent.delegate_to_planner(task_message)
    assert not result.success
    assert "Maximum delegation depth" in str(result.error)


async def test_delegate_to_child(planner_agent: PlannerAgent) -> None:
    """Test delegation to child agent."""
    child_id = "test_child"
    planner_agent.add_child(child_id)

    # Create a mock child agent
    child_agent = PlannerAgent(
        provider=planner_agent.provider,
        config=planner_agent.config,
        max_delegation_depth=planner_agent.max_delegation_depth - 1,
    )

    # Set the agent ID
    child_agent.get_agent_id = lambda: child_id

    # Mock the process method to return a successful result
    async def mock_process(_: Message) -> Result[str]:
        return type("Result", (), {"success": True, "data": "Task processed by child", "error": None})

    child_agent.process = mock_process

    # Register the child agent in the state
    planner_agent.state._agents[child_id] = child_agent

    # Create a properly structured message
    task_message = create_human_message(content="Test task")

    # Test with non-existent child
    result = await planner_agent.delegate_to_child("invalid_child", task_message)
    assert not result.success
    assert "is not a child" in str(result.error)

    # Test with existing child
    result = await planner_agent.delegate_to_child(child_id, task_message)
    assert result.success
    assert "Task processed by child" in str(result.data)

    # Test delegation depth limit
    planner_agent._current_delegation_depth = planner_agent.max_delegation_depth
    result = await planner_agent.delegate_to_child(child_id, task_message)
    assert not result.success
    assert "Maximum delegation depth" in str(result.error)


def test_synchronize_dependent_tasks(planner_agent: PlannerAgent) -> None:
    """Test task synchronization."""
    # Create test tasks
    task1 = Task(task_id=uuid4(), description="Task 1")
    task2 = Task(task_id=uuid4(), description="Task 2")
    task3 = Task(task_id=uuid4(), description="Task 3")

    # Add dependencies
    task2.dependencies = [TaskDependency(task_id=task1.task_id, description="Depends on task 1")]
    task3.dependencies = [TaskDependency(task_id=task2.task_id, description="Depends on task 2")]

    # Test synchronization
    batches = planner_agent.synchronize_dependent_tasks([task1, task2, task3])
    assert len(batches) == 3  # Should be three batches due to dependencies
    assert task1 in batches[0]  # First batch should contain task1
    assert task2 in batches[1]  # Second batch should contain task2
    assert task3 in batches[2]  # Third batch should contain task3

    # Test empty input
    assert planner_agent.synchronize_dependent_tasks([]) == []

    # Test circular dependency
    task1.dependencies = [TaskDependency(task_id=task3.task_id, description="Circular dependency")]
    batches = planner_agent.synchronize_dependent_tasks([task1, task2, task3])
    assert len(batches) > 0  # Should still produce batches despite circular dependency


async def test_execute_synchronized_tasks(planner_agent: PlannerAgent) -> None:
    """Test execution of synchronized tasks."""
    # Create test tasks
    task1 = Task(task_id=uuid4(), description="Task 1")
    task2 = Task(task_id=uuid4(), description="Task 2")

    # Add to state
    planner_agent.state.add_task(task1)
    planner_agent.state.add_task(task2)

    # Test execution
    results, errors = await planner_agent.execute_synchronized_tasks([task1, task2])
    assert isinstance(results, dict)
    assert isinstance(errors, list)

    # Test empty input
    results, errors = await planner_agent.execute_synchronized_tasks([])
    assert results == {}
    assert errors == []


def test_analyze_task_dependencies(planner_agent: PlannerAgent, mock_provider: MockLLMProvider) -> None:
    """Test task dependency analysis."""
    # Create test tasks
    task1 = Task(task_id=uuid4(), description="Create database schema")
    task2 = Task(task_id=uuid4(), description="Implement database queries")

    # Set up mock LLM response
    mock_provider.set_response(
        "Analyze dependencies between these tasks:\n"
        f"- {task1.task_id}: {task1.description}\n"
        f"- {task2.task_id}: {task2.description}\n"
        "Return a JSON object with 'dependencies' key containing a list of task dependencies.",
        {"dependencies": [{"task_id": task1.task_id, "dependent_task_ids": [str(task2.task_id)]}]},
    )

    # Test with LLM
    dependencies = planner_agent.analyze_task_dependencies([task1, task2])
    assert len(dependencies) == 1
    assert dependencies[0]["task_id"] == str(task1.task_id)

    # Test fallback to rule-based approach
    mock_provider.set_response(
        "Analyze dependencies between these tasks:\n"
        f"- {task1.task_id}: {task1.description}\n"
        f"- {task2.task_id}: {task2.description}\n"
        "Return a JSON object with 'dependencies' key containing a list of task dependencies.",
        {"invalid": "response"},
    )
    dependencies = planner_agent.analyze_task_dependencies([task1, task2])
    assert isinstance(dependencies, list)

    # Test empty input
    assert planner_agent.analyze_task_dependencies([]) == []


def test_estimate_task_completion_time(planner_agent: PlannerAgent) -> None:
    """Test task completion time estimation."""
    # Create test tasks with different complexities
    simple_task = Task(task_id=uuid4(), description="Simple task", complexity=TaskComplexity.SIMPLE)
    complex_task = Task(task_id=uuid4(), description="Complex task", complexity=TaskComplexity.COMPLEX)

    # Test base estimates
    assert planner_agent.estimate_task_completion_time(simple_task) == 30  # 30 minutes
    assert planner_agent.estimate_task_completion_time(complex_task) == 360  # 6 hours

    # Test with dependencies
    complex_task.dependencies = [TaskDependency(task_id=uuid4(), description="Dependency")]
    time_with_dep = planner_agent.estimate_task_completion_time(complex_task)
    assert time_with_dep > 360  # Should increase with dependency

    # Test with subtasks
    complex_task.subtasks = [Task(task_id=uuid4(), description="Subtask")]
    time_with_subtask = planner_agent.estimate_task_completion_time(complex_task)
    assert time_with_subtask > time_with_dep  # Should increase with subtask
