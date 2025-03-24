from unittest.mock import AsyncMock, Mock

import pytest

from src.agent.agent_types import ArchitectAgent, ExecutorAgent, PlannerAgent
from src.agent.coordination import AgentCoordinator, InMemoryAgentRegistry
from src.common_types.enums import TaskComplexity
from src.common_types.result_types import Result


@pytest.fixture
def agent_coordinator():
    registry = InMemoryAgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Create agents with minimal configuration
    architect = ArchitectAgent()
    planner = PlannerAgent()
    executor = ExecutorAgent()

    # Register the agents
    registry.register_agent(architect)
    registry.register_agent(planner)
    registry.register_agent(executor)

    return coordinator


async def test_complete_breakdown_to_execution_workflow(agent_coordinator) -> None:
    """Test the complete workflow from task breakdown to execution."""
    # Create a task
    task = "Implement user profile system"

    # Mock the architect's response
    mock_architect = Mock()
    mock_architect.get_agent_id.return_value = "architect-1"
    mock_architect.process = AsyncMock(
        return_value=Result(
            success=True,
            data={"subtasks": [{"description": "Design database schema", "complexity": TaskComplexity.COMPLEX}]},
        ),
    )

    # Mock the planner's response
    mock_planner = Mock()
    mock_planner.get_agent_id.return_value = "planner-1"
    mock_planner.process = AsyncMock(
        return_value=Result(
            success=True,
            data={"implementation_steps": ["Create tables", "Add indexes"]},
        ),
    )

    # Mock the executor's response
    mock_executor = Mock()
    mock_executor.get_agent_id.return_value = "executor-1"
    mock_executor.process = AsyncMock(
        return_value=Result(
            success=True,
            data={"completed_steps": ["Tables created", "Indexes added"]},
        ),
    )

    # Register mock agents with the coordinator's registry
    agent_coordinator._registry.register_agent(mock_architect)
    agent_coordinator._registry.register_agent(mock_planner)
    agent_coordinator._registry.register_agent(mock_executor)

    # Delegate the task
    result = await agent_coordinator.delegate_task("architect-1", task)

    # Verify the result
    assert result.success
    assert "Design database schema" in str(result.data)


async def test_complex_task_multiple_levels(agent_coordinator) -> None:
    """Test handling of complex tasks with multiple levels of breakdown."""
    task = "Build e-commerce platform"

    # Mock the architect's response with a complex hierarchical breakdown
    mock_architect = Mock()
    mock_architect.get_agent_id.return_value = "architect-1"
    mock_architect.process = AsyncMock(
        return_value=Result(
            success=True,
            data={
                "subtasks": [
                    {
                        "description": "Design system architecture",
                        "complexity": TaskComplexity.COMPLEX,
                        "subtasks": [
                            {
                                "description": "Define microservices architecture",
                                "complexity": TaskComplexity.MODERATE,
                                "dependencies": [],
                            },
                            {
                                "description": "Design API interfaces",
                                "complexity": TaskComplexity.COMPLEX,
                                "dependencies": ["Define microservices architecture"],
                            },
                            {
                                "description": "Plan scalability strategy",
                                "complexity": TaskComplexity.COMPLEX,
                                "dependencies": ["Define microservices architecture"],
                            },
                        ],
                    },
                    {
                        "description": "Plan database structure",
                        "complexity": TaskComplexity.COMPLEX,
                        "subtasks": [
                            {
                                "description": "Design data models",
                                "complexity": TaskComplexity.MODERATE,
                                "dependencies": [],
                            },
                            {
                                "description": "Plan database sharding",
                                "complexity": TaskComplexity.COMPLEX,
                                "dependencies": ["Design data models"],
                            },
                            {
                                "description": "Define indexing strategy",
                                "complexity": TaskComplexity.MODERATE,
                                "dependencies": ["Design data models"],
                            },
                        ],
                    },
                ],
            },
        ),
    )

    # Mock the planner's response with detailed implementation steps
    mock_planner = Mock()
    mock_planner.get_agent_id.return_value = "planner-1"
    mock_planner.process = AsyncMock(
        return_value=Result(
            success=True,
            data={
                "implementation_steps": [
                    {
                        "description": "Create microservices",
                        "subtasks": [
                            "Define service boundaries",
                            "Setup service communication",
                            "Implement service discovery",
                        ],
                    },
                    {
                        "description": "Set up databases",
                        "subtasks": [
                            "Create database schemas",
                            "Implement sharding logic",
                            "Setup indexes",
                        ],
                    },
                ],
                "dependencies": {
                    "Set up databases": ["Create microservices"],
                },
            },
        ),
    )

    # Mock the executor's response with detailed completion status
    mock_executor = Mock()
    mock_executor.get_agent_id.return_value = "executor-1"
    mock_executor.process = AsyncMock(
        return_value=Result(
            success=True,
            data={
                "completed_steps": [
                    {
                        "description": "Microservices created",
                        "details": [
                            "Service boundaries defined",
                            "Communication layer implemented",
                            "Service discovery operational",
                        ],
                        "status": "completed",
                    },
                    {
                        "description": "Databases configured",
                        "details": [
                            "Schemas created",
                            "Sharding implemented",
                            "Indexes optimized",
                        ],
                        "status": "completed",
                    },
                ],
                "execution_order": [
                    "Define service boundaries",
                    "Setup service communication",
                    "Implement service discovery",
                    "Create database schemas",
                    "Implement sharding logic",
                    "Setup indexes",
                ],
            },
        ),
    )

    # Register mock agents with the coordinator's registry
    agent_coordinator._registry.register_agent(mock_architect)
    agent_coordinator._registry.register_agent(mock_planner)
    agent_coordinator._registry.register_agent(mock_executor)

    # Delegate the task
    result = await agent_coordinator.delegate_task("architect-1", task)

    # Verify the result structure
    assert result.success
    assert isinstance(result.data, dict)

    # Verify architectural breakdown
    subtasks = result.data.get("subtasks", [])
    assert len(subtasks) == 2
    assert any(t["description"] == "Design system architecture" for t in subtasks)
    assert any(t["description"] == "Plan database structure" for t in subtasks)

    # Verify task dependencies
    arch_task = next(t for t in subtasks if t["description"] == "Design system architecture")
    assert len(arch_task["subtasks"]) == 3
    api_task = next(t for t in arch_task["subtasks"] if t["description"] == "Design API interfaces")
    assert "Define microservices architecture" in api_task["dependencies"]

    # Verify execution order respects dependencies
    execution_order = result.data.get("execution_order", [])
    if execution_order:
        service_boundary_idx = execution_order.index("Define service boundaries")
        service_comm_idx = execution_order.index("Setup service communication")
        assert service_boundary_idx < service_comm_idx, "Dependencies not respected in execution order"


async def test_parallel_execution_scenarios(agent_coordinator) -> None:
    """Test parallel execution of independent tasks."""
    task = "Implement payment processing"

    # Mock the architect's response
    mock_architect = Mock()
    mock_architect.get_agent_id.return_value = "architect-1"
    mock_architect.process = AsyncMock(
        return_value=Result(
            success=True,
            data={
                "subtasks": [
                    {"description": "Implement payment gateway", "complexity": TaskComplexity.COMPLEX},
                    {"description": "Set up security measures", "complexity": TaskComplexity.COMPLEX},
                ],
            },
        ),
    )

    # Mock the planner's response
    mock_planner = Mock()
    mock_planner.get_agent_id.return_value = "planner-1"
    mock_planner.process = AsyncMock(
        return_value=Result(
            success=True,
            data={
                "implementation_steps": [
                    "Integrate payment API",
                    "Add encryption",
                ],
            },
        ),
    )

    # Mock the executor's response
    mock_executor = Mock()
    mock_executor.get_agent_id.return_value = "executor-1"
    mock_executor.process = AsyncMock(
        return_value=Result(
            success=True,
            data={
                "completed_steps": [
                    "Payment API integrated",
                    "Encryption implemented",
                ],
            },
        ),
    )

    # Register mock agents with the coordinator's registry
    agent_coordinator._registry.register_agent(mock_architect)
    agent_coordinator._registry.register_agent(mock_planner)
    agent_coordinator._registry.register_agent(mock_executor)

    # Delegate tasks in parallel
    result = await agent_coordinator.delegate_task("architect-1", task)

    # Verify the result
    assert result.success
    assert "Implement payment gateway" in str(result.data)
