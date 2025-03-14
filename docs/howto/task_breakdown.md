# Task Breakdown in APS

## Overview

The task breakdown functionality in APS allows agents to decompose complex problems into manageable tasks with clear descriptions, priorities, and dependencies. This guide explains how to use the task breakdown capabilities in your applications.

## Task Schema

Tasks in APS follow a standardized schema defined in `src/common_types/task_types.py`. Each task includes:

- **Description**: A clear explanation of what needs to be done
- **Priority**: Importance level (LOW, MEDIUM, HIGH, CRITICAL)
- **Complexity**: Difficulty level (SIMPLE, MODERATE, COMPLEX, VERY_COMPLEX)
- **Status**: Current state (PENDING, IN_PROGRESS, BLOCKED, COMPLETED, FAILED)
- **Dependencies**: Other tasks that must be completed first
- **Parent-Child Relationships**: Hierarchical task structure
- **Assignment Information**: Which agent role/ID is responsible
- **Metadata**: Additional custom information

## Using Task Breakdown with Different Agent Roles

Each agent role has specialized task breakdown capabilities:

### Architect Agent

The Architect Agent breaks down high-level problems into major architectural components:

```python
from src.agent.agent_types import create_architect_agent
from src.common_types.message_types import HumanMessage
from src.llm_providers.factory import create_provider

# Create an architect agent
provider = create_provider("openai")  # or any other supported provider
architect = create_architect_agent(provider=provider)

# Create a task description
task_description = "Build a task management system with user authentication"

# Process the task with the architect agent
message = HumanMessage(content=task_description)
result = await architect.process(message)

# Access the broken-down tasks
if result.success:
    tasks = result.data
    for task in tasks:
        print(f"Task: {task.description}")
        print(f"Complexity: {task.complexity}")
        print(f"Priority: {task.priority}")
        print("---")
```

### Planner Agent

The Planner Agent refines architectural components into specific implementation tasks:

```python
from src.agent.agent_types import create_planner_agent

# Create a planner agent
planner = create_planner_agent(provider=provider)

# Create a component description
component_description = "Implement user authentication module"

# Process the component with the planner agent
message = HumanMessage(content=component_description)
result = await planner.process(message)

# Access the implementation tasks
if result.success:
    tasks = result.data
    for task in tasks:
        print(f"Implementation Task: {task.description}")
        print(f"Complexity: {task.complexity}")
        print(f"Priority: {task.priority}")
        print("---")
```

### Executor Agent

The Executor Agent breaks down implementation tasks into specific coding tasks:

```python
from src.agent.agent_types import create_executor_agent

# Create an executor agent
executor = create_executor_agent(provider=provider)

# Create an implementation task
implementation_task = "Implement JWT authentication"

# Process the implementation task with the executor agent
message = HumanMessage(content=implementation_task)
result = await executor.process(message)

# Access the coding tasks
if result.success:
    tasks = result.data
    for task in tasks:
        print(f"Coding Task: {task.description}")
        print(f"Complexity: {task.complexity}")
        print(f"Priority: {task.priority}")
        print("---")
```

## Using the TaskBreakdownStep Directly

For more control, you can use the `TaskBreakdownStep` class directly:

```python
from src.agent.steps import TaskBreakdownStep
from src.common_types.enums import AgentRole
from src.common_types.task_types import TaskComplexity, TaskPriority
from src.agent.state.base import InMemoryStateManager

# Create a state manager
state_manager = InMemoryStateManager()

# Create a task breakdown step for a specific agent role
breakdown_step = TaskBreakdownStep(agent_role=AgentRole.ARCHITECT)

# Set the agent for the step
breakdown_step.set_agent(architect)

# Execute the task breakdown step
result = await breakdown_step(
    state=state_manager.get_state(),
    task_description="Build a task management system",
    complexity=TaskComplexity.COMPLEX,
    priority=TaskPriority.HIGH
)

# Access the broken-down tasks
if result.success:
    tasks = result.data
    for task in tasks:
        print(f"Task: {task.description}")
        print(f"Complexity: {task.complexity}")
        print(f"Priority: {task.priority}")
        print("---")
```

## Task Validation

APS includes validation utilities to ensure tasks conform to the schema:

```python
from src.utils.validation import validate_task, validate_task_list
from src.common_types.task_types import Task, TaskComplexity, TaskPriority, TaskStatus

# Create a task
task = Task(
    description="Implement user authentication",
    complexity=TaskComplexity.MODERATE,
    priority=TaskPriority.HIGH,
    status=TaskStatus.PENDING
)

# Validate a single task
is_valid, error = validate_task(task)
if not is_valid:
    print(f"Task validation error: {error}")

# Validate a list of tasks
tasks = [task, another_task]
is_valid, error = validate_task_list(tasks)
if not is_valid:
    print(f"Task list validation error: {error}")
```

## Task Serialization

For storing or transmitting tasks between agents, use the serialization utilities:

```python
from src.utils.serialization import (
    serialize_task,
    deserialize_task,
    serialize_task_list,
    deserialize_task_list
)

# Serialize a task to a dictionary
task_dict = serialize_task(task)

# Deserialize a dictionary back to a task
task_obj = deserialize_task(task_dict)

# Serialize a list of tasks to a JSON string
tasks_json = serialize_task_list([task1, task2])

# Deserialize a JSON string back to a list of tasks
task_list = deserialize_task_list(tasks_json)
```

## Hierarchical Task Breakdown

APS supports hierarchical task breakdown through the agent hierarchy:

1. **Architect Agent** breaks down the problem into major components
2. **Planner Agent** refines components into implementation tasks
3. **Executor Agent** breaks down implementation tasks into coding tasks

This hierarchical approach allows for effective management of complex problems:

```python
from src.agent.coordination import InMemoryAgentRegistry

# Create an agent registry
registry = InMemoryAgentRegistry()

# Register agents
registry.register_agent(architect)
registry.register_agent(planner)
registry.register_agent(executor)

# Set up hierarchy
registry.register_parent_child_relationship(architect.get_agent_id(), planner.get_agent_id())
registry.register_parent_child_relationship(planner.get_agent_id(), executor.get_agent_id())

# Process a high-level task with the architect
architect_message = HumanMessage(content="Build a task management system")
architect_result = await architect.process(architect_message)

# Get the architectural components
components = architect_result.data

# For each component, let the planner create implementation tasks
for component in components:
    planner_message = HumanMessage(content=f"Plan the implementation of {component.description}")
    planner_result = await planner.process(planner_message)

    # Get the implementation tasks
    implementation_tasks = planner_result.data

    # For each implementation task, let the executor create coding tasks
    for impl_task in implementation_tasks:
        executor_message = HumanMessage(content=f"Implement {impl_task.description}")
        executor_result = await executor.process(executor_message)

        # Get the coding tasks
        coding_tasks = executor_result.data
```

## Best Practices

1. **Task Granularity**: Ensure tasks are broken down to an appropriate level of detail
2. **Clear Descriptions**: Write task descriptions that are specific and actionable
3. **Proper Dependencies**: Accurately capture dependencies between tasks
4. **Complexity Assessment**: Assign appropriate complexity levels to guide delegation
5. **Priority Assignment**: Set clear priorities to guide execution order
6. **Validation**: Always validate tasks to ensure they conform to the schema
7. **State Management**: Store tasks in a state manager for persistence and retrieval

## Conclusion

The task breakdown functionality in APS provides a powerful way to decompose complex problems into manageable tasks. By leveraging the hierarchical agent system and specialized agent roles, you can create detailed task breakdowns that guide the implementation process from high-level architecture to specific coding tasks.
