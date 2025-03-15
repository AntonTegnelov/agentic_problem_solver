# Agent Creation

This guide explains how to create and manage agents in the APS hierarchical agent system.

## Overview

The APS system allows for dynamic creation of specialized agents with different roles:

- **ArchitectAgent**: High-level system design and task decomposition
- **PlannerAgent**: Mid-level planning and task refinement
- **ExecutorAgent**: Low-level implementation and execution

Agents can be created both programmatically and dynamically by other agents during runtime.

## Creating Agents Programmatically

### Using Role-Based Factory Functions

The simplest way to create agents is using the role-based factory functions:

```python
from src.agent.agent_types import create_agent
from src.common_types.enums import AgentRole

# Create an architect agent
architect = create_agent(AgentRole.ARCHITECT)

# Create a planner agent with a parent
planner = create_agent(AgentRole.PLANNER, parent_id=architect.get_agent_id())

# Create an executor agent with a parent
executor = create_agent(AgentRole.EXECUTOR, parent_id=planner.get_agent_id())
```

### Using Role-Specific Factory Functions

You can also use role-specific factory functions:

```python
from src.agent.agent_types import create_architect_agent, create_planner_agent, create_executor_agent

# Create an architect agent
architect = create_architect_agent()

# Create a planner agent with a parent
planner = create_planner_agent(parent_id=architect.get_agent_id())

# Create an executor agent with a parent
executor = create_executor_agent(parent_id=planner.get_agent_id())
```

### Using the AgentCoordinator

For more advanced scenarios, you can use the `AgentCoordinator`:

```python
from src.agent.coordination import AgentCoordinator, InMemoryAgentRegistry
from src.common_types.enums import AgentRole

# Create registry and coordinator
registry = InMemoryAgentRegistry()
coordinator = AgentCoordinator(registry)

# Create an architect agent
architect = coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})

# Create a planner agent with a parent
planner = coordinator.create_agent_by_role(
    AgentRole.PLANNER,
    {"parent_id": architect.get_agent_id()}
)

# Create an executor agent with a parent
executor = coordinator.create_agent_by_role(
    AgentRole.EXECUTOR,
    {"parent_id": planner.get_agent_id()}
)
```

## Establishing Parent-Child Relationships

When creating agents with parent-child relationships, you need to register these relationships:

```python
# Create agents
architect = coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
planner = coordinator.create_agent_by_role(
    AgentRole.PLANNER,
    {"parent_id": architect.get_agent_id()}
)

# Register the parent-child relationship
registry.register_parent_child_relationship(
    architect.get_agent_id(),
    planner.get_agent_id()
)
```

## Dynamic Agent Creation

Agents can create other agents at runtime. This is typically done by the higher-level agents (Architect, Planner) creating lower-level agents to delegate tasks.

### Example: Architect Creating Planners

```python
from src.messages.creation import create_human_message

# Create an architect agent
architect = coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
registry.register_agent(architect)

# Send a message to the architect to create planners
message = create_human_message("Design a system with frontend and backend components")
result = await architect.process(message)

# The architect will analyze the task and may create planner agents
# for different components of the system
```

### Example: Planner Creating Executors

```python
# Assuming we have a planner agent
planner = coordinator.create_agent_by_role(AgentRole.PLANNER, {})
registry.register_agent(planner)

# Send a message to the planner to break down a task
message = create_human_message("Implement the user authentication module")
result = await planner.process(message)

# The planner will break down the task and may create executor agents
# for different parts of the implementation
```

## Resource Management

The system includes resource management to prevent excessive agent creation:

```python
# Configure resource limits in the coordinator
coordinator._resource_limits = {
    "max_agents": 10,                          # Maximum total agents
    "max_agents_per_role": {                   # Maximum agents per role
        AgentRole.ARCHITECT.value: 1,
        AgentRole.PLANNER.value: 3,
        AgentRole.EXECUTOR.value: 6
    },
    "max_children_per_agent": 5,               # Maximum children per agent
    "max_hierarchy_depth": 3                   # Maximum hierarchy depth
}
```

## Capability Discovery

The system supports discovering agent capabilities:

```python
# Create agents with different capabilities
architect = coordinator.create_agent_by_role(AgentRole.ARCHITECT, {})
planner = coordinator.create_agent_by_role(AgentRole.PLANNER, {})
executor = coordinator.create_agent_by_role(AgentRole.EXECUTOR, {})

# Discover capabilities
capabilities = coordinator.discover_capabilities()

# Find agents with specific capabilities
design_agents = registry.find_agents_by_capability("design")
planning_agents = registry.find_agents_by_capability("planning")
implementation_agents = registry.find_agents_by_capability("implementation")
```

## Hierarchical Operations

The system supports various hierarchical operations:

```python
# Get the parent of an agent
parent = registry.get_parent_agent(agent_id)

# Get the children of an agent
children = registry.get_child_agents(agent_id)

# Get the siblings of an agent
siblings = registry.get_sibling_agents(agent_id)

# Get the entire hierarchy starting from a root agent
hierarchy = registry.get_agent_hierarchy(root_agent_id)

# Get all root agents (agents without parents)
root_agents = registry.get_root_agents()

# Get all leaf agents (agents without children)
leaf_agents = registry.get_leaf_agents()

# Get all ancestors of an agent
ancestors = registry.get_ancestors(agent_id)

# Get all descendants of an agent
descendants = registry.get_descendants(agent_id)
```

## Best Practices

1. **Follow the Hierarchy**: Generally, follow the Architect → Planner → Executor hierarchy for complex tasks.

2. **Direct Delegation for Simple Tasks**: For simple tasks, Architect agents can delegate directly to Executor agents.

3. **Resource Management**: Configure appropriate resource limits to prevent excessive agent creation.

4. **Capability Matching**: Use capability discovery to find the most appropriate agents for specific tasks.

5. **Cleanup**: Unregister agents when they're no longer needed to free up resources.

6. **Validation**: Periodically validate the agent hierarchy to detect and repair issues.
