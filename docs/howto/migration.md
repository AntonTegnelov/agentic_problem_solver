# Migrating from SolverAgent to the Hierarchical Agent System

## Overview

This guide provides a step-by-step migration path from the deprecated `SolverAgent` to the new hierarchical agent system. The hierarchical system offers improved task decomposition, specialized agent roles, and more effective problem-solving capabilities.

> **IMPORTANT**: The `SolverAgent` is deprecated and will be removed in a future version. All new development should use the hierarchical agent system.

## Migration Steps

### Step 1: Understand the Hierarchical Agent System

Before migrating, familiarize yourself with the hierarchical agent system:

- **ArchitectAgent**: Top-level agent for high-level task decomposition and system design
- **PlannerAgent**: Mid-level agent for detailed task planning and refinement
- **ExecutorAgent**: Bottom-level agent for implementing specific tasks

See the [Hierarchical Agents](../explanation/hierarchical_agents.md) documentation for detailed information.

### Step 2: Identify SolverAgent Usage

Identify all places in your codebase where `SolverAgent` is used:

```python
# Example of SolverAgent usage
from src.agent.solver import SolverAgent

solver = SolverAgent(provider=llm_provider)
result = solver.process("Implement a function to calculate factorial")
```

### Step 3: Replace with Appropriate Hierarchical Agents

Replace `SolverAgent` with the appropriate hierarchical agent based on your use case:

```python
# Example of hierarchical agent usage
from src.agent.agent_types import create_architect_agent

# For high-level tasks, use ArchitectAgent
architect = create_architect_agent(provider=llm_provider)
result = await architect.process(create_message(role="human", content="Design a system for calculating factorials"))

# For direct implementation tasks, you can still use ExecutorAgent
from src.agent.agent_types import create_executor_agent
executor = create_executor_agent(provider=llm_provider)
result = await executor.process(create_message(role="human", content="Implement a function to calculate factorial"))
```

### Step 4: Update Message Handling

The hierarchical agents use a more structured message system:

```python
# Old approach with SolverAgent
result = solver.process("Implement a function")

# New approach with hierarchical agents
from src.messages.creation import create_message
message = create_message(role="human", content="Implement a function")
result = await agent.process(message)
```

### Step 5: Update Result Handling

The hierarchical agents return `Result` objects asynchronously:

```python
# Old approach with SolverAgent
response = solver.process("Implement a function")
print(response)  # Direct string response

# New approach with hierarchical agents
result = await agent.process(message)
if result.success:
    print(result.data)  # Access data field of Result object
else:
    print(f"Error: {result.error}")
```

### Step 6: Implement Hierarchical Workflow (Optional)

For complex tasks, implement a full hierarchical workflow:

```python
from src.agent.agent_types import (
    create_architect_agent,
    create_planner_agent,
    create_executor_agent,
)
from src.agent.agent_types.agent_types import InMemoryAgentRegistry

# Create registry
registry = InMemoryAgentRegistry()

# Create agents
architect = create_architect_agent(provider=llm_provider)
planner = create_planner_agent(provider=llm_provider)
executor = create_executor_agent(provider=llm_provider)

# Register relationships
registry.register_agent(architect)
registry.register_agent(planner)
registry.register_agent(executor)
registry.register_parent_child_relationship(architect.get_agent_id(), planner.get_agent_id())
registry.register_parent_child_relationship(planner.get_agent_id(), executor.get_agent_id())

# Process task with architect (which will delegate to other agents)
result = await architect.process(create_message(role="human", content="Design and implement a factorial calculator"))
```

## API Differences

### SolverAgent vs. Hierarchical Agents

| SolverAgent                                    | Hierarchical Agents                                       | Notes                                  |
| ---------------------------------------------- | --------------------------------------------------------- | -------------------------------------- |
| `SolverAgent(provider, state_manager, config)` | `create_architect_agent(provider, state_manager, config)` | Factory functions for creating agents  |
| `solver.process(message)`                      | `await agent.process(message)`                            | Hierarchical agents use async methods  |
| `solver.process_stream(message)`               | `await agent.process_stream(message)`                     | Streaming is supported in both systems |
| Returns string or Result                       | Always returns Result                                     | More consistent return types           |
| Single agent handles everything                | Specialized agents with clear roles                       | Better separation of concerns          |
| No parent-child relationships                  | Hierarchical relationships                                | Enables complex workflows              |

### Message Handling Differences

| SolverAgent               | Hierarchical Agents          | Notes                                       |
| ------------------------- | ---------------------------- | ------------------------------------------- |
| Accepts string or Message | Requires Message objects     | More structured message handling            |
| Simple message flow       | Hierarchical message routing | Messages can flow up and down the hierarchy |
| No message metadata       | Rich message metadata        | Supports additional context in messages     |

## Migration Verification Checklist

- [ ] All `SolverAgent` imports replaced with hierarchical agent imports
- [ ] All `SolverAgent` instantiations replaced with appropriate factory functions
- [ ] All synchronous `process()` calls updated to use `await` with async methods
- [ ] All direct string responses updated to handle `Result` objects
- [ ] All tests updated to use hierarchical agents
- [ ] Verify functionality with hierarchical agents matches previous behavior
- [ ] Remove any remaining `SolverAgent` references

## Troubleshooting

### Common Issues

1. **Async/Await Errors**: Hierarchical agents use async methods, ensure your code properly uses `async`/`await`.
2. **Message Format Errors**: Ensure you're creating proper Message objects using `create_message()`.
3. **Result Handling**: Remember to check `result.success` and access data via `result.data`.
4. **Agent Selection**: Choose the right agent for the task (Architect for high-level, Planner for mid-level, Executor for implementation).

### Getting Help

If you encounter issues during migration, please:

1. Check the [Hierarchical Agents](../explanation/hierarchical_agents.md) documentation
2. Review the [API Reference](../reference/api.md) for detailed method signatures
3. Look at the example code in the `examples/` directory
4. File an issue in the project repository with details about your migration challenges
