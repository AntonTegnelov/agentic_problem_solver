# Hierarchical Agent System

## Overview

The hierarchical agent system is a protocol-based architecture that organizes agents into a structured hierarchy with specialized roles. This design enables more effective task decomposition, delegation, and execution compared to a single-agent approach. The system follows a top-down structure where higher-level agents break down complex tasks and delegate them to specialized agents with specific capabilities.

## Agent Roles

The hierarchical system defines three primary agent roles:

### 1. Architect Agent (`ArchitectAgent`)

- **Purpose**: High-level system design and task decomposition
- **Responsibilities**:
  - Breaking down complex problems into major components
  - Making architectural decisions
  - Delegating component implementation to Planner Agents
  - Ensuring overall system coherence
- **Capabilities**: System design, architecture planning, component identification
- **Position in Hierarchy**: Top-level agent, parent to Planner Agents

### 2. Planner Agent (`PlannerAgent`)

- **Purpose**: Mid-level task refinement and planning
- **Responsibilities**:
  - Converting architectural components into detailed implementation plans
  - Breaking down components into specific tasks
  - Delegating implementation tasks to Executor Agents
  - Ensuring component-level coherence
- **Capabilities**: Implementation planning, task sequencing, technical specification
- **Position in Hierarchy**: Mid-level agent, child to Architect Agent, parent to Executor Agents

### 3. Executor Agent (`ExecutorAgent`)

- **Purpose**: Low-level task execution and implementation
- **Responsibilities**:
  - Implementing specific tasks according to plans
  - Writing code, tests, and documentation
  - Self-prompting until tasks are complete
  - Reporting implementation details back to Planner Agents
- **Capabilities**: Coding, testing, debugging, documentation
- **Position in Hierarchy**: Leaf-level agent, child to Planner Agent

## Hierarchical Structure

The agents are organized in a tree-like structure:

```
ArchitectAgent
├── PlannerAgent1
│   ├── ExecutorAgent1
│   └── ExecutorAgent2
└── PlannerAgent2
    ├── ExecutorAgent3
    └── ExecutorAgent4
```

This structure allows for:

- Clear separation of concerns
- Specialized focus at each level
- Parallel execution of tasks
- Effective management of complex problems

## Agent Registry and Coordination

The `AgentRegistry` protocol and its implementation (`InMemoryAgentRegistry`) provide the infrastructure for managing agent relationships and facilitating communication:

### Key Capabilities

- **Parent-Child Relationships**: Register, retrieve, and manage parent-child connections between agents
- **Hierarchical Queries**: Find parent, child, and sibling agents
- **Capability-Based Discovery**: Locate agents with specific capabilities
- **Agent Hierarchy Visualization**: Generate a complete view of the agent hierarchy

## Communication Flow

Communication in the hierarchical system follows both top-down and bottom-up patterns:

### Top-Down Communication (Task Delegation)

1. Architect Agent receives a high-level task
2. Architect breaks down the task into components and delegates to Planner Agents
3. Planner Agents refine components into specific tasks and delegate to Executor Agents
4. Executor Agents implement the specific tasks

### Bottom-Up Communication (Result Aggregation)

1. Executor Agents complete tasks and report results to their parent Planner Agents
2. Planner Agents aggregate results from Executor Agents and report component completion to Architect Agent
3. Architect Agent combines all component results into a complete solution

## Implementation Details

### Asynchronous Design with Synchronous Interface

The hierarchical agent system implements internal asynchronous processing while providing synchronous interfaces for ease of use:

- **Core Implementation**: Agent implementations (`ArchitectAgent`, `PlannerAgent`, `ExecutorAgent`) use async/await patterns internally
- **Synchronous Wrapper**: Factory methods create adapter classes (e.g., `SyncArchitectAgent`) that provide synchronous interfaces
- **Dual Processing**: Agents support both `process` (async) and `process_sync` (synchronous) methods

This approach allows for:

- Efficient asynchronous operation internally
- Simpler synchronous API for most use cases
- Compatibility with existing synchronous code

### Agent Creation

Agents are created using factory methods that ensure proper initialization with role-specific capabilities:

```python
# Creating agents with specific roles using factory methods
from src.agent.agent_types import create_architect_agent, create_planner_agent, create_executor_agent

# Create an architect agent (top level)
architect = create_architect_agent(provider=llm_provider)

# Create a planner agent with parent reference
planner = create_planner_agent(provider=llm_provider, parent_id=architect.get_agent_id())

# Create an executor agent with parent reference
executor = create_executor_agent(provider=llm_provider, parent_id=planner.get_agent_id())
```

The factory methods handle both setting up the agent with the correct configuration and creating the synchronous wrapper classes necessary for the API compatibility.

### Hierarchical Relationships

Relationships between agents are established through the registry:

```python
# Get the agent registry
from src.agent.coordination import InMemoryAgentRegistry
registry = InMemoryAgentRegistry()

# Register the agents
registry.register_agent(architect)
registry.register_agent(planner)
registry.register_agent(executor)

# Establishing parent-child relationships
registry.register_parent_child_relationship(architect.get_agent_id(), planner.get_agent_id())
registry.register_parent_child_relationship(planner.get_agent_id(), executor.get_agent_id())
```

### Task Processing

Each agent type processes tasks according to its role:

```python
# Create a message for the architect
from src.messages.creation import create_human_message
message = create_human_message("Design a system to manage user authentication")

# Process synchronously
architect_result = architect.process_sync(message)

# Get the first planner's task from the architect's result
planner_task = architect_result.get_child_tasks()[0]
planner_message = create_human_message(planner_task.description)
planner_result = planner.process_sync(planner_message)

# Get an executor task from the planner's result
executor_task = planner_result.get_child_tasks()[0]
executor_message = create_human_message(executor_task.description)
executor_result = executor.process_sync(executor_message)
```

## Agent State Management

Each agent maintains its state through a state management system:

- **AgentState**: Maintains the agent's current status, task history, and context
- **StateManager**: Provides persistence and retrieval capabilities for agent states
- **InMemoryStateManager**: Default implementation that maintains states in memory

Agent states allow for:

- Context preservation between interactions
- Task tracking and history
- Dependency management between tasks

## Task Types and Management

The system defines structured task types for effective delegation and tracking:

- **TaskComplexity**: Indicates the estimated complexity of a task (SIMPLE, MODERATE, COMPLEX)
- **TaskPriority**: Defines the priority level of tasks (LOW, MEDIUM, HIGH, CRITICAL)
- **TaskStatus**: Tracks the current status of tasks (PENDING, IN_PROGRESS, COMPLETED, FAILED)

These attributes help agents make informed decisions about task delegation, sequencing, and resource allocation.

## Execution and Self-Prompting

The `ExecutorAgent` includes self-prompting capabilities to iteratively work on tasks until completion:

- Task is broken down into implementation stages (PLANNING, IMPLEMENTING, TESTING, REFINING, FINALIZING)
- Agent prompts itself through each stage, tracking progress
- Verification steps ensure quality and completeness
- Execution continues until success criteria are met

## Benefits of Hierarchical Design

1. **Improved Problem Decomposition**: Complex problems are broken down systematically at multiple levels
2. **Specialized Expertise**: Each agent focuses on tasks aligned with its capabilities
3. **Parallel Processing**: Multiple components can be developed simultaneously
4. **Scalability**: The system can scale to handle problems of varying complexity
5. **Maintainability**: Clear separation of concerns makes the system easier to understand and modify

## Future Enhancements

The hierarchical agent system is designed to be extensible with planned enhancements:

1. **Dynamic Agent Creation**: Enabling agents to create new agents as needed
2. **Flexible Delegation Paths**: Supporting direct delegation from Architect to Executor for simple tasks
3. **Task Complexity Analysis**: Automatically determining the appropriate level for task delegation
4. **Recursive Planning**: Allowing Planner Agents to create sub-planners for complex components
5. **Result Aggregation**: Sophisticated mechanisms for combining results from multiple agents

## Integration with Other Systems

The hierarchical agent system integrates with other components of the architecture:

- **Message System**: For communication between agents
- **State Management**: For tracking agent status and task progress
- **Provider System**: For accessing LLM capabilities
- **Configuration System**: For customizing agent behavior

## Conclusion

The hierarchical agent system provides a structured approach to complex problem-solving through specialized roles and effective task delegation. The implementation balances asynchronous internal processing with synchronous interfaces, making it both powerful and easy to use in various contexts.
