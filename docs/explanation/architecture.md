# APS (Agentic Problem Solver) Architecture

## Overview

APS is designed as a modular agent-based system for solving complex programming problems. The system implements a hierarchical multi-agent architecture with specialized roles for effective task decomposition and execution.

## Current Architecture

The system is built around a protocol-based design with clear separation of concerns:

### Core Components

#### 1. Configuration System (`src/config/`)

- Each subsystem has its own config that extends the base configurations
- Environment-based configuration loading
- Type-safe configuration validation

#### 2. Agent System (`src/agent/`)

- `agent_types/`: Protocol definitions and specialized agent implementations
  - `agent_types.py`: Protocol definitions for agents
  - `architect.py`: Implementation of the top-level ArchitectAgent
  - `planner.py`: Implementation of the mid-level PlannerAgent
  - `executor.py`: Implementation of the bottom-level ExecutorAgent
- `state/`: Agent state management
  - `base.py`: Base state classes and interfaces
  - `memory.py`: Agent memory implementations
- `steps/`: Step execution framework
  - `base.py`: Step abstractions
  - `executors/`: Step execution implementations
- `coordination.py`: Agent coordination capabilities for hierarchical operations

#### 3. Provider System (`src/llm_providers/`)

- `base.py`: Provider interface definition
- `factory.py`: Provider instantiation logic
- Provider-specific implementations (e.g., `gemini.py`, `openai.py`)

#### 4. Message System (`src/messages/`)

- `base.py`: Message abstractions
- `handlers/`: Message processing logic
- `schemas/`: Message structure definitions
- `transformers/`: Message transformation utilities

#### 5. CLI System (`src/cli/`)

- Command-line interface for interacting with the system
- Configuration management

#### 6. Common Types (`src/common_types/`)

- Consolidates common types and enums

## Hierarchical Architecture

The current architecture implements a hierarchical multi-agent system with specialized roles:

### Hierarchical Agent Structure

1. **ArchitectAgent (Top Level)**

   - Responsible for high-level problem decomposition
   - Breaks down complex problems into independent microservices/components
   - Designs interfaces between components
   - Primarily delegates to PlannerAgents for further refinement
   - Can delegate directly to ExecutorAgents for simple, well-defined tasks that don't require further decomposition

2. **PlannerAgent (Middle Level)**

   - Receives component tasks from ArchitectAgent
   - Further decomposes components into implementable tasks
   - Manages dependencies between tasks
   - Can delegate to additional PlannerAgents for complex sub-components that require further specialized planning
   - Primarily delegates to ExecutorAgents for implementation

3. **ExecutorAgent (Bottom Level)**
   - Receives well-defined tasks with clear scope
   - Implements actual code solutions
   - Self-prompts through task execution stages until completion
   - Focuses on specific, manageable pieces of work
   - Reports results back up the hierarchy

### Delegation Decision Responsibility

The system implements agent-driven delegation decisions rather than using a separate delegation layer:

- **Agent-Driven Approach**: Each agent evaluates its assigned tasks and determines the appropriate delegation strategy

  - ArchitectAgents assess task complexity to decide between PlannerAgent or direct ExecutorAgent delegation
  - PlannerAgents evaluate sub-tasks to determine if further planning is needed or if they're ready for execution
  - This approach keeps decision-making close to the context where it's most relevant

- **Benefits of Agent-Driven Delegation**:

  - Maintains agent autonomy within the system
  - Leverages the contextual understanding each agent has of its specific tasks
  - Simplifies the architecture by avoiding an additional coordination layer
  - Allows for specialized delegation strategies per agent type

- **Coordination Support**:
  - While agents make delegation decisions, the coordination system provides supporting infrastructure
  - Resource management to prevent excessive agent creation
  - Capability matching to ensure tasks are assigned to appropriate agents
  - Load balancing considerations for optimal task distribution

### Protocol-Based Implementation

The hierarchical system is implemented using protocols rather than inheritance:

- **Agent Protocol**: All agent types implement the same `Agent` protocol defined in `agent_types.py`
- **Specialized Implementations**: Each agent type has its own implementation file in the `agent/agent_types/` directory
- **Asynchronous Design**: The agent implementations use asynchronous patterns internally while providing synchronous interfaces through wrapper classes

### Enhanced Coordination System

The coordination system supports hierarchical operations:

- **Parent-Child Relationships**: Track relationships between agents in the hierarchy
- **Task Delegation**: Route tasks to appropriate agent types based on capabilities
- **Result Aggregation**: Combine results from multiple child agents
- **Error Propagation**: Escalate errors to appropriate parent agents

### Task Breakdown and Management

A standardized task system:

- **Task Schema**: Defined structure for tasks with description, priority, dependencies, and status
- **Task Breakdown**: Specialized steps for decomposing complex tasks
- **Task Tracking**: Monitor progress of delegated tasks throughout the hierarchy

## Hierarchical Data Flow

1. User Input → CLI
2. CLI → ArchitectAgent
3. ArchitectAgent:
   - Breaks down problem into components
   - Analyzes each component's complexity
   - For complex components: Creates PlannerAgents and delegates component tasks
   - For simple components: Delegates directly to ExecutorAgents
4. PlannerAgents:
   - Further decompose components into implementable tasks
   - For complex sub-components: Create additional PlannerAgents for specialized planning
   - For implementable tasks: Create ExecutorAgents and delegate implementation
5. ExecutorAgents:
   - Implement assigned tasks
   - Report results back to parent agent (either PlannerAgent or ArchitectAgent)
6. PlannerAgents aggregate results from their ExecutorAgents and sub-PlannerAgents
7. ArchitectAgent combines all component results into final solution
8. Final result → User

## Key Design Principles

The system adheres to these principles in both current and planned architectures:

1. **Protocol-Based Design**: Using protocols rather than inheritance for flexible composition
2. **Single Responsibility**: Each component has one clear purpose
3. **Interface Segregation**: Clean interfaces between components
4. **Dependency Inversion**: High-level modules don't depend on low-level modules
5. **Open/Closed**: Extend functionality without modifying existing code
6. **DRY**: No code duplication across modules

## Configuration Management

- Hierarchical configuration system
- Environment-based overrides
- Type-safe configurations
- Validation at load time
- Sensible defaults

## Error Handling

- Consistent error types
- Proper error propagation
- Retry mechanisms
- Graceful degradation
- Detailed logging
- Planned: Hierarchical error escalation

## Testing Strategy

- Unit tests for each component
- Integration tests for workflows
- End-to-end tests for critical paths
- Performance benchmarks
- Stress testing
- Planned: Hierarchical agent interaction tests

## Monitoring and Observability

- Structured logging
- Performance metrics
- Error tracking
- Resource usage monitoring
- Task progress tracking
- Planned: Hierarchical task monitoring

## Security Considerations

- API key management
- Rate limiting
- Input validation
- Output sanitization
- Secure configuration handling

## Implementation Roadmap

The transition to the hierarchical architecture will follow these key steps:

1. Extend the Agent protocol with hierarchical capabilities
2. Create specialized agent implementations (Architect, Planner, Executor)
3. Enhance coordination system for hierarchical relationships
4. Implement standardized task schema and breakdown
5. Add hierarchical message routing
6. Create result aggregation mechanisms
7. Implement hierarchical error handling
8. Update CLI for hierarchical operations
9. Enhance documentation and testing

Look at docs/todos/TODO.md for a more detailed roadmap
