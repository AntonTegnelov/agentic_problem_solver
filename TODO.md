## TODO

- Implement protocol-based hierarchical agent system with specialized agent roles

  - [x] Extend `src/agent/agent_types/agent_types.py` to add hierarchical methods to the `Agent` protocol
  - [x] Add agent role enums in `src/common_types/enums.py` (ARCHITECT, PLANNER, EXECUTOR)
  - [x] Enhance `AgentInfo` in `src/common_types/agent_types.py` to include parent/child relationship tracking
  - [x] Add deprecation warning to `SolverAgent` in `src/agent/solver.py` while maintaining compatibility
  - [x] Create `ArchitectAgent` in `src/agent/agent_types/architect.py` for high-level task decomposition
  - [x] Create `PlannerAgent` in `src/agent/agent_types/planner.py` for mid-level task refinement
  - [x] Create `ExecutorAgent` in `src/agent/agent_types/executor.py` for low-level task execution
  - [x] Add agent factory methods in `src/agent/agent_types/__init__.py` for role-based instantiation
  - [x] Update `AgentRegistry` protocol in `src/agent/agent_types/agent_types.py` with hierarchical query methods
  - [x] Create unit tests in `tests/unit/test_specialized_agents.py` for the new agent types
  - [x] Create integration tests in `tests/integration/test_agent_hierarchy.py` for multi-tier workflow
  - [x] Update documentation in `docs/explanation/hierarchical_agents.md` with architecture details

- enable the ai to break down the prompt into TODO lists following a standardized format

  - [x] Create a standardized task schema in `src/common_types/task_types.py` with fields for description, priority, dependencies, and status
  - [x] Add task complexity estimation fields to the schema to support delegation decisions
  - [x] Extend `src/agent/steps.py` to add a new `TaskBreakdownStep` class for task decomposition
  - [x] Add prompt templates in `src/prompts/templates.py` for architectural breakdown, planning, and execution
  - [x] Add specialized prompt for `ArchitectAgent` in `src/prompts/templates.py` focused on system design
  - [ ] Add specialized prompt for `PlannerAgent` in `src/prompts/templates.py`
  - [ ] Add specialized prompt for `ExecutorAgent` in `src/prompts/templates.py`
  - [ ] Implement validation logic in `src/utils/validation.py` to ensure generated tasks conform to the schema
  - [ ] Add dependency tracking functionality in `src/agent/state/base.py` to manage task relationships
  - [ ] Create task serialization utilities in `src/utils/serialization.py` for task interchange between agents
  - [x] Create unit tests in `tests/unit/test_task_breakdown.py` to verify breakdown functionality
  - [ ] Add integration tests in `tests/integration/test_task_workflow.py` to verify end-to-end workflow
  - [ ] Update documentation in `docs/howto/task_breakdown.md` with usage examples
  - [ ] Ensure CI/CD pipeline validates the new functionality

- enable flexible delegation paths between agent types

  - [ ] Implement task complexity analysis logic in `src/agent/agent_types/architect.py` for delegation decisions
  - [ ] Add direct delegation capability from `ArchitectAgent` to `ExecutorAgent` for simple tasks
  - [ ] Implement sub-task complexity evaluation in `src/agent/agent_types/planner.py`
  - [ ] Add recursive delegation capability from `PlannerAgent` to additional `PlannerAgent` instances for complex sub-components
  - [ ] Enhance `AgentCoordinator` in `src/agent/coordination.py` to support flexible delegation paths
  - [ ] Update agent creation methods to support dynamic parent-child relationships
  - [ ] Implement capability matching in `src/agent/coordination.py` to route tasks to appropriate agent types
  - [ ] Create decision logging in each agent type to track delegation choices
  - [ ] Add unit tests in `tests/unit/test_flexible_delegation.py` to verify delegation logic
  - [ ] Create integration tests that verify direct and recursive delegation patterns
  - [ ] Update documentation in `docs/howto/delegation_strategies.md` with delegation decision guidelines

- enable the top agent to create other agents by itself, as a tool

  - [ ] Update `AgentCoordinator` in `src/agent/coordination.py` to support role-based agent creation
  - [ ] Enhance `InMemoryAgentRegistry` in `src/agent/coordination.py` to support parent-child relationships
  - [ ] Add parent-child agent registration methods to `AgentRegistry` implementations
  - [ ] Update the message system in `src/messages/utils.py` to include sender/receiver hierarchy information
  - [ ] Add resource management functionality in `src/agent/coordination.py` to prevent excessive agent creation
  - [ ] Create an agent capability discovery mechanism in `src/agent/coordination.py`
  - [ ] Add unit tests in `tests/unit/test_agent_creation.py` to verify agent creation workflow
  - [ ] Create integration tests in `tests/integration/test_dynamic_agents.py` to verify agent creation workflow
  - [ ] Update documentation in `docs/howto/agent_creation.md` with usage examples
  - [ ] Ensure CI/CD pipeline validates the new functionality

- combine the two approaches and make the top agent break down the problem into TODO lists and delegate them to new agents

  - [ ] Extend `AgentCoordinator` in `src/agent/coordination.py` to implement hierarchical task delegation
  - [ ] Implement capability-based task routing in `src/agent/coordination.py` to match tasks to appropriate agents
  - [ ] Add task complexity evaluation to support flexible delegation decisions
  - [ ] Implement progress tracking in `src/agent/state/base.py` to monitor delegated tasks
  - [ ] Add message routing capabilities in `src/messages/routing.py` for hierarchical message delivery
  - [ ] Add result aggregation functionality in `src/
