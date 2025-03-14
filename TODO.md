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
  - [x] Add specialized prompt for `PlannerAgent` in `src/prompts/templates.py`
  - [x] Add specialized prompt for `ExecutorAgent` in `src/prompts/templates.py`
  - [x] Implement validation logic in `src/utils/validation.py` to ensure generated tasks conform to the schema
  - [x] Add dependency tracking functionality in `src/agent/state/base.py` to manage task relationships
  - [x] Create task serialization utilities in `src/utils/serialization.py` for task interchange between agents
  - [x] Create unit tests in `tests/unit/test_task_breakdown.py` to verify breakdown functionality
  - [x] Add integration tests in `tests/integration/test_task_workflow.py` to verify end-to-end workflow
  - [x] Update documentation in `docs/howto/task_breakdown.md` with usage examples
  - [x] Ensure CI/CD pipeline validates the new functionality

- implement a safe migration path from SolverAgent to the hierarchical agent system

  - [x] Create migration documentation in `docs/howto/migration.md`
    - [x] Outline a clear, step-by-step migration path for users
    - [x] Document API differences between SolverAgent and hierarchical system
    - [x] Include a checklist for verifying successful migration
  - [ ] Transition solver agent into a temporary compatibility Layer
    - [x] Identify all references to SolverAgent in the codebase (see `docs/migration/solver_agent_references.md` for the complete list)
    - [x] Add appropriate deprecation warnings throughout the codebase
    - [x] Gradually update SolverAgent internals one-by-one (+corresponding tests), testing between each one, to delegate to hierarchical agents
  - [ ] Apply incremental updates to the rest of system to transition from SolverAgent to hierarchical system
    - [ ] Carefully refactor one deprecated component (+corresponding tests) at a time to not use SolverAgent, with tests between each change, run linter and test suite after each change to verify functionality
  - [ ] Update examples to show new approach
    - [ ] Annotate existing examples with deprecation notices if they no longer apply
    - [ ] Create parallel examples showing the new hierarchical approach where it differs
  - [ ] Update CLI code to use hierarchical agents
    - [ ] Identify all CLI dependencies on SolverAgent (see `docs/migration/solver_agent_references.md` for details)
    - [ ] Incrementally update each reference (+corresponding tests), testing between changes
    - [ ] Ensure backward compatibility during transition
  - [ ] Plan for eventual removal of SolverAgent
    - [ ] Ensure all documentation is updated to focus on hierarchical system
    - [ ] Verify that nothing uses the SolverAgent implementation anymore (check against `docs/migration/solver_agent_references.md`)
    - [ ] Remove SolverAgent implementation completely
    - [ ] Verify all tests pass with SolverAgent removed
  - [ ] Update documentation
    - [ ] remove all deprecated examples
    - [ ] Remove all traces of SolverAgent in the documentation to avoid confusion (see `docs/migration/solver_agent_references.md` for a list of documentation to update)
  - [ ] Update CLI
    - [ ] Remove any backward compatibility still left
    - [ ] Review the codebase.
    - [ ] Analyzse the migration to see if it left any undesireable traces.
    - [ ] Run tests and linter

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
