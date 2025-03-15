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
    - [x] Carefully refactor one deprecated component (+corresponding tests) at a time to not use SolverAgent, with tests between each change, run linter and test suite after each change to verify functionality
    - [x] Continue refactoring remaining deprecated components
  - [x] Update examples to show new approach
    - [x] Annotate existing examples with deprecation notices if they no longer apply
    - [x] Create parallel examples showing the new hierarchical approach where it differs
  - [x] Update CLI code to use hierarchical agents
    - [x] Identify all CLI dependencies on SolverAgent (see `docs/migration/solver_agent_references.md` for details)
    - [x] Incrementally update each reference (+corresponding tests), testing between changes
    - [x] Ensure backward compatibility during transition
  - [x] Plan for eventual removal of SolverAgent
    - [x] Ensure all documentation is updated to focus on hierarchical system
    - [x] Verify that nothing uses the SolverAgent implementation anymore (check against `docs/migration/solver_agent_references.md`)
    - [x] Remove SolverAgent implementation completely
    - [x] Verify all tests pass with SolverAgent removed
  - [x] Update documentation
    - [x] remove all deprecated examples
    - [x] Remove all traces of SolverAgent in the documentation to avoid confusion (see `docs/migration/solver_agent_references.md` for a list of documentation to update)
  - [x] Update CLI
    - [x] Remove any backward compatibility still left
    - [x] Review the codebase.
    - [x] Analyzse the migration to see if it left any undesireable traces.
    - [x] Run tests and linter

- enable flexible delegation paths between agent types

  - [x] Implement task complexity analysis logic in `src/agent/agent_types/architect.py` for delegation decisions
  - [x] Add direct delegation capability from `ArchitectAgent` to `ExecutorAgent` for simple tasks
  - [x] Implement sub-task complexity evaluation in `src/agent/agent_types/planner.py`
  - [x] Add recursive delegation capability from `PlannerAgent` to additional `PlannerAgent` instances for complex sub-components
  - [x] Enhance `AgentCoordinator` in `src/agent/coordination.py` to support flexible delegation paths
  - [x] Update agent creation methods to support dynamic parent-child relationships
  - [x] Implement capability matching in `src/agent/coordination.py` to route tasks to appropriate agent types
  - [x] Create decision logging in each agent type to track delegation choices
  - [x] Add unit tests in `tests/unit/test_flexible_delegation.py` to verify delegation logic
  - [x] Create integration tests that verify direct and recursive delegation patterns
  - [x] Update documentation in `docs/howto/delegation_strategies.md` with delegation decision guidelines

- enable the top agent to create other agents by itself, as a tool

  - [x] Update `AgentCoordinator` in `src/agent/coordination.py` to support role-based agent creation
  - [x] Enhance `InMemoryAgentRegistry` in `src/agent/coordination.py` to support parent-child relationships
  - [x] Add parent-child agent registration methods to `AgentRegistry` implementations
  - [x] Update the message system in `src/messages/utils.py` to include sender/receiver hierarchy information
  - [x] Add resource management functionality in `src/agent/coordination.py` to prevent excessive agent creation
  - [x] Create an agent capability discovery mechanism in `src/agent/coordination.py`
  - [x] Add unit tests in `tests/unit/test_agent_creation.py` to verify agent creation workflow
  - [x] Create integration tests in `tests/integration/test_dynamic_agents.py` to verify agent creation workflow
  - [ ] Update documentation in `docs/howto/agent_creation.md` with usage examples
  - [ ] Ensure CI/CD pipeline validates the new functionality

- combine the two approaches and make the top agent break down the problem into TODO lists and delegate them to new agents

  - [ ] Extend `AgentCoordinator` in `src/agent/coordination.py` to implement hierarchical task delegation
  - [ ] Implement capability-based task routing in `src/agent/coordination.py` to match tasks to appropriate agents
  - [ ] Add task complexity evaluation to support flexible delegation decisions
  - [ ] Implement progress tracking in `src/agent/state/base.py` to monitor delegated tasks
  - [ ] Add message routing capabilities in `src/messages/routing.py` for hierarchical message delivery
  - [ ] Add result aggregation functionality in `src/

- enable the executor agent to systematically solve the TODO lists prompting themselves until the TODO is done

  - [ ] Enhance common types for task execution
    - [ ] Add execution-related enums in `src/common_types/enums.py` (EXECUTION_STAGE, VERIFICATION_STATUS)
    - [ ] Extend task schema in `src/common_types/task_types.py` with execution tracking fields
  - [ ] Enhance `ExecutorAgent` in `src/agent/agent_types/executor.py` with self-prompting capabilities
    - [ ] Add task iteration mechanism
    - [ ] Implement progress tracking
    - [ ] Add completion criteria evaluation
  - [ ] Create specialized steps in `src/agent/steps.py` for task execution
    - [ ] Add `TaskExecutionStep` for implementing tasks
    - [ ] Create `TaskVerificationStep` for validation
  - [ ] Add progress monitoring in `src/agent/state/memory.py`
    - [ ] Track task completion status
    - [ ] Add blockers and dependencies tracking
  - [ ] Implement retry mechanisms in `src/agent/agent_types/executor.py`
    - [ ] Add failure detection
    - [ ] Create strategy adjustment logic
  - [ ] Enhance prompts in `src/prompts/templates.py` for self-guided execution
    - [ ] Create implementation prompts
    - [ ] Create verification prompts
  - [ ] Add reporting capabilities in `src/messages/schemas/execution.py`
    - [ ] Create progress report messages
    - [ ] Add completion report
  - [ ] Implement success criteria in `src/utils/validation.py`
    - [ ] Add code quality checks
    - [ ] Implement requirements validation
  - [ ] Create unit tests in `tests/unit/test_executor_self_prompting.py`
  - [ ] Create integration tests in `tests/integration/test_task_execution.py`
  - [ ] Update documentation in `docs/howto/self_executing_tasks.md`

- setup a system for the agents to use tools with Model Context Protocol (MCP)
  https://modelcontextprotocol.io/introduction
  https://github.com/modelcontextprotocol/python-sdk

  - [ ] Extend common types for MCP
    - [ ] Add MCP-related enums in `src/common_types/enums.py` (TOOL_TYPE, RESOURCE_TYPE, PROMPT_TYPE)
    - [ ] Create MCP tool types in `src/common_types/tool_types.py` with proper protocols and implementations
    - [ ] Add MCP message schemas in `src/common_types/message_types.py` for tool requests and responses
    - [ ] Define MCP result types in `src/common_types/result_types.py` for tool executions and prompt handling
    - [ ] Create MCP error types in `src/common_types/error_types.py` for handling MCP-specific errors
  - [ ] Integrate Model Context Protocol (MCP) in `src/agent/tools/`
    - [ ] Create `mcp_client.py` with MCP client implementation and connection handling
    - [ ] Create `mcp_server.py` with server implementation for exposing agent capabilities
    - [ ] Implement protocol handler in `protocol.py` for MCP message processing
    - [ ] Add `mcp_factory.py` for creating and configuring MCP server/client instances
  - [ ] Implement MCP primitives in `src/agent/tools/primitives/`
    - [ ] Create `tools.py` for model-controlled functions that LLMs can invoke
    - [ ] Create `resources.py` for application-controlled contextual data access
    - [ ] Create `prompts.py`
