# Task Delegation Workflow

This guide explains the process of task delegation within the APS hierarchical agent system, from task breakdown to execution.

## Overview

The APS task delegation workflow is a structured process that connects the task breakdown phase with actual task execution. This approach allows complex problems to be decomposed into manageable pieces that can be efficiently executed by specialized agents.

## Delegation Flow

The delegation workflow follows these key steps:

1. **Task Receipt**: The ArchitectAgent receives a high-level task from the user
2. **Task Breakdown**: The task is broken down into component tasks with appropriate complexity
3. **Delegation Decision**: Each task is evaluated to determine the appropriate agent type for delegation
4. **Task Distribution**: Tasks are delegated to agent instances based on complexity and capabilities
5. **Task Execution**: Receiving agents process their assigned tasks
6. **Result Aggregation**: Results from all tasks are collected and combined
7. **Solution Delivery**: The final solution is returned to the user

## Delegation Components

### Task Breakdown

Task breakdown is performed by the `TaskBreakdownStep` class, which takes a high-level task and breaks it into multiple smaller tasks with:

- Detailed descriptions
- Estimated complexity levels
- Priority assignments
- Dependencies between tasks

### Complexity Analysis

Both ArchitectAgent and PlannerAgent implement complexity analysis to determine the appropriate delegation path:

- **ArchitectAgent.analyze_task_complexity()**: Evaluates high-level tasks for delegation to PlannerAgent or ExecutorAgent
- **PlannerAgent.evaluate_subtask_complexity()**: Determines if subtasks need further planning or can be executed directly

### Task Delegation Methods

The system implements several delegation methods:

- **ArchitectAgent.delegate_breakdown_tasks()**: Delegates tasks from the breakdown process
- **ArchitectAgent.delegate_to_planner()**: Delegates complex tasks to PlannerAgents
- **ArchitectAgent.delegate_to_executor()**: Delegates simple tasks directly to ExecutorAgents
- **PlannerAgent.delegate_to_planner()**: Delegates complex subtasks to other PlannerAgents
- **PlannerAgent.delegate_to_executor()**: Delegates implementable tasks to ExecutorAgents

### Parallel Task Processing

The system supports different parallelization strategies for task processing:

- **Sequential**: Tasks are processed one after another (default)
- **Parallel All**: All tasks are processed in parallel
- **Parallel Independent**: Only tasks without dependencies are processed in parallel
- **Parallel with Dependencies**: Tasks are processed in batches based on dependency relationships
- **Parallel Groups**: Tasks are processed in predefined groups

## Implementation Details

### ArchitectAgent Delegation

The ArchitectAgent implements the `delegate_breakdown_tasks` method to process tasks from the breakdown step:

```python
async def delegate_breakdown_tasks(self, tasks: list[Task]) -> Result[str]:
    """Delegate broken-down tasks to appropriate agents."""
    if not tasks:
        return Result.failure("No tasks to delegate")

    self._logger.info("Delegating %d broken-down tasks", len(tasks))

    # Process tasks with retry logic
    results, errors = await self._process_tasks_with_retry(tasks)

    # Return appropriate result based on success/failure
    return self._create_delegation_result(results, errors)
```

The internal `_process_tasks_with_retry` method handles:

- Task dependency management
- Parallel execution of compatible tasks
- Automatic retries for failed delegations
- Error handling and aggregation

### Delegation Decision Logic

The system makes delegation decisions based on task complexity:

1. **Simple tasks** (Architect → Executor): Tasks that can be executed directly
2. **Moderate tasks** (Architect → Executor with guidance): Tasks that require some planning but are straightforward
3. **Complex tasks** (Architect → Planner → Executor): Tasks that require significant planning
4. **Very complex tasks** (Architect → Planner → Planner → Executor): Tasks that require multiple levels of planning

### Task Execution by ExecutorAgent

The ExecutorAgent uses an iterative self-prompting approach to solve tasks:

1. Receives a well-defined task from a parent agent
2. Progresses through execution stages (PLANNING, IMPLEMENTING, TESTING, REFINING, FINALIZING)
3. Continuously evaluates completion criteria
4. Makes adjustments when encountering issues
5. Returns completed solutions to parent agent

## Example Workflow

1. User submits a request: "Create a user authentication system"
2. ArchitectAgent breaks down the request into component tasks:
   - Design database schema for user data
   - Implement user registration functionality
   - Create login/logout mechanisms
   - Add password reset functionality
   - Implement session management
3. ArchitectAgent analyzes complexity of each task:
   - Database schema task: COMPLEX → delegate to PlannerAgent
   - Other tasks: MODERATE → delegate directly to ExecutorAgents
4. PlannerAgent further decomposes database schema task:
   - Define user table structure
   - Create role/permission tables
   - Implement relationships
   - Add indexes and constraints
5. PlannerAgent delegates these subtasks to ExecutorAgents
6. ExecutorAgents implement their assigned tasks
7. PlannerAgent collects database schema results
8. ArchitectAgent collects all component results
9. Final solution is returned to user

## Advanced Features

### Fallback Mechanisms

The system implements several fallback mechanisms for handling delegation failures:

- Timeout handling for stuck tasks
- Automatic retries for failed delegations
- Alternative delegation paths when preferred routes fail
- Task reassignment when agents encounter capability limitations

### Progress Tracking

The system tracks delegation progress through:

- Task status updates (PENDING, IN_PROGRESS, COMPLETED, FAILED)
- Execution stage tracking for ExecutorAgent tasks
- Roll-up progress calculations for parent tasks
- Deadlock detection and resolution

### Visualization

Basic visualization of the task delegation workflow is available through:

- Dependency graphs showing task relationships
- Execution batch logs showing parallel processing groups
- Detailed delegation decision logs

## Best Practices

1. **Design for Granularity**: Break tasks down to a level where each ExecutorAgent has a clear, specific responsibility
2. **Manage Dependencies**: Explicitly define task dependencies to ensure proper execution order
3. **Use Appropriate Parallelization**: Choose parallelization strategies based on task relationships
4. **Set Reasonable Timeouts**: Configure timeouts based on expected task complexity
5. **Implement Proper Error Handling**: Always include error handling in delegation workflows
