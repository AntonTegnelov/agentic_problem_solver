# Self-Executing Tasks

This guide explains how the ExecutorAgent in APS systematically solves tasks through self-prompting capabilities.

## Overview

The ExecutorAgent has been enhanced with self-prompting capabilities that allow it to systematically work through tasks by iterating through execution stages until completion. This approach enables the agent to break down complex implementation tasks into manageable steps, track progress, and verify results at each stage.

## Execution Stages

The ExecutorAgent follows a structured workflow with the following execution stages:

1. **PLANNING**: Analyze requirements and create an implementation plan
2. **IMPLEMENTING**: Write the actual code implementation
3. **TESTING**: Create and run tests to verify the implementation
4. **REFINING**: Address issues identified during testing
5. **FINALIZING**: Complete final review and documentation

Each stage has specific objectives and outputs that feed into subsequent stages, creating a comprehensive execution pipeline.

## Task Execution Process

### 1. Task Initialization

When a task is assigned to an ExecutorAgent, it initializes the task with execution tracking fields:

```python
task.execution_stage = ExecutionStage.PLANNING  # Start with planning
task.verification_status = VerificationStatus.PENDING
task.execution_attempts = 0
task.execution_logs = []
task.verification_details = {}
task.execution_metadata = {}
```

### 2. Stage-Specific Prompting

For each execution stage, the agent creates specialized prompts that focus on the specific requirements of that stage:

- **Planning Stage**: Focus on breaking down the implementation into logical steps
- **Implementing Stage**: Write clean, efficient code based on the planning
- **Testing Stage**: Create comprehensive test cases for the implementation
- **Refining Stage**: Address issues identified during testing
- **Finalizing Stage**: Perform final review and complete documentation

### 3. Task Iteration

The ExecutorAgent uses the `iterate_task` method to process a task through its execution stages:

1. Initialize task metadata and execution tracking
2. Generate a stage-specific prompt based on the current execution stage
3. Process the prompt to generate a result
4. Store the result in the task's execution metadata
5. Verify the result using the TaskVerificationStep
6. Advance to the next execution stage if verification passes
7. Repeat until the task reaches the FINALIZING stage and passes verification

### 4. Result Verification

After each execution stage, the TaskVerificationStep evaluates the results:

1. Create a verification prompt that includes the task description, acceptance criteria, and execution results
2. Process the verification prompt to generate a verification assessment
3. Update the task's verification status based on the assessment
4. If verification fails, the agent may retry the current stage with adjusted prompts

### 5. Progress Tracking

Throughout the execution process, the agent tracks progress and updates the task status:

- Increments `execution_attempts` counter
- Adds entries to `execution_logs` for significant events
- Updates `verification_details` with assessment results
- Stores stage results in `execution_metadata`
- Updates task status to COMPLETED when finalized and verified

## Implementation Example

Here's a simplified example of how the ExecutorAgent processes a task:

```python
# Task assigned to ExecutorAgent
task = Task(
    description="Implement a function to calculate the sum of two numbers",
    acceptance_criteria=["Function should handle integer inputs", "Should return the sum as an integer"]
)

# ExecutorAgent processes the task through stages
result = await executor_agent.iterate_task(task)

# After processing, the task contains:
# - Planning result in task.execution_metadata["planning_result"]
# - Implementation in task.execution_metadata["implementation_result"]
# - Testing results in task.execution_metadata["testing_result"]
# - Refined implementation in task.execution_metadata["refined_implementation"]
# - Final result in task.execution_metadata["final_result"]
```

## Retry Mechanisms

The ExecutorAgent includes retry mechanisms for handling failures:

1. **Failure Detection**: Identifies verification failures and execution errors
2. **Strategy Adjustment**: Modifies prompts based on previous failures
3. **Maximum Attempts**: Limits the number of retry attempts to prevent infinite loops

## Integration with Agent Hierarchy

The self-executing task system integrates with the hierarchical agent system:

1. **ArchitectAgent** breaks down complex problems into components
2. **PlannerAgent** refines components into implementable tasks
3. **ExecutorAgent** systematically implements tasks through self-prompting
4. Results flow back up the hierarchy for aggregation

## Best Practices

When working with self-executing tasks:

1. Provide clear, specific task descriptions
2. Include detailed acceptance criteria
3. Set appropriate task complexity and priority
4. Consider dependencies between tasks
5. Monitor execution logs for debugging

## Customization

The self-executing task system can be customized by:

1. Modifying stage-specific prompts in `src/prompts/templates.py`
2. Adjusting verification criteria in `src/utils/validation.py`
3. Extending the execution stages enum in `src/common_types/enums.py`
4. Customizing the TaskExecutionStep in `src/agent/steps.py`
