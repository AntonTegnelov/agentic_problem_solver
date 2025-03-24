# Steps Module Structural Analysis

## Overview

The `src/agent/steps.py` file contains classes and functions related to agent step processing. The file provides a framework for executing step-based workflows in agents, with specialized steps for task breakdown, execution, and verification.

## Step Classes and Responsibilities

### Protocol and Base Classes

- `StepFunction` (Protocol): Defines the protocol for step functions, which take an agent state and return a result
- `Step` (Dataclass): Represents an agent execution step with name, function, required keys, etc.
- `StepExecutor` (Protocol): Defines the protocol for executing steps
- `BaseStepExecutor`: Base implementation of the StepExecutor protocol

### Task-Related Steps

- `TaskBreakdownStep`: Breaks down complex tasks into manageable subtasks

  - Main responsibilities: Parse task descriptions, create structured tasks, store tasks in state
  - Key methods: `__call__`, `_process_task`, `_parse_tasks_from_result`

- `TaskExecutionStep`: Executes tasks based on their current execution stage

  - Main responsibilities: Handle different execution stages (planning, implementing, testing, etc.)
  - Key methods: `__call__`, `_process_task`, `_create_execution_prompt`

- `TaskVerificationStep`: Verifies task execution results against requirements
  - Main responsibilities: Evaluate implementations, update verification status
  - Key methods: `__call__`, `_process_verification`, `_update_task_with_verification`

### Utility Functions

- `get_next_step`: Determines the next step in a sequence
- `validate_step_result`: Validates the result of a step
- `execute_step_with_retry`: Executes a step with retry logic
- `_handle_step_success`: Handles successful step execution

## Dependencies

- Internal:

  - `src.common_types`: For types, enums, and error definitions
  - `src.prompts`: For prompt generation and templates
  - `src.utils`: For serialization utilities

- External:
  - Standard libraries: json, logging, re, traceback, dataclasses, datetime, pathlib
  - Type-related: Protocol, TypeVar, runtime_checkable from typing
  - Testing: AsyncMock, MagicMock from unittest.mock
  - LangChain: HumanMessage from langchain_core.messages

## Natural Boundaries for Code Splitting

Based on the file structure and responsibilities, the following natural boundaries emerge:

1. **Core Protocol and Base Classes**: Step, StepFunction, StepExecutor protocols and their base implementations
2. **Task Breakdown**: TaskBreakdownStep and related functionality
3. **Task Execution**: TaskExecutionStep and execution stage handling
4. **Task Verification**: TaskVerificationStep and verification logic
5. **Utility Functions**: Helper functions like get_next_step, validate_step_result

## Relationships Between Steps and Functional Areas

### Task Processing Flow

The steps collectively implement a processing flow for tasks:

1. `TaskBreakdownStep` → `TaskExecutionStep` → `TaskVerificationStep`: Sequential processing flow
2. Each step produces output that feeds into the next step in the chain
3. The outputs are stored in the agent state for tracking and reference

### Agent Type Integration

Different agent types use specific steps for their responsibilities:

1. `ArchitectAgent` primarily uses `TaskBreakdownStep` for high-level task decomposition
2. `PlannerAgent` uses `TaskBreakdownStep` for refining mid-level tasks
3. `ExecutorAgent` uses both `TaskExecutionStep` and `TaskVerificationStep` for implementation and validation

### State Management Integration

All steps interact with `AgentState` for persistence:

1. `TaskBreakdownStep._store_task_in_state`: Stores generated tasks in state
2. `TaskExecutionStep._update_task_with_result`: Updates task execution status and results
3. `TaskVerificationStep._update_task_with_verification`: Records verification outcomes

### Prompt Generation Relationships

Steps depend on prompts from `src.prompts.templates`:

1. `TaskBreakdownStep` uses specialized prompts for different agent roles (Architect, Planner)
2. `TaskExecutionStep` uses stage-specific prompts for different execution phases
3. `TaskVerificationStep` uses validation-focused prompts

### Error Handling and Retry Mechanisms

Steps implement a common approach to error handling:

1. All steps inherit from `Step` which includes `retry_on_error` and `max_retries` fields
2. `BaseStepExecutor.execute_step` implements retry logic for all steps
3. `execute_step_with_retry` provides a higher-level retry mechanism

### Hierarchical Task Management

Steps collaborate to manage task hierarchies:

1. `TaskBreakdownStep._update_parent_task_with_subtasks`: Maintains parent-child relationships
2. `TaskExecutionStep` respects task dependencies during execution
3. `TaskVerificationStep` can propagate verification status to dependent tasks

### LLM Integration

Each step interfaces with language models in a similar pattern:

1. `_create_X_prompt`: Generates a specialized prompt for the LLM
2. `_process_message`: Sends the prompt to the LLM and processes the response
3. Result parsing and validation logic to handle the LLM output
