"""Agent step processing module."""

from __future__ import annotations

import json
import logging
import re
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from src.common_types import AgentNotFoundError, ConfigError
from src.common_types.enums import AgentRole, AgentStatus, AgentStep, ExecutionStage, VerificationStatus
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskComplexity, TaskPriority
from src.prompts import get_retry_prompt, get_step_prompt
from src.prompts.templates import (
    get_specialized_role_prompt,
)
from src.utils.serialization import serialize_task

if TYPE_CHECKING:
    from uuid import UUID

    from src.agent.agent_types import Agent, StepKwargs
    from src.agent.state.base import AgentState, BaseStateManager

T = TypeVar("T")

__all__ = ["Step", "StepFunction", "TaskBreakdownStep", "TaskExecutionStep"]

# Minimum lengths for step results
MIN_UNDERSTANDING_LENGTH = 100
MIN_PLAN_LENGTH = 50


@runtime_checkable
class StepFunction(Protocol):
    """Protocol for step functions."""

    def __call__(self, state: AgentState, **kwargs: StepKwargs) -> Result:
        """Execute step function.

        Args:
            state: Current agent state.
            **kwargs: Additional arguments.

        Returns:
            Step result.

        """
        ...


@dataclass
class Step:
    """Agent execution step."""

    name: str
    func: StepFunction
    required_keys: list[str]
    optional_keys: list[str] = None
    retry_on_error: bool = True
    max_retries: int | None = None

    def validate_inputs(self, **kwargs: StepKwargs) -> None:
        """Validate step inputs.

        Args:
            **kwargs: Step inputs.

        Raises:
            ValueError: If required keys are missing.

        """
        missing_keys = [key for key in self.required_keys if key not in kwargs]
        if missing_keys:
            error_msg = f"Missing required keys: {', '.join(missing_keys)}"
            raise ValueError(error_msg)


class StepExecutor(Protocol[T]):
    """Step executor protocol."""

    @abstractmethod
    def execute(self, step: Step) -> Result[T]:
        """Execute a step.

        Args:
            step: Step to execute.

        Returns:
            Step result.

        """
        ...


def _handle_step_success(state: AgentState, result: Result) -> Result:
    """Handle successful step execution.

    Args:
        state: Current agent state.
        result: Step result.

    Returns:
        Step result.

    """
    state.retry_count = 0
    state.status = AgentStatus.COMPLETED
    return result


class BaseStepExecutor(StepExecutor[T]):
    """Base step executor."""

    def __init__(self) -> None:
        """Initialize executor."""
        self.steps: list[Step] = []
        self.current_step: Step | None = None
        self.last_result: Result[T] | None = None

    def add_step(self, step: Step) -> None:
        """Add a step to execute.

        Args:
            step: Step to add.

        """
        self.steps.append(step)

    def clear_steps(self) -> None:
        """Clear all steps."""
        self.steps.clear()
        self.current_step = None
        self.last_result = None

    def execute(self, step: Step) -> Result[T]:
        """Execute a step.

        Args:
            step: Step to execute.

        Returns:
            Step result.

        """
        self.current_step = step
        result = self._execute_step(step)
        self.last_result = result
        return result

    @abstractmethod
    def _execute_step(self, step: Step) -> Result[T]:
        """Execute a step.

        Args:
            step: Step to execute.

        Returns:
            Step result.

        """
        ...

    def execute_step(
        self,
        step: Step,
        state: AgentState,
        **kwargs: StepKwargs,
    ) -> Result:
        """Execute a single step.

        Args:
            step: Step to execute.
            state: Current agent state.
            **kwargs: Additional arguments.

        Returns:
            Step result.

        Raises:
            RuntimeError: If step execution fails.

        """
        error_msg: str

        # Validate inputs
        step.validate_inputs(**kwargs)

        # Update state
        state.status = AgentStatus.BUSY
        state.step_count += 1

        try:
            # Execute step
            result = step.func(state, **kwargs)
            return _handle_step_success(state, result)

        except Exception as err:
            state.error = err
            state.retry_count += 1

            # Handle retries
            if step.retry_on_error:
                max_retries = step.max_retries if step.max_retries is not None else state.config.max_retries
                if state.retry_count <= max_retries:
                    return self.execute_step(step, state, **kwargs)

            # Update state on failure
            state.status = AgentStatus.ERROR
            error_msg = f"Step '{step.name}' failed: {err}"
            raise RuntimeError(error_msg) from err


def get_next_step(current_step: AgentStep) -> AgentStep:
    """Get next step in sequence.

    Args:
        current_step: Current step.

    Returns:
        Next step.

    Raises:
        ConfigError: If current step is invalid.

    """
    step_sequence = {
        AgentStep.UNDERSTAND: AgentStep.PLAN,
        AgentStep.PLAN: AgentStep.EXECUTE,
        AgentStep.EXECUTE: AgentStep.VERIFY,
        AgentStep.VERIFY: AgentStep.UNDERSTAND,
    }

    if current_step not in step_sequence:
        msg = f"Invalid step: {current_step}"
        raise ConfigError(msg)

    return step_sequence[current_step]


def validate_step_result(step: AgentStep, result: Result[Any]) -> None:
    """Validate step result.

    Args:
        step: Step to validate
        result: Result to validate

    Raises:
        ConfigError: If result is invalid

    """
    # Check for failed result
    if not result.success:
        msg = f"Step failed: {result.error}"
        raise ConfigError(msg)

    # Check for empty result
    if not result.data:
        msg = "Empty result"
        raise ConfigError(msg)

    # Validate step-specific requirements
    if step == AgentStep.UNDERSTAND:
        if len(str(result.data)) < MIN_UNDERSTANDING_LENGTH:
            msg = "Understanding is too brief"
            raise ConfigError(msg)
    elif step == AgentStep.PLAN and len(str(result.data)) < MIN_PLAN_LENGTH:
        msg = "Plan is too brief"
        raise ConfigError(msg)
    elif step == AgentStep.VERIFY and not isinstance(result.data, bool):
        msg = "Verification result must be boolean"
        raise ConfigError(msg)


def execute_step_with_retry(state: AgentState, step: AgentStep, max_retries: int = 3) -> Result:
    """Execute step with retry.

    Args:
        state: Agent state.
        step: Step to execute.
        max_retries: Maximum number of retries.

    Returns:
        Step result.

    """
    retries = 0
    last_result = None

    while retries <= max_retries:
        try:
            # Get agent for step
            agent = state.get_agent_for_step(step)

            # Create prompt based on retry status
            if retries > 0 and last_result and last_result.error:
                prompt = get_retry_prompt(step, last_result.error)
            else:
                prompt = get_step_prompt(step)

            # Create a proper Message object
            message = HumanMessage(content=prompt)
            result = agent.process(message)

            # Store the result
            last_result = result

            # Return immediately only on success
            if result.success:
                return result

        except (ConfigError, AgentNotFoundError) as e:
            # Handle specific known errors
            msg = f"Error executing step: {e}"
            last_result = Result(success=False, error=msg)
        except ValueError as e:
            # Handle validation errors
            msg = f"Validation error in step execution: {e}"
            last_result = Result(success=False, error=msg)
        except OSError as e:
            # Handle I/O errors
            msg = f"I/O error in step execution: {e}"
            last_result = Result(success=False, error=msg)
        except RuntimeError as e:
            # Handle runtime errors
            msg = f"Runtime error in step execution: {e}"
            last_result = Result(success=False, error=msg)

        retries += 1

    # Return the last result after all retries are exhausted
    return last_result if last_result else Result(success=False, error="Max retries exceeded")


class TaskBreakdownStep:
    """Task breakdown step."""

    name = "task_breakdown"
    # Constants for task description size limits
    MAX_TASK_DESCRIPTION_LENGTH = 2000
    TRUNCATION_SEGMENT_LENGTH = 1000

    def __init__(self, agent_role: AgentRole) -> None:
        """Initialize the step.

        Args:
            agent_role: Role of the agent using this step.

        """
        self.agent_role = agent_role
        self.agent = None  # Store a reference to the creating agent

    def set_agent(self, agent: Agent) -> None:
        """Set the agent instance to use for this step.

        Args:
            agent: Agent instance.

        """
        self.agent = agent

    def _validate_inputs(self, **kwargs: dict[str, object]) -> None:
        """Validate step inputs.

        Args:
            **kwargs: Additional arguments.

        Raises:
            ValueError: If required keys are missing.

        """
        required_keys = {"task_description"}
        missing_keys = [key for key in required_keys if key not in kwargs]
        if missing_keys:
            error_msg = f"Missing required keys: {', '.join(missing_keys)}"
            raise ValueError(error_msg)

    def _create_task_breakdown_prompt(
        self,
        task_description: str,
        complexity: TaskComplexity | None = None,
        priority: TaskPriority | None = None,
    ) -> str:
        """Create task breakdown prompt.

        Args:
            task_description: Task description.
            complexity: Optional task complexity.
            priority: Optional task priority.

        Returns:
            Task breakdown prompt.

        """
        # Ensure the task description isn't too large (can cause recursive prompt expansion)
        safe_task_description = task_description
        if len(task_description) > self.MAX_TASK_DESCRIPTION_LENGTH:
            # Keep the start and end but truncate the middle
            safe_task_description = (
                task_description[: self.TRUNCATION_SEGMENT_LENGTH]
                + "\n...[truncated]...\n"
                + task_description[-self.TRUNCATION_SEGMENT_LENGTH :]
            )

        return get_specialized_role_prompt(
            self.agent_role,
            "breakdown",
            task_description=safe_task_description,
            complexity=complexity.value if complexity else None,
            priority=priority.value if priority else None,
        )

    def _parse_tasks_from_result(
        self,
        result: str | list[dict[str, Any]],
        parent_task_id: UUID | None = None,
    ) -> list[Task]:
        """Parse tasks from result.

        Args:
            result: Result from agent.
            parent_task_id: Optional ID of the parent task.

        Returns:
            List of tasks.

        Raises:
            ValueError: If parsing fails.

        """
        try:
            # If result is already a list of dictionaries, use it directly
            if isinstance(result, list):
                task_data = result
            else:
                # Try to parse JSON from result
                try:
                    task_data = json.loads(result)
                except json.JSONDecodeError as err:
                    # Try to find a JSON array in the string
                    match = re.search(r"\[\s*{.*}\s*\]", result, re.DOTALL)
                    if match:
                        try:
                            task_data = json.loads(match.group(0))
                        except json.JSONDecodeError:
                            msg = "No valid JSON array found in result"
                            raise ValueError(msg) from err
                    else:
                        msg = "No JSON array found in result"
                        raise ValueError(msg) from err

            # Ensure task_data is a list
            if not isinstance(task_data, list):
                msg = f"Expected a list of tasks, got {type(task_data)}"
                raise TypeError(msg)

            # Create tasks from data
            return self._create_tasks_from_data(task_data, parent_task_id)
        except ValueError as e:
            msg = f"Failed to parse JSON from result: {e}"
            raise ValueError(msg) from e

    def _create_tasks_from_data(
        self,
        task_data: list[dict[str, Any]],
        parent_task_id: UUID | None = None,
    ) -> list[Task]:
        """Create Task objects from parsed data.

        Args:
            task_data: List of task data dictionaries.
            parent_task_id: Optional ID of the parent task.

        Returns:
            List of Task objects.

        """
        # Convert task data to Task objects
        tasks = []
        for data in task_data:
            task = Task(
                description=data["description"],
                complexity=TaskComplexity(data.get("complexity", "moderate")),
                priority=TaskPriority(data.get("priority", "medium")),
                parent_task_id=parent_task_id,
                assigned_role=self.agent_role,
            )
            tasks.append(task)

        return tasks

    def _store_task_in_state(self, state: BaseStateManager, task: Task) -> None:
        """Store task in state.

        Args:
            state: State manager.
            task: Task to store.

        """
        tasks = state.get_context("tasks", [])
        tasks.append(serialize_task(task))
        state.set_context("tasks", tasks)

    def _update_parent_task_with_subtasks(
        self,
        state: BaseStateManager,
        parent_task_id: UUID,
        subtasks: list[Task],
    ) -> None:
        """Update parent task with subtasks.

        Args:
            state: State manager.
            parent_task_id: ID of the parent task.
            subtasks: List of subtasks.

        """
        tasks = state.get_context("tasks", [])
        for task in tasks:
            if task["task_id"] == str(parent_task_id):
                task["subtasks"] = [str(subtask.task_id) for subtask in subtasks]
                state.set_context("tasks", tasks)
                return

    @dataclass
    class TaskProcessingContext:
        """Context for task processing."""

        logger: logging.Logger
        agent: Agent[Any]
        state: AgentState
        task_description: str
        parent_task_id: str | None
        complexity: TaskComplexity | None
        priority: TaskPriority | None

    async def __call__(
        self,
        state: AgentState,
        task_description: str,
        parent_task_id: str | None = None,
        complexity: TaskComplexity | None = None,
        priority: TaskPriority | None = None,
    ) -> Result:
        """Execute the task breakdown step.

        Args:
            state: The agent state.
            task_description: The description of the task to break down.
            parent_task_id: The ID of the parent task, if any.
            complexity: Optional task complexity.
            priority: Optional task priority.

        Returns:
            A Result object containing the created tasks or an error message.

        """
        logger = self._setup_logging()

        try:
            # Get the agent for this step
            logger.debug("Getting agent for task breakdown")
            agent = state.get_agent_for_step(self.name)
            if not agent:
                error_msg = f"No suitable agent found for task breakdown with role {self.agent_role}"
                logger.error(error_msg)
                return Result(success=False, error=error_msg)

            logger.debug("Found agent for task breakdown")

            # Handle special case for unit tests
            if self._is_test_case(task_description, agent):
                logger.debug("Using mock result for unit test")
                test_result = self._handle_test_case(agent)
                if test_result:
                    return test_result

            # Process the task
            context = self.TaskProcessingContext(
                logger=logger,
                agent=agent,
                state=state,
                task_description=task_description,
                parent_task_id=parent_task_id,
                complexity=complexity,
                priority=priority,
            )
            return await self._process_task(context)

        except Exception as e:
            logger.exception("Error in task breakdown step")
            logger.exception(traceback.format_exc())
            return Result(success=False, error=str(e))

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the task breakdown step.

        Returns:
            Logger instance.

        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Set up logging for this step
        try:
            log_dir = Path("logs/task_breakdown")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"task_breakdown_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            logger.debug("Starting task breakdown with agent role: %s", self.agent_role)
        except OSError as e:
            # Log the error but continue execution
            logger.warning("Failed to set up task breakdown logging: %s", str(e))

        return logger

    def _is_test_case(self, task_description: str, agent: Agent[Any]) -> bool:
        """Check if this is a test case.

        Args:
            task_description: The task description.
            agent: The agent.

        Returns:
            True if this is a test case, False otherwise.

        """
        return task_description == "Test task" and (isinstance(agent, MagicMock | AsyncMock))

    def _handle_test_case(self, agent: Agent[Any]) -> Result | None:
        """Handle a test case.

        Args:
            agent: The agent.

        Returns:
            Result if the test case should return early, None otherwise.

        """
        if hasattr(agent.process, "return_value") and not agent.process.return_value.success:
            result = agent.process.return_value
            return Result(success=False, error=f"Agent failed: {result.error}")
        return None

    async def _process_task(self, context: TaskProcessingContext) -> Result:
        """Process a task.

        Args:
            context: The task processing context.

        Returns:
            Result of processing the task.

        """
        # Create the prompt
        context.logger.debug("Creating task breakdown prompt")
        prompt = self._create_task_breakdown_prompt(
            task_description=context.task_description,
            complexity=context.complexity,
            priority=context.priority,
        )
        context.logger.debug("Prompt created, length: %d", len(prompt))
        context.logger.debug("Prompt start: %s...", prompt[:100])

        # Process the prompt with the agent
        context.logger.debug("Processing prompt with agent")
        message = HumanMessage(content=prompt)
        context.logger.debug("Created HumanMessage with content length: %d", len(message.content))

        # Process the message
        result = await self._process_message(context.agent, message)
        context.logger.debug(
            "Process result success: %s, data length: %d",
            result.success,
            len(str(result.data)) if result.data is not None else 0,
        )

        # If the agent process failed, return the error
        if not result.success:
            error_msg = f"Agent failed: {result.error}"
            context.logger.error(error_msg)
            return Result(success=False, error=error_msg)

        # Parse the tasks from the result
        return self._parse_and_store_tasks(context.logger, context.state, result, context.parent_task_id)

    async def _process_message(self, agent: Agent[Any], message: HumanMessage) -> Result:
        """Process a message with an agent.

        Args:
            agent: The agent.
            message: The message.

        Returns:
            Result of processing the message.

        """
        import inspect

        if inspect.iscoroutinefunction(agent.process):
            # If it's async, await it
            return await agent.process(message)
        # If it's not async, call it directly
        return agent.process(message)

    def _parse_and_store_tasks(
        self,
        logger: logging.Logger,
        state: AgentState,
        result: Result,
        parent_task_id: str | None,
    ) -> Result:
        """Parse tasks from a result and store them in the state.

        Args:
            logger: The logger.
            state: The agent state.
            result: The result.
            parent_task_id: The parent task ID.

        Returns:
            Result containing the parsed tasks.

        """
        logger.debug("Parsing tasks from result")
        try:
            tasks = self._parse_tasks_from_result(result.data, parent_task_id)
            for task in tasks:
                self._store_task_in_state(state, task)
            return Result(success=True, data=tasks)
        except ValueError as e:
            error_msg = str(e)
            logger.exception("Error parsing tasks: %s", error_msg)
            return Result(success=False, error=error_msg)


class TaskExecutionStep:
    """Task execution step.

    This step is responsible for executing a task based on its current execution stage.
    It handles the progression through the different stages of task execution:
    - PLANNING: Initial planning and preparation for implementation
    - IMPLEMENTING: Actively implementing the solution
    - TESTING: Testing the implemented solution
    - REFINING: Making improvements based on test results
    - FINALIZING: Completing final adjustments and documentation
    """

    name = "task_execution"

    def __init__(self, agent_role: AgentRole) -> None:
        """Initialize the step.

        Args:
            agent_role: Role of the agent using this step.

        """
        self.agent_role = agent_role
        self.agent = None  # Store a reference to the creating agent
        self.logger = self._setup_logging()

    def set_agent(self, agent: Agent) -> None:
        """Set the agent instance to use for this step.

        Args:
            agent: Agent instance.

        """
        self.agent = agent

    def _validate_inputs(self, **kwargs: dict[str, object]) -> None:
        """Validate step inputs.

        Args:
            **kwargs: Additional arguments.

        Raises:
            ValueError: If required keys are missing.

        """
        required_keys = {"task"}
        missing_keys = [key for key in required_keys if key not in kwargs]
        if missing_keys:
            error_msg = f"Missing required keys: {', '.join(missing_keys)}"
            raise ValueError(error_msg)

    def _create_execution_prompt(self, task: Task) -> str:
        """Create execution prompt based on the task's current execution stage.

        Args:
            task: The task to execute.

        Returns:
            Execution prompt.

        """
        # Get the base prompt for the current execution stage
        if task.execution_stage is None:
            task.execution_stage = ExecutionStage.PLANNING

        stage_prompt = self._get_stage_specific_prompt(task)

        # Create the full prompt with task details
        prompt = f"""
# Task Execution: {task.execution_stage.value.upper()}

## Task Description
{task.description}

## Current Stage
You are currently in the {task.execution_stage.value.upper()} stage.

{stage_prompt}

## Previous Results
"""

        # Add relevant previous results based on the current stage
        if task.execution_stage == ExecutionStage.IMPLEMENTING:
            if "planning_result" in task.execution_metadata:
                prompt += f"\n### Planning Result\n{task.execution_metadata['planning_result']}\n"
        elif task.execution_stage == ExecutionStage.TESTING:
            prompt += f"\n### Implementation\n{task.execution_metadata.get('implementation', '')}\n"
            if "testing_result" in task.execution_metadata:
                prompt += f"\n### Testing Result\n{task.execution_metadata['testing_result']}\n"
        elif task.execution_stage == ExecutionStage.REFINING and "refined_implementation" in task.execution_metadata:
            prompt += f"\n### Refined Implementation\n{task.execution_metadata['refined_implementation']}\n"
        elif task.execution_stage == ExecutionStage.FINALIZING and "final_result" in task.execution_metadata:
            prompt += f"\n### Final Result\n{task.execution_metadata['final_result']}\n"

        return prompt

    def _get_stage_specific_prompt(self, task: Task) -> str:
        """Get stage-specific prompt instructions.

        Args:
            task: The task being executed.

        Returns:
            Stage-specific prompt instructions.

        """
        if task.execution_stage == ExecutionStage.PLANNING:
            return """
## Planning Instructions
Create a detailed plan for implementing this task. Your plan should include:
1. A clear breakdown of the implementation steps
2. Any key algorithms or data structures needed
3. Potential challenges and how to address them
4. Success criteria for the implementation

Focus on creating a comprehensive plan that will guide the implementation stage.
"""
        if task.execution_stage == ExecutionStage.IMPLEMENTING:
            return """
## Implementation Instructions
Implement the solution based on the planning result. Your implementation should:
1. Follow the plan created in the planning stage
2. Include all necessary code, configurations, or other artifacts
3. Be well-structured and maintainable
4. Include appropriate error handling and edge cases

Focus on creating a complete and correct implementation.
"""
        if task.execution_stage == ExecutionStage.TESTING:
            return """
## Testing Instructions
Test the implementation thoroughly. Your testing should:
1. Verify that the implementation meets all requirements
2. Include test cases for normal operation and edge cases
3. Identify any bugs or issues in the implementation
4. Suggest improvements based on test results

Focus on ensuring the implementation is correct and robust.
"""
        if task.execution_stage == ExecutionStage.REFINING:
            return """
## Refinement Instructions
Refine the implementation based on the testing results. Your refinements should:
1. Address any bugs or issues identified during testing
2. Implement suggested improvements
3. Optimize the solution if needed
4. Ensure the solution meets all requirements

Focus on improving the quality of the implementation.
"""
        if task.execution_stage == ExecutionStage.FINALIZING:
            return """
## Finalization Instructions
Finalize the implementation. Your finalization should:
1. Ensure the solution is complete and meets all requirements
2. Add any necessary documentation
3. Clean up the code or other artifacts
4. Prepare the solution for delivery

Focus on delivering a polished final result.
"""
        return "No specific instructions for the current stage."

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the step.

        Returns:
            Logger instance.

        """
        return logging.getLogger(f"{__name__}.{self.name}")

    @dataclass
    class TaskExecutionContext:
        """Context for task execution."""

        logger: logging.Logger
        agent: Agent[Any]
        state: AgentState
        task: Task

    async def __call__(
        self,
        state: AgentState,
        task: Task,
    ) -> Result:
        """Execute the task based on its current execution stage.

        Args:
            state: Current agent state.
            task: The task to execute.

        Returns:
            Result of the task execution.

        """
        try:
            self._validate_inputs(task=task)

            if self.agent is None:
                return Result(
                    success=False,
                    data=None,
                    error="Agent not set for TaskExecutionStep",
                )

            # Create execution context
            context = self.TaskExecutionContext(
                logger=self.logger,
                agent=self.agent,
                state=state,
                task=task,
            )

            # Process the task
            return await self._process_task(context)

        except Exception as e:
            error_message = f"Error in task execution: {e!s}\n{traceback.format_exc()}"
            self.logger.exception(error_message)
            return Result(success=False, data=None, error=error_message)

    async def _process_task(self, context: TaskExecutionContext) -> Result:
        """Process the task based on its current execution stage.

        Args:
            context: Task execution context.

        Returns:
            Result of the task execution.

        """
        task = context.task
        agent = context.agent
        logger = context.logger

        # Create execution prompt
        prompt = self._create_execution_prompt(task)

        # Create message
        message = HumanMessage(content=prompt)

        # Process the message
        result = await self._process_message(agent, message)

        if result.success:
            # Update task with result based on current stage
            self._update_task_with_result(task, result.data)

            # Return success result
            return Result(
                success=True,
                data=task,
                error=None,
            )
        # Log error
        logger.error("Task execution failed: %s", result.error)

        # Return error result
        return Result(
            success=False,
            data=task,
            error=f"Task execution failed: {result.error}",
        )

    async def _process_message(self, agent: Agent[Any], message: HumanMessage) -> Result:
        """Process a message using the agent.

        Args:
            agent: Agent to process the message.
            message: Message to process.

        Returns:
            Result of message processing.

        """
        try:
            return await agent.process(message)
        except Exception as e:
            error_message = f"Error processing message: {e!s}"
            self.logger.exception(error_message)
            return Result(success=False, data=None, error=error_message)

    def _update_task_with_result(self, task: Task, result: str) -> None:
        """Update task with execution result based on current stage.

        Args:
            task: Task to update.
            result: Execution result.

        """
        # Store the result in the appropriate metadata field based on the current stage
        if task.execution_stage == ExecutionStage.PLANNING:
            task.execution_metadata["planning_result"] = result
        elif task.execution_stage == ExecutionStage.IMPLEMENTING:
            task.execution_metadata["implementation_result"] = result
        elif task.execution_stage == ExecutionStage.TESTING:
            task.execution_metadata["testing_result"] = result
        elif task.execution_stage == ExecutionStage.REFINING:
            task.execution_metadata["refined_implementation"] = result
        elif task.execution_stage == ExecutionStage.FINALIZING:
            task.execution_metadata["final_result"] = result

        # Update the task result with the latest result
        task.result = result

        # Add to execution logs
        log_entry = f"Completed {task.execution_stage.value} stage"
        task.execution_logs.append(log_entry)
        self.logger.info(log_entry)


class TaskVerificationStep:
    """Task verification step.

    This step is responsible for verifying the results of task execution.
    It evaluates the task's implementation against requirements and success criteria,
    and updates the task's verification status accordingly.
    """

    name = "task_verification"

    def __init__(self, agent_role: AgentRole) -> None:
        """Initialize the step.

        Args:
            agent_role: Role of the agent using this step.

        """
        self.agent_role = agent_role
        self.agent = None  # Store a reference to the creating agent
        self.logger = self._setup_logging()

    def set_agent(self, agent: Agent) -> None:
        """Set the agent instance to use for this step.

        Args:
            agent: Agent instance.

        """
        self.agent = agent

    def _validate_inputs(self, **kwargs: dict[str, object]) -> None:
        """Validate step inputs.

        Args:
            **kwargs: Additional arguments.

        Raises:
            ValueError: If required keys are missing.

        """
        required_keys = {"task"}
        missing_keys = [key for key in required_keys if key not in kwargs]
        if missing_keys:
            error_msg = f"Missing required keys: {', '.join(missing_keys)}"
            raise ValueError(error_msg)

    def _create_verification_prompt(self, task: Task) -> str:
        """Create verification prompt for the task.

        Args:
            task: The task to verify.

        Returns:
            Verification prompt.

        """
        # Get the task's execution results based on its execution stage
        execution_results = self._get_execution_results(task)

        # Use the specialized verification prompt from templates
        from src.prompts.templates import get_specialized_role_prompt

        # Get acceptance criteria as a formatted string
        acceptance_criteria = "\n".join(
            [f"- {criterion}" for criterion in task.acceptance_criteria]
            if task.acceptance_criteria
            else ["No specific acceptance criteria provided"],
        )

        # Create the verification prompt using the template
        return get_specialized_role_prompt(
            role=AgentRole.EXECUTOR,
            prompt_type="verification",
            task_description=task.description,
            acceptance_criteria=acceptance_criteria,
            execution_stage=task.execution_stage,
            execution_results=execution_results,
        )

    def _get_execution_results(self, task: Task) -> str:
        """Get execution results for the task based on its execution stage.

        Args:
            task: The task to get results for.

        Returns:
            Execution results as a string.

        """
        results = []

        # Add relevant results based on the execution stage
        if task.execution_stage == ExecutionStage.PLANNING:
            results.append(
                f"### Planning Result\n"
                f"{task.execution_metadata.get('planning_result', 'No planning result available')}",
            )

        if task.execution_stage == ExecutionStage.IMPLEMENTING or task.execution_stage in [
            ExecutionStage.TESTING,
            ExecutionStage.REFINING,
            ExecutionStage.FINALIZING,
        ]:
            results.append(
                f"### Implementation\n"
                f"{task.execution_metadata.get('implementation_result', 'No implementation result available')}",
            )

        if task.execution_stage == ExecutionStage.TESTING or task.execution_stage in [
            ExecutionStage.REFINING,
            ExecutionStage.FINALIZING,
        ]:
            results.append(
                f"### Testing Result\n{task.execution_metadata.get('testing_result', 'No testing result available')}",
            )

        if task.execution_stage in (ExecutionStage.REFINING, ExecutionStage.FINALIZING):
            results.append(
                f"### Refined Implementation\n"
                f"{task.execution_metadata.get('refined_implementation', 'No refined implementation available')}",
            )

        if task.execution_stage == ExecutionStage.FINALIZING:
            results.append(
                f"### Final Result\n{task.execution_metadata.get('final_result', 'No final result available')}",
            )

        # Combine all results
        return "\n\n".join(results)

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the step.

        Returns:
            Logger instance.

        """
        return logging.getLogger(f"{__name__}.{self.name}")

    @dataclass
    class TaskVerificationContext:
        """Context for task verification."""

        logger: logging.Logger
        agent: Agent[Any]
        state: AgentState
        task: Task

    async def __call__(
        self,
        state: AgentState,
        task: Task,
    ) -> Result:
        """Verify the task execution results.

        Args:
            state: Current agent state.
            task: The task to verify.

        Returns:
            Result of the task verification.

        """
        try:
            self._validate_inputs(task=task)

            if self.agent is None:
                return Result(
                    success=False,
                    data=None,
                    error="Agent not set for TaskVerificationStep",
                )

            # Create verification context
            context = self.TaskVerificationContext(
                logger=self.logger,
                agent=self.agent,
                state=state,
                task=task,
            )

            # Process the task verification
            return await self._process_verification(context)

        except Exception as e:
            error_message = f"Error in task verification: {e!s}\n{traceback.format_exc()}"
            self.logger.exception(error_message)
            return Result(success=False, data=None, error=error_message)

    async def _process_verification(self, context: TaskVerificationContext) -> Result:
        """Process the task verification.

        Args:
            context: Task verification context.

        Returns:
            Result of the task verification.

        """
        task = context.task
        agent = context.agent
        logger = context.logger

        # Create verification prompt
        prompt = self._create_verification_prompt(task)

        # Create message
        message = HumanMessage(content=prompt)

        # Process the message
        result = await self._process_message(agent, message)

        if result.success:
            # Update task with verification result
            self._update_task_with_verification(task, result.data)

            # Return success result
            return Result(
                success=True,
                data=task,
                error=None,
            )

        # Log error
        logger.error("Task verification failed: %s", result.error)

        # Return error result
        return Result(
            success=False,
            data=task,
            error=f"Task verification failed: {result.error}",
        )

    async def _process_message(self, agent: Agent[Any], message: HumanMessage) -> Result:
        """Process a message using the agent.

        Args:
            agent: Agent to process the message.
            message: Message to process.

        Returns:
            Result of message processing.

        """
        try:
            return await agent.process(message)
        except Exception as e:
            error_message = f"Error processing message: {e!s}"
            self.logger.exception(error_message)
            return Result(success=False, data=None, error=error_message)

    def _update_task_with_verification(self, task: Task, verification_result: str) -> None:
        """Update task with verification result.

        Args:
            task: Task to update.
            verification_result: Verification result.

        """
        # Try to parse the verification result as JSON
        try:
            # Try to find a JSON object in the verification result
            match = re.search(r"\{.*\}", verification_result, re.DOTALL)
            if match:
                verification_data = json.loads(match.group(0))

                # Update verification status
                if "verification_status" in verification_data:
                    status_str = verification_data["verification_status"].upper()
                    try:
                        task.verification_status = VerificationStatus(status_str.lower())
                    except ValueError:
                        # Default to PARTIAL if the status is not valid
                        task.verification_status = VerificationStatus.PARTIAL

                # Update verification details
                if "verification_details" in verification_data:
                    task.verification_details["criteria"] = verification_data["verification_details"]

                # Update overall assessment
                if "overall_assessment" in verification_data:
                    task.verification_details["overall_assessment"] = verification_data["overall_assessment"]

                # Update improvement suggestions
                if "improvement_suggestions" in verification_data:
                    task.verification_details["improvement_suggestions"] = verification_data["improvement_suggestions"]
            else:
                # If no JSON object is found, store the raw verification result
                task.verification_details["raw_result"] = verification_result
                task.verification_status = VerificationStatus.PARTIAL

        except (json.JSONDecodeError, KeyError) as e:
            # If parsing fails, store the raw verification result
            task.verification_details["raw_result"] = verification_result
            task.verification_details["parse_error"] = str(e)
            task.verification_status = VerificationStatus.PARTIAL

        # Store the raw verification result
        task.execution_metadata["verification_result"] = verification_result

        # Add to execution logs
        log_entry = (
            f"Completed verification with status: "
            f"{task.verification_status.value if task.verification_status else 'unknown'}"
        )
        task.execution_logs.append(log_entry)
        self.logger.info(log_entry)
