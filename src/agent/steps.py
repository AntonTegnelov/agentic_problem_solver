"""Agent step processing module."""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable
from uuid import UUID, uuid4

from langchain.schema import HumanMessage

from src.common_types import AgentNotFoundError, ConfigError
from src.common_types.enums import AgentRole, AgentStatus, AgentStep
from src.common_types.result_types import Result as StepResult
from src.common_types.task_types import Task, TaskComplexity, TaskDependency, TaskPriority
from src.prompts import get_retry_prompt, get_step_prompt

if TYPE_CHECKING:
    from src.agent.agent_types import StepKwargs
    from src.agent.state.base import AgentState

T = TypeVar("T")

__all__ = ["Step", "StepFunction", "TaskBreakdownStep"]

# Minimum lengths for step results
MIN_UNDERSTANDING_LENGTH = 100
MIN_PLAN_LENGTH = 50


@runtime_checkable
class StepFunction(Protocol):
    """Protocol for step functions."""

    def __call__(self, state: AgentState, **kwargs: StepKwargs) -> StepResult:
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
        error_msg: str
        missing_keys = [key for key in self.required_keys if key not in kwargs]
        if missing_keys:
            error_msg = f"Missing required keys: {', '.join(missing_keys)}"
            raise ValueError(error_msg)


class StepExecutor(Protocol[T]):
    """Step executor protocol."""

    @abstractmethod
    def execute(self, step: Step) -> StepResult[T]:
        """Execute a step.

        Args:
            step: Step to execute.

        Returns:
            Step result.

        """
        ...


def _handle_step_success(state: AgentState, result: StepResult) -> StepResult:
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
        self.last_result: StepResult[T] | None = None

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

    def execute(self, step: Step) -> StepResult[T]:
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
    def _execute_step(self, step: Step) -> StepResult[T]:
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
    ) -> StepResult:
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


def validate_step_result(step: AgentStep, result: StepResult[Any]) -> None:
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


def execute_step_with_retry(state: AgentState, step: AgentStep, max_retries: int = 3) -> StepResult:
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
            last_result = StepResult(success=False, error=msg)
        except ValueError as e:
            # Handle validation errors
            msg = f"Validation error in step execution: {e}"
            last_result = StepResult(success=False, error=msg)
        except OSError as e:
            # Handle I/O errors
            msg = f"I/O error in step execution: {e}"
            last_result = StepResult(success=False, error=msg)
        except RuntimeError as e:
            # Handle runtime errors
            msg = f"Runtime error in step execution: {e}"
            last_result = StepResult(success=False, error=msg)

        retries += 1

    # Return the last result after all retries are exhausted
    return last_result if last_result else StepResult(success=False, error="Max retries exceeded")


class TaskBreakdownStep:
    """Task breakdown step.

    This class is responsible for breaking down a task into subtasks
    following the standardized task schema.
    """

    def __init__(self, agent_role: AgentRole) -> None:
        """Initialize task breakdown step.

        Args:
            agent_role: Role of the agent performing the breakdown.

        """
        self.agent_role = agent_role
        self.name = "task_breakdown"
        self.required_keys = ["task_description"]
        self.optional_keys = ["parent_task_id", "complexity", "priority"]
        self.retry_on_error = True
        self.max_retries = 3

    def __call__(self, state: AgentState, **kwargs: StepKwargs) -> StepResult:
        """Execute task breakdown step.

        Args:
            state: Current agent state.
            **kwargs: Additional arguments.
                - task_description: Description of the task to break down.
                - parent_task_id: Optional ID of the parent task.
                - complexity: Optional complexity of the task.
                - priority: Optional priority of the task.

        Returns:
            Step result containing a list of Task objects.

        """
        # Validate inputs
        self._validate_inputs(**kwargs)

        # Extract task information
        task_description = kwargs["task_description"]
        parent_task_id = kwargs.get("parent_task_id")
        complexity = kwargs.get("complexity", TaskComplexity.MODERATE)
        priority = kwargs.get("priority", TaskPriority.MEDIUM)

        # Create parent task if not provided
        if not parent_task_id:
            parent_task = Task(
                description=task_description,
                complexity=complexity,
                priority=priority,
                assigned_role=self.agent_role,
                created_at=datetime.now().timestamp(),
                updated_at=datetime.now().timestamp(),
            )
            parent_task_id = parent_task.task_id
            # Store parent task in state
            self._store_task_in_state(state, parent_task)

        # Get agent for task breakdown
        agent = state.get_agent_for_step(AgentStep.UNDERSTAND)

        # Create prompt for task breakdown
        prompt = self._create_task_breakdown_prompt(task_description, complexity, priority)

        # Create a proper Message object
        message = HumanMessage(content=prompt)

        # Process the message
        result = agent.process(message)

        if not result.success:
            return result

        # Parse the result into Task objects
        try:
            tasks = self._parse_tasks_from_result(result.data, parent_task_id)

            # Store tasks in state
            for task in tasks:
                self._store_task_in_state(state, task)

            # Update parent task with subtask IDs
            self._update_parent_task_with_subtasks(state, parent_task_id, [task.task_id for task in tasks])

            return StepResult(success=True, data=tasks)

        except Exception as e:
            return StepResult(success=False, error=f"Failed to parse tasks: {e!s}")

    def _validate_inputs(self, **kwargs: StepKwargs) -> None:
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

    def _create_task_breakdown_prompt(
        self,
        task_description: str,
        complexity: TaskComplexity,
        priority: TaskPriority,
    ) -> str:
        """Create prompt for task breakdown.

        Args:
            task_description: Description of the task to break down.
            complexity: Complexity of the task.
            priority: Priority of the task.

        Returns:
            Prompt for task breakdown.

        """
        return f"""
        You are tasked with breaking down a complex task into smaller, manageable subtasks.

        Task Description: {task_description}
        Task Complexity: {complexity.value}
        Task Priority: {priority.value}

        Please break down this task into subtasks following these guidelines:

        1. Each subtask should be clearly defined with a specific goal
        2. Subtasks should be independent where possible
        3. Identify dependencies between subtasks when necessary
        4. Assign appropriate complexity to each subtask
        5. Assign appropriate priority to each subtask

        Valid complexity values (use exactly as shown):
        - simple: Task can be directly executed without further decomposition
        - moderate: Task may benefit from some planning but is relatively straightforward
        - complex: Task requires significant planning and decomposition
        - very_complex: Task requires multiple levels of planning and decomposition

        Valid priority values (use exactly as shown):
        - low: Task is not urgent and can be deferred
        - medium: Task has normal priority
        - high: Task is important and should be prioritized
        - critical: Task is extremely important and should be done immediately

        Format your response as a JSON array of tasks with the following structure:

        ```json
        [
          {{
            "description": "Subtask description",
            "complexity": "simple|moderate|complex|very_complex",
            "priority": "low|medium|high|critical",
            "dependencies": [
              {{
                "task_index": 0,
                "description": "Dependency description",
                "is_blocking": true|false
              }}
            ]
          }},
          // Additional tasks...
        ]
        ```

        Ensure that the dependencies reference other tasks in the list by their index (0-based).
        """

    def _parse_tasks_from_result(self, result_data: Any, parent_task_id: UUID) -> list[Task]:
        """Parse tasks from result data.

        Args:
            result_data: Result data from agent.
            parent_task_id: ID of the parent task.

        Returns:
            List of Task objects.

        Raises:
            ValueError: If result data is invalid.

        """
        # Extract JSON from result if needed
        if isinstance(result_data, str):
            # Find JSON array in the string
            json_start = result_data.find("[")
            json_end = result_data.rfind("]") + 1

            if json_start == -1 or json_end == 0:
                msg = "No JSON array found in result"
                raise ValueError(msg)

            json_str = result_data[json_start:json_end]
            try:
                task_dicts = json.loads(json_str)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON in result: {e!s}"
                raise ValueError(msg)
        elif isinstance(result_data, list):
            task_dicts = result_data
        else:
            msg = f"Unexpected result data type: {type(result_data)}"
            raise ValueError(msg)

        # Create Task objects
        tasks = []
        task_ids = {}  # Map of index to task_id

        # First pass: create tasks without dependencies
        for i, task_dict in enumerate(task_dicts):
            # Convert complexity and priority to lowercase for enum lookup
            complexity_str = task_dict.get("complexity", "MODERATE").lower()
            priority_str = task_dict.get("priority", "MEDIUM").lower()

            task = Task(
                description=task_dict["description"],
                complexity=TaskComplexity(complexity_str),
                priority=TaskPriority(priority_str),
                parent_task_id=parent_task_id,
                assigned_role=self.agent_role,
                created_at=datetime.now().timestamp(),
                updated_at=datetime.now().timestamp(),
            )
            tasks.append(task)
            task_ids[i] = task.task_id

        # Second pass: add dependencies
        for i, task_dict in enumerate(task_dicts):
            if task_dict.get("dependencies"):
                for dep in task_dict["dependencies"]:
                    if "task_index" in dep and dep["task_index"] < len(tasks):
                        dep_task_id = task_ids[dep["task_index"]]
                    else:
                        # Create a new UUID for external dependencies
                        dep_task_id = uuid4()

                    dependency = TaskDependency(
                        task_id=dep_task_id,
                        description=dep["description"],
                        is_blocking=dep.get("is_blocking", True),
                    )
                    tasks[i].dependencies.append(dependency)

        return tasks

    def _store_task_in_state(self, state: AgentState, task: Task) -> None:
        """Store task in agent state.

        Args:
            state: Agent state.
            task: Task to store.

        """
        # Get existing tasks or create new list
        tasks = state.get_context("tasks", [])

        # Convert task to dictionary
        task_dict = {
            "task_id": str(task.task_id),
            "description": task.description,
            "complexity": task.complexity.value,
            "priority": task.priority.value,
            "status": task.status.value,
            "parent_task_id": str(task.parent_task_id) if task.parent_task_id else None,
            "subtasks": [str(subtask_id) for subtask_id in task.subtasks],
            "dependencies": [
                {
                    "task_id": str(dep.task_id),
                    "description": dep.description,
                    "is_blocking": dep.is_blocking,
                }
                for dep in task.dependencies
            ],
            "assigned_role": task.assigned_role.value if task.assigned_role else None,
            "assigned_agent_id": task.assigned_agent_id,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "completed_at": task.completed_at,
        }

        # Add or update task in list
        for i, existing_task in enumerate(tasks):
            if existing_task.get("task_id") == str(task.task_id):
                tasks[i] = task_dict
                break
        else:
            tasks.append(task_dict)

        # Store updated tasks list in state
        state.set_context("tasks", tasks)

    def _update_parent_task_with_subtasks(
        self,
        state: AgentState,
        parent_task_id: UUID,
        subtask_ids: list[UUID],
    ) -> None:
        """Update parent task with subtask IDs.

        Args:
            state: Agent state.
            parent_task_id: ID of the parent task.
            subtask_ids: List of subtask IDs.

        """
        tasks = state.get_context("tasks", [])

        for i, task in enumerate(tasks):
            if task.get("task_id") == str(parent_task_id):
                # Update subtasks list
                tasks[i]["subtasks"] = [str(subtask_id) for subtask_id in subtask_ids]
                break

        # Store updated tasks list in state
        state.set_context("tasks", tasks)
