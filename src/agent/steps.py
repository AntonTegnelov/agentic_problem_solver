"""Agent step processing module."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from src.agent.result import Result
from src.common_types.enums import AgentStatus, AgentStep
from src.config.base import ConfigError
from src.exceptions import ConfigError
from src.prompts import get_retry_prompt, get_step_prompt

if TYPE_CHECKING:
    from src.agent.agent_types import StepKwargs, StepResult
    from src.agent.state.base import AgentState

T = TypeVar("T")

__all__ = ["Step", "StepFunction"]

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
        state.status = AgentStatus.RUNNING
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
                max_retries = (
                    step.max_retries if step.max_retries is not None else state.config.max_retries
                )
                if state.retry_count <= max_retries:
                    return self.execute_step(step, state, **kwargs)

            # Update state on failure
            state.status = AgentStatus.FAILED
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


def execute_step_with_retry(state: AgentState, step: AgentStep, max_retries: int = 3) -> Result:
    """Execute step with retry mechanism.

    Args:
        state: Agent state.
        step: Step to execute.
        max_retries: Maximum number of retries.

    Returns:
        Step result.

    Raises:
        ConfigError: If no agent registered.

    """
    agent = state.get_agent()
    if not agent:
        msg = "No agent registered"
        raise ConfigError(msg)

    retries = 0
    last_result = None

    while retries < max_retries:
        try:
            prompt = get_step_prompt(step)
            result = agent.process(prompt)
            if result.success:
                return result
            last_result = result
        except Exception as e:
            msg = f"Error executing step: {e}"
            last_result = Result(success=False, error=msg)

        if retries < max_retries - 1:
            retry_prompt = get_retry_prompt(step, last_result.error)
            result = agent.process(retry_prompt)
            if result.success:
                return result
            last_result = result

        retries += 1

    return last_result
