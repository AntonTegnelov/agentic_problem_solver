"""Agent step processing module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .state.base import AgentState, AgentStatus
from .types import StepKwargs, StepResult


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


class StepExecutor(ABC):
    """Abstract base class for step executors."""

    @abstractmethod
    def execute_step(
        self, step: Step, state: AgentState, **kwargs: StepKwargs
    ) -> StepResult:
        """Execute a single step.

        Args:
            step: Step to execute.
            state: Current agent state.
            **kwargs: Additional arguments.

        Returns:
            Step result.
        """


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


class BaseStepExecutor(StepExecutor):
    """Base step executor implementation."""

    def execute_step(
        self, step: Step, state: AgentState, **kwargs: StepKwargs
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
                    step.max_retries
                    if step.max_retries is not None
                    else state.config.max_retries
                )
                if state.retry_count <= max_retries:
                    return self.execute_step(step, state, **kwargs)

            # Update state on failure
            state.status = AgentStatus.FAILED
            error_msg = f"Step '{step.name}' failed: {err}"
            raise RuntimeError(error_msg) from err
