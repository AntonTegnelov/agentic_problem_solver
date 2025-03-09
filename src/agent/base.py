"""Base agent implementation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.config import AgentConfig
from src.llm_providers import LLMProviderFactory

from .state.base import AgentState, AgentStatus, InMemoryStateManager, StateManager
from .steps import BaseStepExecutor, Step, StepExecutor
from .types import StepKwargs, StepResult


def _check_max_steps(state: AgentState) -> None:
    """Check if maximum number of steps has been exceeded.

    Args:
        state: Current agent state.

    Raises:
        RuntimeError: If maximum number of steps exceeded.
    """
    if state.step_count >= state.config.max_steps:
        error_msg = "Maximum number of steps exceeded"
        raise RuntimeError(error_msg)


def _handle_execution_failure(state: AgentState) -> None:
    """Handle agent execution failure.

    Args:
        state: Current agent state.

    Raises:
        RuntimeError: Always raised with error details.
    """
    error_msg = f"Agent execution failed: {state.error}"
    raise RuntimeError(error_msg) from state.error


def _handle_execution_success(
    state: AgentState, results: list[StepResult]
) -> list[StepResult]:
    """Handle successful agent execution.

    Args:
        state: Current agent state.
        results: List of step results.

    Returns:
        List of step results.
    """
    state.status = AgentStatus.COMPLETED
    return results


class BaseAgent(ABC):
    """Abstract base class for agents."""

    def __init__(
        self,
        config: AgentConfig,
        steps: Sequence[Step] | None = None,
        state_manager: StateManager | None = None,
        step_executor: StepExecutor | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            config: Agent configuration.
            steps: Sequence of execution steps.
            state_manager: State manager instance.
            step_executor: Step executor instance.
        """
        self.config = config
        self.steps = list(steps) if steps else []

        # Initialize components
        self.llm = LLMProviderFactory().get_provider()
        self.llm.update_config(config.llm.to_dict())

        # Create default state manager if none provided
        if state_manager is None:
            initial_state = AgentState(config=config)
            state_manager = InMemoryStateManager(state=initial_state)
        self.state_manager = state_manager

        # Create default step executor if none provided
        self.step_executor = step_executor or BaseStepExecutor()

    @property
    def state(self) -> AgentState:
        """Get current agent state.

        Returns:
            Current agent state.
        """
        return self.state_manager.get_state()

    def add_step(self, step: Step) -> None:
        """Add execution step.

        Args:
            step: Step to add.
        """
        self.steps.append(step)

    def add_steps(self, steps: Sequence[Step]) -> None:
        """Add multiple execution steps.

        Args:
            steps: Steps to add.
        """
        self.steps.extend(steps)

    def clear_steps(self) -> None:
        """Clear all execution steps."""
        self.steps.clear()

    @abstractmethod
    def setup(self) -> None:
        """Set up agent before execution."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up after execution."""

    def execute_step(self, step: Step, **kwargs: StepKwargs) -> StepResult:
        """Execute a single step.

        Args:
            step: Step to execute.
            **kwargs: Additional arguments.

        Returns:
            Step result.
        """
        return self.step_executor.execute_step(step, self.state, **kwargs)

    def execute(self, **kwargs: StepKwargs) -> list[StepResult]:
        """Execute all steps.

        Args:
            **kwargs: Additional arguments passed to each step.

        Returns:
            List of step results.

        Raises:
            RuntimeError: If execution fails.
        """
        error_msg: str
        results: list[StepResult] = []

        try:
            # Setup
            self.setup()
            self.state_manager.reset_state()

            # Execute steps
            for step in self.steps:
                _check_max_steps(self.state)
                result = self.execute_step(step, **kwargs)
                results.append(result)

                if self.state.status == AgentStatus.FAILED:
                    _handle_execution_failure(self.state)

            # Handle success
            return _handle_execution_success(self.state, results)

        except Exception as err:
            # Update state on failure
            self.state.error = err
            self.state.status = AgentStatus.FAILED
            error_msg = f"Agent execution failed: {err}"
            raise RuntimeError(error_msg) from err

        finally:
            # Always run cleanup
            self.cleanup()
