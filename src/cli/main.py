"""Command line interface for the problem solver."""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import click

from src.agent.agent_types import create_architect_agent
from src.agent.agent_types.agent_types import Agent
from src.agent.state.base import InMemoryStateManager, StateManager
from src.common_types.error_types import AgentError, ConfigError
from src.common_types.result_types import Result
from src.config.agent import AgentConfig
from src.config.constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.config.utils import load_env_var
from src.llm_providers.config.provider_config import GeminiConfig
from src.llm_providers.providers.gemini import GeminiProvider
from src.messages.creation import create_message
from src.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = Path("config.yaml")
DEFAULT_MAX_TOKENS = 1000  # Default max tokens for generation

# Error messages
TASK_ERROR = "Error processing task"
MESSAGE_ERROR = "Error processing message"
API_KEY_ERROR = """
API key not found. Please follow these steps:

1. Copy .env.example to .env in the project root:
   cp .env.example .env

2. Get your API key from: https://makersuite.google.com/app/apikey

3. Add your API key to .env:
   GEMINI_API_KEY=your_api_key_here

4. Try running the command again
"""

# Define a type variable for the coordinator
CoordinatorType = TypeVar("CoordinatorType")


class TaskError(RuntimeError):
    """Raised when task processing fails."""

    def __init__(self, message: str) -> None:
        """Initialize error."""
        super().__init__(message)


@dataclass
class ModelConfig:
    """Configuration for model generation."""

    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def setup_agent(
    model_config: ModelConfig | str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[Agent[Any], object, StateManager]:
    """Set up an agent with the given model and parameters.

    Args:
        model_config: Configuration for the model or model name string.
        temperature: Temperature for generation (used only if model_config is a string or None).
        max_tokens: Maximum tokens to generate (used only if model_config is a string or None).

    Returns:
        A tuple containing the agent, provider, and state manager.

    Raises:
        ValueError: If the provider cannot be created.

    """
    # Handle backward compatibility with old function signature
    if isinstance(model_config, str):
        model_config = ModelConfig(model=model_config, temperature=temperature, max_tokens=max_tokens)
    elif model_config is None:
        model_config = ModelConfig(temperature=temperature, max_tokens=max_tokens)

    # Create the provider
    try:
        provider = create_provider(
            model=model_config.model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
        )
    except ValueError as e:
        msg = f"Failed to create provider: {e}"
        raise ValueError(msg) from e

    # Get the state manager
    state_manager = get_state_manager()

    # Create agent configuration
    config = AgentConfig(
        model=model_config.model,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
    )

    # Create the agent
    agent = create_architect_agent(
        provider=provider,
        state_manager=state_manager,
        config=config,
    )

    return agent, provider, state_manager


def _get_coordinator_from_agent(agent: Agent[Any]) -> CoordinatorType | None:
    """Extract the coordinator from an agent's state.

    Args:
        agent: The agent to extract the coordinator from.

    Returns:
        The coordinator if found, None otherwise.

    """
    try:
        # Check if the agent has a state with a coordinator
        if hasattr(agent, "state") and agent.state:
            # Try different ways to access the coordinator
            if hasattr(agent.state, "coordinator"):
                return cast(CoordinatorType, agent.state.coordinator)
            if hasattr(agent.state, "get_coordinator"):
                return cast(CoordinatorType, agent.state.get_coordinator())
            if hasattr(agent.state, "agent_registry") and hasattr(agent.state.agent_registry, "coordinator"):
                return cast(CoordinatorType, agent.state.agent_registry.coordinator)
    except (AttributeError, TypeError) as e:
        # Log the error but continue with fallback methods
        logging.debug("Error accessing coordinator: %s", str(e))

    return None


def _get_result_from_coordinator(coordinator: CoordinatorType, agent: Agent[Any]) -> str | None:
    """Get the final result using the coordinator.

    Args:
        coordinator: The coordinator to use.
        agent: The agent that produced the result.

    Returns:
        The final result as a string if successful, None otherwise.

    """
    try:
        # Get the agent ID
        agent_id = agent.get_agent_id()

        # Use the coordinator to get the final result
        final_result = coordinator.get_final_result_sync(agent_id)  # type: ignore[attr-defined]

        if final_result.success and final_result.data:
            # Extract the result from the data
            if isinstance(final_result.data, dict) and "result" in final_result.data:
                return str(final_result.data["result"])
            return str(final_result.data)
    except (AttributeError, ValueError, TypeError, KeyError, RuntimeError) as e:
        # Log the error but continue with fallback methods
        logging.debug("Error using coordinator to get final result: %s", str(e))

    return None


def _handle_delegation(agent: Agent[Any], data_str: str) -> str | None:
    """Handle delegation by checking child agents for results.

    Args:
        agent: The agent that delegated the task.
        data_str: The result data string.

    Returns:
        The final solution from a child agent if found, None otherwise.

    """
    if "delegated" in data_str.lower() and agent.get_child_ids():
        # This is likely a delegation message, try to get results from child agents
        child_results = agent.collect_results_from_children()
        if child_results:
            # Return the first non-empty result from child agents
            for child_id, child_result in child_results.items():
                if child_result.success and child_result.data:
                    # Recursively get the final solution from the child result
                    child_agent = agent.state.get_agent(child_id)
                    if child_agent:
                        return get_final_solution(child_agent, child_result)

    return None


def get_final_solution(agent: Agent[Any], result: Result[Any]) -> str:
    """Extract the final solution from a result.

    This method navigates through delegation chains if necessary to find
    the actual solution content rather than just delegation messages.

    Args:
        agent: The agent that produced the result.
        result: The result object containing the solution or delegation info.

    Returns:
        The final solution content as a string.

    Raises:
        ValueError: If solution retrieval fails.

    """
    # If the result doesn't have data, return the error or a default message
    if not result.success or not result.data:
        return str(result.error) if result.error else "No solution data available"

    # Try to get the coordinator from the agent's state
    coordinator = _get_coordinator_from_agent(agent)

    # If we have a coordinator, use the new method to get the final result
    if coordinator and hasattr(coordinator, "get_final_result"):
        coordinator_result = _get_result_from_coordinator(coordinator, agent)
        if coordinator_result:
            return coordinator_result

    # Fallback: Check if the result data contains a delegation message
    data_str = str(result.data)
    delegation_result = _handle_delegation(agent, data_str)
    if delegation_result:
        return delegation_result

    # If no delegation or no child results, return the original result data
    return data_str


@click.group()
def cli() -> None:
    """Agentic Problem Solver CLI."""
    setup_logging(level=logging.INFO)


def format_error_message(error: Exception | None) -> str:
    """Format an error message based on the error type.

    Args:
        error: The error to format.

    Returns:
        A formatted error message.

    """
    if error is None:
        return "Unknown error occurred during processing"

    if isinstance(error, AgentError):
        return f"Agent error - {error}"
    if "timeout" in str(error).lower() or isinstance(error, TimeoutError):
        return f"Request timed out - {error}"
    if "connection" in str(error).lower() or isinstance(error, ConnectionError):
        return f"Connection error - {error}"

    return str(error)


def handle_solution_retrieval(agent: Agent[Any], result: Result[Any], *, verbose: bool = False) -> None:
    """Handle solution retrieval and display.

    Args:
        agent: The agent that produced the result.
        result: The result object containing the solution or delegation info.
        verbose: Whether to show verbose output including delegation messages.

    Raises:
        ValueError: If solution retrieval fails.

    """
    try:
        # Get the final solution content instead of just the raw result data
        solution = get_final_solution(agent, result)

        # If solution is a JSON string, parse it to extract the actual solution content
        if solution.startswith("{") and "solution" in solution:
            try:
                parsed = json.loads(solution)
                if isinstance(parsed, dict) and "solution" in parsed:
                    solution = parsed["solution"]
            except json.JSONDecodeError:
                pass

        if verbose:
            # In verbose mode, add a separator between delegation messages and the solution
            click.echo("\n" + "-" * 80 + "\n")
            # Display the full solution
            click.echo(solution)
        else:
            # Special handling for HTML-like tags
            if "<code>" in solution and "</code>" in solution:
                code_only = solution.replace("<code>", "").replace("</code>", "").strip()
            else:
                # In non-verbose mode, extract only the code without explanations
                code_only = extract_code_only(solution)
            click.echo(code_only)
    except (ValueError, KeyError, AttributeError, TypeError) as e:
        # Handle specific errors that might occur during solution retrieval
        error_msg = f"Error retrieving solution: {e}"
        click.echo(error_msg, err=True)
        sys.exit(1)


def extract_code_only(solution: str) -> str:
    """Extract only the code from a solution, removing explanations and instructions.

    Args:
        solution: The full solution text.

    Returns:
        The code-only portion of the solution.

    """
    # Check if the solution contains a code block with triple backticks
    import re

    # Try to find Python code blocks with triple backticks
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", solution, re.DOTALL)

    if code_blocks:
        # Return the first code block found
        return code_blocks[0].strip()

    # If no code blocks with backticks, try to find indented code blocks
    lines = solution.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        # Skip lines that are likely explanations or instructions
        if any(
            keyword in line.lower()
            for keyword in [
                "key improvements",
                "how to run",
                "explanation",
                "improvements",
                "features",
                "functionality",
                "save the code",
                "run the code",
            ]
        ):
            continue

        # If line starts with code-like patterns, include it
        if (
            line.strip().startswith("def ")
            or line.strip().startswith("class ")
            or line.strip().startswith("import ")
            or line.strip().startswith("from ")
            or line.strip().startswith("if ")
            or line.strip().startswith("while ")
            or line.strip().startswith("for ")
            or line.strip().startswith("print(")
            or line.strip().startswith("return ")
            or line.strip().startswith("#")
            or "=" in line
            or in_code
        ):
            code_lines.append(line)
            in_code = True

    if code_lines:
        # Join the code lines and ensure no trailing newline
        code = "\n".join(code_lines)
        # Remove any trailing newlines to match expected test output
        return code.rstrip()

    # If we couldn't extract code specifically, return the original solution
    return solution


def _solve_task(
    task: str,
    model_config: ModelConfig,
    *,
    verbose: bool = False,
) -> None:
    """Solve a task using the provided model configuration.

    Args:
        task: The task to solve.
        model_config: Configuration for the model.
        verbose: Whether to show verbose output.

    Raises:
        ValueError: If an error occurs during setup.
        AgentError: If an error occurs during processing.

    """
    try:
        # Set up the agent
        agent, _, _ = setup_agent(model_config)
    except ValueError as e:
        if "API key" in str(e):
            click.echo(API_KEY_ERROR, err=True)
            sys.exit(1)
        raise

    # Process task - create a human message and process it
    message = create_message(role="human", content=task)
    result = agent.process_sync(message)

    # Extract data from the Result object
    if result.success:
        handle_solution_retrieval(agent, result, verbose=verbose)
    else:
        # Provide more detailed error information
        error_msg = f"Error: {format_error_message(result.error)}"
        click.echo(error_msg, err=True)
        sys.exit(1)


def _create_model_config(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ModelConfig:
    """Create a model configuration.

    Args:
        model: Model to use for generation.
        temperature: Temperature for generation.
        max_tokens: Maximum tokens to generate.

    Returns:
        A model configuration.

    """
    return ModelConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _solve_with_options(
    task: str,
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    verbose: bool,
) -> None:
    """Solve a task with the given options.

    Args:
        task: The task to solve.
        model: Model to use for generation.
        temperature: Temperature for generation.
        max_tokens: Maximum tokens to generate.
        verbose: Whether to show verbose output.

    """
    # Create model configuration
    model_config = _create_model_config(model, temperature, max_tokens)

    # Solve the task
    _solve_task(task, model_config, verbose=verbose)


@cli.command()
@click.argument("task")
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help="Model to use for generation.",
)
@click.option(
    "--temperature",
    default=DEFAULT_TEMPERATURE,
    type=float,
    help="Temperature for generation.",
)
@click.option(
    "--max-tokens",
    default=DEFAULT_MAX_TOKENS,
    type=int,
    help="Maximum tokens to generate.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with maximum logging.",
)
def solve(task: str, **kwargs: dict[str, object]) -> None:
    """Solve a task using the architect agent.

    Args:
        task: The task to solve.
        **kwargs: Additional arguments for the solver.
            model: Model to use for generation.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.
            verbose: Enable verbose logging.
            debug: Enable debug mode with maximum logging.

    """
    # Extract parameters from kwargs with defaults
    model = kwargs.get("model", DEFAULT_MODEL)
    temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
    verbose = kwargs.get("verbose", False)
    debug = kwargs.get("debug", False)

    # Set up logging based on verbosity
    if debug:
        setup_logging(level=logging.DEBUG, verbose=True)
    elif verbose:
        setup_logging(verbose=True)
    else:
        # In non-verbose mode, only show warnings and errors
        setup_logging(level=logging.WARNING)

    try:
        _solve_with_options(task, model, temperature, max_tokens, verbose=verbose)
    except (ValueError, KeyError, AttributeError, TypeError, AgentError) as e:
        # Catch specific errors
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except (ConnectionError, TimeoutError) as e:
        # Handle network-related errors
        click.echo(f"Network error: {e}", err=True)
        sys.exit(1)


def process_message(
    message: str,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Process a message using the architect agent.

    Args:
        message: The message to process.
        model: Model to use for generation. If None, uses environment variable or default.
        temperature: Temperature for generation.
        max_tokens: Maximum tokens to generate.

    Returns:
        The processed response.

    Raises:
        TaskError: If an error occurs during processing.

    """
    try:
        # Use the model from parameters or default
        model_to_use = model or DEFAULT_MODEL

        # Set up the agent
        agent, _, _ = setup_agent(model_to_use, temperature, max_tokens)

        # Create a human message and process it
        human_message = create_message(role="human", content=message)
        result = agent.process_sync(human_message)

        # Extract data from the Result object
        if result.success:
            return result.data
        error_msg = f"Processing failed: {result.error}"
        raise TaskError(error_msg)  # noqa: TRY301

    except ValueError as err:
        logger.exception("Configuration error")
        # Extract the original error message if it's a nested error from setup_agent
        if "Failed to create provider: " in str(err):
            original_error = str(err).replace("Failed to create provider: ", "")
            error_msg = f"Configuration error: {original_error}"
        else:
            error_msg = f"Configuration error: {err}"
        raise TaskError(error_msg) from err
    except AgentError as err:
        logger.exception("Agent error")
        error_msg = f"Agent error: {err}"
        raise TaskError(error_msg) from err
    except Exception as err:
        logger.exception(MESSAGE_ERROR)
        error_msg = f"{MESSAGE_ERROR}: {err}"
        raise TaskError(error_msg) from err


def create_provider(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> object:
    """Create a provider with the given parameters.

    Args:
        model: Model to use for generation.
        temperature: Temperature for generation.
        max_tokens: Maximum tokens to generate.

    Returns:
        A provider instance.

    Raises:
        ValueError: If API key is not found or other configuration errors.

    """
    try:
        # Create provider
        api_key = load_env_var("GEMINI_API_KEY")

        # Only use the environment variable if no model is explicitly provided
        if model == DEFAULT_MODEL:
            env_model = load_env_var("GEMINI_MODEL", default=DEFAULT_MODEL)
            model = env_model

        provider_config = GeminiConfig(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return GeminiProvider(config=provider_config)

    except ConfigError as e:
        # Convert ConfigError to ValueError with a descriptive message
        msg = f"Configuration error: {e}"
        raise ValueError(msg) from e


def get_state_manager() -> StateManager:
    """Get a state manager instance.

    Returns:
        A state manager instance.

    """
    return InMemoryStateManager()


def main() -> None:
    """Entry point for the CLI."""
    cli()  # pragma: no cover


if __name__ == "__main__":
    main()  # pragma: no cover
