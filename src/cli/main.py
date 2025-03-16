"""Command line interface for the problem solver."""

import logging
import sys
from pathlib import Path
from typing import Any

import click

# Import hierarchical agent system
from src.agent.agent_types import Agent, create_architect_agent
from src.agent.state.base import InMemoryStateManager, StateManager
from src.common_types.error_types import AgentError, ConfigError
from src.common_types.result_types import Result
from src.config.agent import AgentConfig
from src.config.constants import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)
from src.config.utils import load_env_var
from src.llm_providers.providers.gemini import GeminiConfig, GeminiProvider
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


class TaskError(RuntimeError):
    """Raised when task processing fails."""

    def __init__(self, message: str) -> None:
        """Initialize error."""
        super().__init__(message)


def setup_agent(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[Agent[Any], object, StateManager]:
    """Set up the agent with the given parameters.

    Args:
        model: Model to use for generation.
        temperature: Temperature for generation.
        max_tokens: Maximum tokens to generate.

    Returns:
        A tuple of (agent, provider, state_manager).

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

        provider_config = GeminiConfig(api_key=api_key, model=model)
        provider = GeminiProvider(config=provider_config)

        # Create state manager
        state_manager = InMemoryStateManager()

        # Create configuration
        config = AgentConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create architect agent
        agent = create_architect_agent(
            provider=provider,
            state_manager=state_manager,
            config=config,
        )
    except ConfigError as e:
        # Convert ConfigError to ValueError with a descriptive message
        msg = f"Configuration error: {e}"
        raise ValueError(msg) from e
    else:
        return agent, provider, state_manager


def get_final_solution(agent: Agent[Any], result: Result[Any]) -> str:
    """Extract the final solution from a result.

    This method navigates through delegation chains if necessary to find
    the actual solution content rather than just delegation messages.

    Args:
        agent: The agent that produced the result.
        result: The result object containing the solution or delegation info.

    Returns:
        The final solution content as a string.

    """
    # If the result doesn't have data, return the error or a default message
    if not result.data:
        return str(result.error) if result.error else "No solution data available"

    # Check if the result data contains a delegation message
    data_str = str(result.data)
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


def handle_solution_retrieval(agent: Agent[Any], result: Result[Any]) -> None:
    """Handle solution retrieval and display.

    Args:
        agent: The agent that produced the result.
        result: The result object containing the solution or delegation info.

    Raises:
        ValueError: If solution retrieval fails.

    """
    try:
        # Get the final solution content instead of just the raw result data
        solution = get_final_solution(agent, result)
        click.echo(solution)
    except (ValueError, KeyError, AttributeError, TypeError) as e:
        # Handle specific errors that might occur during solution retrieval
        error_msg = f"Error retrieving solution: {e}"
        click.echo(error_msg, err=True)
        sys.exit(1)


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
def solve(
    task: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    """Solve a task using the architect agent.

    Args:
        task: The task to solve.
        model: Model to use for generation.
        temperature: Temperature for generation.
        max_tokens: Maximum tokens to generate.

    """
    try:
        # Set up the agent
        try:
            agent, _, _ = setup_agent(model, temperature, max_tokens)
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
            handle_solution_retrieval(agent, result)
        else:
            # Provide more detailed error information
            error_msg = f"Error: {format_error_message(result.error)}"
            click.echo(error_msg, err=True)
            sys.exit(1)
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


def main() -> None:
    """Entry point for the CLI."""
    cli()  # pragma: no cover


if __name__ == "__main__":
    main()  # pragma: no cover
