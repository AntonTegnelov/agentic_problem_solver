"""Command line interface for the problem solver."""

import asyncio
import logging
import sys
from pathlib import Path

import click

# Import hierarchical agent system
from src.agent.agent_types import create_architect_agent
from src.agent.state.base import InMemoryStateManager
from src.common_types.error_types import AgentError
from src.config import AgentConfig
from src.config.utils import load_env_var
from src.llm_providers.config.provider_config import GeminiConfig
from src.llm_providers.providers.gemini import GeminiProvider
from src.messages.creation import create_message
from src.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = Path("config.yaml")
DEFAULT_MODEL = "gemini-2.0-flash-lite"  # Standard model for all operations - matches AgentConfig
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

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
) -> tuple:
    """Set up the agent with provider and configuration.

    Args:
        model: Model to use for generation.
        temperature: Temperature for generation.
        max_tokens: Maximum tokens to generate.

    Returns:
        Tuple containing the agent and any other resources.

    Raises:
        ValueError: If API key is not found or other configuration errors.

    """
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

    return agent, provider, state_manager


@click.group()
def cli() -> None:
    """Agentic Problem Solver CLI."""
    setup_logging(level=logging.INFO)


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

        # Process task - create a human message and run it through asyncio
        message = create_message(role="human", content=task)
        result = asyncio.run(agent.process(message))
        click.echo(result)

    except AgentError as e:
        logger.exception("Agent error")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception:
        logger.exception(TASK_ERROR)
        click.echo("An unexpected error occurred. Check logs for details.", err=True)
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

        # Create a human message and run it through asyncio
        human_message = create_message(role="human", content=message)
        result = asyncio.run(agent.process(human_message))
        return str(result)
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
