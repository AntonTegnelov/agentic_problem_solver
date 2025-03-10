"""Command line interface for the problem solver."""

import logging
import sys
from pathlib import Path

import click

from src.agent.state.base import InMemoryStateManager
from src.agents.solver_agent import SolverAgent
from src.config import AgentConfig
from src.llm_providers.providers.gemini import GeminiProvider
from src.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = Path("config.yaml")
DEFAULT_MODEL = (
    "gemini-2.0-flash-lite"  # Standard model for all operations - matches AgentConfig
)
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
    """Solve a programming task."""
    try:
        # Create configuration
        config = AgentConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create provider
        try:
            provider = GeminiProvider(model=model)
        except ValueError as e:
            if "API key" in str(e):
                click.echo(API_KEY_ERROR, err=True)
                sys.exit(1)
            raise

        # Create state manager
        state_manager = InMemoryStateManager()

        # Create agent
        agent = SolverAgent(
            provider=provider,
            state_manager=state_manager,
            config=config,
        )

        # Process task
        result = agent.process(task)
        click.echo(result)

    except Exception:
        logger.exception(TASK_ERROR)
        sys.exit(1)


def process_message(message: str) -> str:
    """Process a message using the solver agent.

    Args:
        message: The message to process.

    Returns:
        The processed response.

    Raises:
        TaskError: If an error occurs during processing.

    """
    try:
        agent = SolverAgent()
        return agent.process(message)
    except Exception as err:
        logger.exception(MESSAGE_ERROR)
        error_msg = f"{MESSAGE_ERROR}: {err}"
        raise TaskError(error_msg) from err


def main() -> None:
    """Run the CLI application."""
    cli()
