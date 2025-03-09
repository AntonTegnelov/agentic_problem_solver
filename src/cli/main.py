"""Command-line interface for the Agentic Problem Solver."""

import asyncio
import os
import sys
from typing import NoReturn

import click
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agents.solver_agent import SolverAgent
from src.config import VERSION
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Error messages
TASK_ERROR = "Error processing task: {}"
API_KEY_ERROR = "GEMINI_API_KEY environment variable not set"

DEFAULT_TEMPERATURE = 0.7


@click.group()
def cli() -> None:
    """Agentic Problem Solver CLI."""
    load_dotenv()
    # Suppress gRPC warnings
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
    os.environ["GRPC_POLL_STRATEGY"] = "epoll1"


@cli.command()
def version() -> None:
    """Show version information."""
    click.echo(f"Version: {VERSION}")


@cli.command()
@click.argument("task")
@click.option("--stream", is_flag=True, help="Stream output")
@click.option("--temperature", type=float, help="Temperature for text generation")
@click.option("--max-tokens", type=int, help="Maximum number of tokens to generate")
@click.option("--model", type=str, help="Model name")
def solve(
    task: str,
    stream: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> None:
    """Solve a task using the agent."""
    try:
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("GEMINI_API_KEY environment variable not set")
            sys.exit(1)

        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if model is not None:
            config["model"] = model

        agent = SolverAgent(config)
        logger.info("Initialized solver agent with %s", agent.llm.__class__.__name__)

        if stream:
            try:
                asyncio.run(process_stream(agent, task))
            except Exception as err:
                error_msg = f"Error processing task: {err}"
                logger.error("Error processing task\n%s", err, exc_info=True)
                raise RuntimeError(error_msg) from err
        else:
            response = asyncio.run(agent.process_message(HumanMessage(content=task)))
            click.echo(response.content)

    except Exception as err:
        logger.error("Error processing task\n%s", err, exc_info=True)
        click.echo(f"Error: {err}", err=True)
        sys.exit(1)


async def process_stream(agent: SolverAgent, task: str) -> None:
    """Process task in streaming mode.

    Args:
        agent: Solver agent instance.
        task: Task to process.

    Raises:
        RuntimeError: If an error occurs during processing.
    """
    try:
        async for response in agent.process_message_stream(HumanMessage(content=task)):
            click.echo(response.content, nl=False)
    except Exception as err:
        error_msg = f"Error processing task: {err}"
        logger.error("Error processing message\n%s", err, exc_info=True)
        raise RuntimeError(error_msg) from err


def main() -> NoReturn:
    """Entry point for the CLI."""
    cli()
