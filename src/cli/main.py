"""Command-line interface for the agent."""

import click
from dotenv import load_dotenv
from typing import Optional
from src.agents.solver_agent import SolverAgent
from src.config import DEFAULT_LLM_CONFIG


@click.group()
def cli():
    """Agentic Problem Solver CLI."""
    load_dotenv()


@cli.command()
@click.argument("prompt", type=str)
@click.option(
    "--max-tokens",
    "-m",
    type=int,
    default=DEFAULT_LLM_CONFIG["max_tokens"],
    help="Maximum tokens to generate",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=DEFAULT_LLM_CONFIG["temperature"],
    help="Temperature for generation",
)
@click.option(
    "--stream/--no-stream", "-s", default=False, help="Stream output as it is generated"
)
async def solve(
    prompt: str, max_tokens: Optional[int], temperature: Optional[float], stream: bool
):
    """Solve a problem using the agent."""
    agent = SolverAgent("solver")

    # Update agent configuration with LLM parameters
    if max_tokens is not None or temperature is not None:
        config = {}
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        agent.update_config(config)

    if stream:
        async for chunk in agent.process_message_stream(prompt):
            click.echo(chunk, nl=False)
        click.echo()
    else:
        response = await agent.process_message(prompt)
        click.echo(response)


@cli.command()
def version():
    """Show version information."""
    click.echo("Agentic Problem Solver v0.1.0")


def main():
    """Entry point for the CLI."""
    cli(_anyio_backend="asyncio")
