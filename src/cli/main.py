import click
import asyncio
from dotenv import load_dotenv
from typing import Optional

@click.group()
def cli():
    """Agentic Problem Solver CLI."""
    load_dotenv()

@cli.command()
@click.argument('prompt', required=True)
@click.option('--max-tokens', '-m', type=int, help='Maximum tokens to generate')
@click.option('--temperature', '-t', type=float, help='Temperature for generation')
@click.option('--stream/--no-stream', default=True, help='Stream the output')
async def solve(prompt: str, max_tokens: Optional[int], temperature: Optional[float], stream: bool):
    """Solve a problem using AI agents."""
    click.echo(f"Processing prompt: {prompt}")
    # TODO: Implement agent logic
    click.echo("Solution will be implemented in next steps")

@cli.command()
def version():
    """Show version information."""
    click.echo("Agentic Problem Solver v0.1.0")

def main():
    """Main entry point."""
    cli(_anyio_backend="asyncio") 