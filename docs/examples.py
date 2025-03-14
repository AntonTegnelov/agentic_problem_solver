"""Examples of using the Agentic Problem Solver CLI.

This module demonstrates common usage patterns for the CLI interface.

Examples:
    >>> from src.cli.main import cli
    >>> from click.testing import CliRunner
    >>> runner = CliRunner()

    Basic usage with default settings:
    >>> result = runner.invoke(cli, ['solve', 'What is the meaning of life?'])
    >>> result.exit_code
    0

    Using custom temperature and max tokens:
    >>> result = runner.invoke(cli, ['solve', '--temperature', '0.7',
    ...                             '--max-tokens', '500', 'Write a haiku'])
    >>> result.exit_code
    0

    Streaming output:
    >>> result = runner.invoke(cli, ['solve', '--stream', 'Tell me a story about AI'])
    >>> result.exit_code
    0

    Version information:
    >>> result = runner.invoke(cli, ['version'])
    >>> result.exit_code
    0
    >>> 'Agentic Problem Solver' in result.output
    True

"""

import asyncio
import sys

from src.cli.main import cli


def usage() -> None:
    r"""Print usage instructions for the CLI.

    Examples:
        >>> usage() # doctest: +NORMALIZE_WHITESPACE
        Usage: python -m src.cli.main [OPTIONS] COMMAND [ARGS]
        <BLANKLINE>
        Standard arguments:
        <BLANKLINE>
            --temperature      Temperature for generation (default: 0.7)
            --max-tokens      Maximum tokens to generate (default: 1000)
            --stream          Stream output as it is generated
            -h, --help        Display this help message
        <BLANKLINE>
        Examples:
        <BLANKLINE>
            # Launch the CLI with default settings
            python -m src.cli.main solve "What is the meaning of life?"
        <BLANKLINE>
            # Use custom temperature and max tokens
            python -m src.cli.main solve --temperature 0.7 --max-tokens 500 \\
                "Write a haiku"
        <BLANKLINE>
            # Stream the output as it's generated
            python -m src.cli.main solve --stream "Tell me a story about AI"
        <BLANKLINE>
            # Check version information
            python -m src.cli.main version

    """
    import click

    click.echo("""
Usage: python -m src.cli.main [OPTIONS] COMMAND [ARGS]

Standard arguments:

    --temperature      Temperature for generation (default: 0.7)
    --max-tokens      Maximum tokens to generate (default: 1000)
    --stream          Stream output as it is generated
    -h, --help        Display this help message

Examples:

    # Launch the CLI with default settings
    python -m src.cli.main solve "What is the meaning of life?"

    # Use custom temperature and max tokens
    python -m src.cli.main solve --temperature 0.7 --max-tokens 500 \\
        "Write a haiku"

    # Stream the output as it's generated
    python -m src.cli.main solve --stream "Tell me a story about AI"

    # Check version information
    python -m src.cli.main version
    """)


def main() -> None:
    r"""Entry point for the examples module.

    Examples:
        >>> import sys
        >>> sys.argv = ['examples.py']  # Simulate no arguments
        >>> try:
        ...     main()
        ... except SystemExit as e:
        ...     print(f"Exit code: {e.code}")
        Usage: python -m src.cli.main [OPTIONS] COMMAND [ARGS]
        <BLANKLINE>
        Standard arguments:
        <BLANKLINE>
            --temperature      Temperature for generation (default: 0.7)
            --max-tokens      Maximum tokens to generate (default: 1000)
            --stream          Stream output as it is generated
            -h, --help        Display this help message
        <BLANKLINE>
        Examples:
        <BLANKLINE>
            # Launch the CLI with default settings
            python -m src.cli.main solve "What is the meaning of life?"
        <BLANKLINE>
            # Use custom temperature and max tokens
            python -m src.cli.main solve --temperature 0.7 --max-tokens 500 \\
                "Write a haiku"
        <BLANKLINE>
            # Stream the output as it's generated
            python -m src.cli.main solve --stream "Tell me a story about AI"
        <BLANKLINE>
            # Check version information
            python -m src.cli.main version
        Exit code: 1

    """
    if len(sys.argv) == 1:
        usage()
        sys.exit(1)

    cli()


async def hierarchical_agent_example_async() -> None:
    """Use the hierarchical agent system directly with async/await.

    This example demonstrates how to use the hierarchical agent system
    with proper async/await syntax.
    """
    import os

    from src.agent.agent_types import create_architect_agent
    from src.agent.state.base import InMemoryStateManager
    from src.llm_providers.config.provider_config import GeminiConfig
    from src.llm_providers.providers.gemini import GeminiProvider
    from src.messages.creation import create_message

    # Set up the provider with your API key
    api_key = os.environ.get("GEMINI_API_KEY")
    provider_config = GeminiConfig(
        api_key=api_key,
        model="gemini-2.0-pro",
    )
    provider = GeminiProvider(config=provider_config)

    # Create the state manager and architect agent
    state_manager = InMemoryStateManager()
    agent = create_architect_agent(
        provider=provider,
        state_manager=state_manager,
    )

    # Process a task
    message = create_message(
        role="human",
        content="Write a function to calculate the factorial of a number in Python.",
    )

    # Process the message with the agent
    result = await agent.process(message)

    # Check if the processing was successful
    if result.success:
        pass
    else:
        pass


def hierarchical_agent_example() -> None:
    """Use the hierarchical agent system directly.

    This example demonstrates how to use the hierarchical agent system.

    Examples:
        >>> # This is a demonstration only and won't run in doctest
        >>> # hierarchical_agent_example()
        >>> print("Example of using hierarchical agents")
        Example of using hierarchical agents

    """
    import os

    from src.agent.agent_types import create_architect_agent
    from src.agent.state.base import InMemoryStateManager
    from src.llm_providers.config.provider_config import GeminiConfig
    from src.llm_providers.providers.gemini import GeminiProvider
    from src.messages.creation import create_message

    # Set up the provider with your API key
    api_key = os.environ.get("GEMINI_API_KEY")
    provider_config = GeminiConfig(
        api_key=api_key,
        model="gemini-2.0-pro",
    )
    provider = GeminiProvider(config=provider_config)

    # Create the state manager and architect agent
    state_manager = InMemoryStateManager()
    agent = create_architect_agent(
        provider=provider,
        state_manager=state_manager,
    )

    # Process a task
    message = create_message(
        role="human",
        content="Write a function to calculate the factorial of a number in Python.",
    )

    # Since this is a synchronous function but the agent.process is async,
    # we need to run it in an event loop
    try:
        # For Python 3.7+
        result = asyncio.run(agent.process(message))

        # Check if the processing was successful
        if result.success:
            pass
        else:
            pass
    except RuntimeError:
        # If we're already in an event loop (e.g., in Jupyter notebook)
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(agent.process(message))

        # Check if the processing was successful
        if result.success:
            pass
        else:
            pass


def advanced_hierarchical_example() -> None:
    """Use the hierarchical agent system with custom configuration.

    This example demonstrates how to use the hierarchical agent system
    with custom configuration options.
    """
    import os

    from src.agent.agent_types import create_architect_agent
    from src.agent.state.base import InMemoryStateManager
    from src.config.agent_config import AgentConfig
    from src.llm_providers.config.provider_config import GeminiConfig
    from src.llm_providers.providers.gemini import GeminiProvider
    from src.messages.creation import create_message

    # Set up the provider with your API key and custom configuration
    api_key = os.environ.get("GEMINI_API_KEY")
    provider_config = GeminiConfig(
        api_key=api_key,
        model="gemini-2.0-pro",
        temperature=0.7,
        max_tokens=1000,
    )
    provider = GeminiProvider(config=provider_config)

    # Create custom agent configuration
    agent_config = AgentConfig(
        max_tokens=1000,
        temperature=0.7,
    )

    # Create the state manager and architect agent
    state_manager = InMemoryStateManager()
    agent = create_architect_agent(
        provider=provider,
        state_manager=state_manager,
        config=agent_config,
    )

    # Process a task
    message = create_message(
        role="human",
        content="Write a function to calculate the factorial of a number in Python.",
    )

    # Run the agent in an event loop
    try:
        result = asyncio.run(agent.process(message))
        if result.success:
            pass
        else:
            pass
    except RuntimeError:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(agent.process(message))
        if result.success:
            pass
        else:
            pass


if __name__ == "__main__":
    main()
