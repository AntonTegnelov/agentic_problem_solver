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

import sys

from src.cli.main import cli


def usage() -> None:
    """Print usage instructions for the CLI.

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
    """Entry point for the examples module.

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


if __name__ == "__main__":
    main()
