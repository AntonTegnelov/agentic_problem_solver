"""End-to-end tests for the CLI.

These tests verify that the CLI works end-to-end, including the "APS solve" command.
"""

# ruff: noqa: S603, S607, BLE001
import os
import subprocess
import warnings
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.cli.main import cli
from src.common_types.message_types import Message

# Suppress deprecation warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning, module="src.cli.main")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="src.agent.solver")


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Set up mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test_key",
            "GEMINI_MODEL": "gemini-2.0-flash-lite",
        },
    ):
        yield


def test_cli_solve_command_installed() -> None:
    """Test that the APS solve command is installed and can be called."""
    # Check if the command is available in the system
    try:
        # Use subprocess.run with check=True to raise an exception if the command fails
        result = subprocess.run(
            ["aps", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Check if the command exists and returns help information
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        assert "solve" in result.stdout, "The 'solve' command is not available in APS CLI"
    except FileNotFoundError:
        pytest.skip("APS command not found in PATH. Skipping test.")


@patch("src.cli.main.create_architect_agent")
def test_cli_solve_command_execution(mock_create_architect_agent: MagicMock) -> None:
    """Test the 'solve' command execution with mocked ArchitectAgent.

    This test verifies that the CLI correctly uses the hierarchical agent system.
    """
    # Setup mock
    instance = mock_create_architect_agent.return_value

    # Create a mock coroutine for the async process method
    async def mock_process(message: Message) -> str:
        # Use the message parameter to avoid linter warning
        assert message is not None
        return "Mocked response"

    instance.process = mock_process

    # Create a CLI runner
    from click.testing import CliRunner

    runner = CliRunner()

    # Run the command
    result = runner.invoke(cli, ["solve", "Test prompt"])

    # Verify the command executed successfully
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    # Verify the ArchitectAgent was created with the right parameters
    mock_create_architect_agent.assert_called_once()

    # Check that the output contains our mocked response
    assert "Mocked response" in result.output


@pytest.mark.skip(reason="Requires API keys and real LLM provider")
@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ or "GEMINI_MODEL" not in os.environ,
    reason="API keys not available",
)
def test_cli_solve_command_real_execution() -> None:
    """Test the 'solve' command with real execution.

    This test verifies that the CLI works with the actual LLM provider.
    """
    # Create a CLI runner
    from click.testing import CliRunner

    runner = CliRunner()

    # Run the command with a simple prompt
    result = runner.invoke(cli, ["solve", "What is 2+2?"])

    # Verify the command executed successfully
    assert result.exit_code == 0, f"Command failed with output: {result.output}"

    # Check that the output contains a reasonable response
    assert "4" in result.output or "four" in result.output.lower()
