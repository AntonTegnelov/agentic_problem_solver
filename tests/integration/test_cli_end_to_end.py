"""End-to-end tests for the CLI.

These tests verify that the CLI works end-to-end, including the "APS solve" command.
"""

# ruff: noqa: S603, S607, BLE001
import os
import subprocess
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.cli.main import cli


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


@patch("src.cli.main.SolverAgent")
def test_cli_solve_command_execution(mock_solver_agent: MagicMock) -> None:
    """Test that the APS solve command executes correctly with mocked dependencies."""
    from click.testing import CliRunner

    # Set up the mock
    mock_agent_instance = mock_solver_agent.return_value
    mock_agent_instance.process.return_value = "Task solution"

    # Create a CLI runner
    runner = CliRunner()

    # Run the command
    result = runner.invoke(cli, ["solve", "Create a simple calculator"])

    # Check that the command executed successfully
    assert result.exit_code == 0

    # Check that the agent was called with the correct task
    mock_agent_instance.process.assert_called_once()
    args, _ = mock_agent_instance.process.call_args
    assert "Create a simple calculator" in args[0]


@pytest.mark.skipif(
    "GEMINI_API_KEY" not in os.environ or "GEMINI_MODEL" not in os.environ,
    reason="API keys not available",
)
def test_cli_solve_command_real_execution() -> None:
    """Test the APS solve command with real execution (requires API keys)."""
    # This test will be skipped if API keys are not available

    # Use a simple task that should complete quickly
    task = "What is 2+2?"

    try:
        # Run the command through the CLI
        result = subprocess.run(
            ["python", "-m", "src", "solve", task],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,  # Set a timeout to avoid hanging tests
        )

        # Check that the command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"

        # Check that we got some output
        assert result.stdout.strip(), "No output was returned from the command"

    except subprocess.TimeoutExpired:
        pytest.fail("Command timed out")
    except Exception as e:
        pytest.fail(f"Command failed with exception: {e}")
