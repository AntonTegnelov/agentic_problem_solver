"""End-to-end tests for the CLI.

These tests verify that the CLI works end-to-end, including the "APS solve" command.
"""

# ruff: noqa: S603, S607, BLE001
import os
import subprocess
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli.main import TaskError, cli, main, process_message
from src.common_types.error_types import AgentError


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


@pytest.fixture
def mock_env_vars_missing_api_key() -> Generator[None, None, None]:
    """Set up mock environment variables with missing API key."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_MODEL": "gemini-2.0-flash-lite",
        },
        clear=True,
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

    # Create a mock result for the process method
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = "Mocked response"
    mock_result.error = None

    # Set up the mock process method to return the mock result
    instance.process.return_value = mock_result

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


@patch("src.cli.main.create_architect_agent")
@patch("src.cli.main.GeminiProvider")
@patch("src.cli.main.GeminiConfig")
@patch("src.cli.main.load_env_var")
def test_cli_solve_command_with_options(
    mock_load_env_var: MagicMock,
    mock_gemini_config: MagicMock,
    mock_gemini_provider: MagicMock,
    mock_create_architect_agent: MagicMock,
) -> None:
    """Test the 'solve' command with custom options.

    This test verifies that custom model, temperature, and max tokens are correctly passed.
    """
    # Mock the load_env_var to return our test values
    mock_load_env_var.side_effect = lambda key: ("test_key" if key == "GEMINI_API_KEY" else "gemini-1.5-flash")

    # Mock the GeminiConfig to capture parameters
    mock_config_instance = MagicMock()
    mock_gemini_config.return_value = mock_config_instance

    # Mock the GeminiProvider
    mock_provider_instance = MagicMock()
    mock_gemini_provider.return_value = mock_provider_instance

    # Mock the ArchitectAgent
    instance = mock_create_architect_agent.return_value

    # Create a mock result for the process method
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = "Mocked response with options"
    mock_result.error = None

    # Set up the mock process method to return the mock result
    instance.process.return_value = mock_result

    # Setup the test to intercept AgentConfig creation
    with patch("src.cli.main.AgentConfig") as mock_agent_config:
        agent_config_instance = MagicMock()
        agent_config_instance.model = "gemini-1.5-flash"
        agent_config_instance.temperature = 0.5
        agent_config_instance.max_tokens = 2000
        mock_agent_config.return_value = agent_config_instance

        # Create a CLI runner
        from click.testing import CliRunner

        runner = CliRunner()

        # Run the command with custom options
        result = runner.invoke(
            cli,
            [
                "solve",
                "Test prompt with options",
                "--model",
                "gemini-1.5-flash",
                "--temperature",
                "0.5",
                "--max-tokens",
                "2000",
            ],
        )

        # Verify the command executed successfully
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Verify AgentConfig was created with the expected parameters
        mock_agent_config.assert_called_once_with(
            model="gemini-1.5-flash",
            temperature=0.5,
            max_tokens=2000,
        )

        # Verify ArchitectAgent was created with our config
        mock_create_architect_agent.assert_called_once()
        assert mock_create_architect_agent.call_args[1].get("config") == agent_config_instance

        # Check that the output contains our mocked response
        assert "Mocked response with options" in result.output


@patch("src.cli.main.load_env_var")
def test_cli_solve_command_missing_api_key(mock_load_env_var: MagicMock) -> None:
    """Test the 'solve' command with a missing API key.

    This test verifies that the CLI correctly handles missing API key errors.
    """
    # Setup mock to raise a ValueError for the API key
    mock_load_env_var.side_effect = ValueError("API key not found in environment variables")

    # Create a CLI runner
    from click.testing import CliRunner

    runner = CliRunner()

    # Run the command
    result = runner.invoke(cli, ["solve", "Test prompt"])

    # Verify the command failed with the expected error
    assert result.exit_code == 1, "Command should fail with exit code 1"
    assert "API key not found" in result.output, "Error message should mention API key"
    assert "cp .env.example .env" in result.output, "Error message should include instructions"


@patch("src.cli.main.create_architect_agent")
def test_cli_solve_command_agent_exception(mock_create_architect_agent: MagicMock) -> None:
    """Test the 'solve' command when the agent raises an exception.

    This test verifies that the CLI correctly handles exceptions from the agent.
    """
    # Setup mock to raise an exception during process
    instance = mock_create_architect_agent.return_value
    instance.process = AsyncMock(side_effect=AgentError("Agent process error"))

    # Create a CLI runner
    from click.testing import CliRunner

    runner = CliRunner()

    # Run the command
    result = runner.invoke(cli, ["solve", "Test prompt"])

    # Verify the command failed with the expected error
    assert result.exit_code == 1, "Command should fail with exit code 1"


@patch("src.cli.main.create_architect_agent")
def test_process_message_success(mock_create_architect_agent: MagicMock) -> None:
    """Test the process_message function with successful execution.

    This test verifies that process_message correctly processes messages via the agent.
    """
    # Setup mock
    instance = mock_create_architect_agent.return_value

    # Create a mock result for the process method
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = "Processed message result"
    mock_result.error = None

    # Set up the mock process method to return the mock result
    instance.process.return_value = mock_result

    # Call the function with env vars mock
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key", "GEMINI_MODEL": "test-model"}):
        result = process_message("Test message")

    # Verify the result
    assert result == "Processed message result"


@patch("src.cli.main.load_env_var")
def test_process_message_api_key_error(mock_load_env_var: MagicMock) -> None:
    """Test the process_message function with a missing API key.

    This test verifies that process_message correctly handles missing API key errors.
    """
    # Setup mock to raise a ValueError for the API key
    error_msg = "API key not found in environment variables"
    mock_load_env_var.side_effect = ValueError(error_msg)

    # Verify the function raises the expected error
    with pytest.raises(TaskError, match=f"Configuration error: {error_msg}"):
        process_message("Test message")


@patch("src.cli.main.create_architect_agent")
def test_process_message_agent_error(mock_create_architect_agent: MagicMock) -> None:
    """Test the process_message function when the agent raises an error.

    This test verifies that process_message correctly handles agent errors.
    """
    # Setup mock to raise an AgentError during process
    error_msg = "Agent process error"
    instance = mock_create_architect_agent.return_value
    instance.process.side_effect = AgentError(error_msg)

    # Mock environment variables
    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test_key", "GEMINI_MODEL": "test-model"}),
        pytest.raises(TaskError, match=f"Agent error: {error_msg}"),
    ):
        process_message("Test message")


@patch("src.cli.main.cli")
def test_main_function(mock_cli: MagicMock) -> None:
    """Test the main function.

    This test verifies that the main function correctly calls the CLI.
    """
    # Call the main function
    main()

    # Verify that the CLI was called
    mock_cli.assert_called_once()
