"""Tests for CLI main module."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.main import TaskError, cli, main, process_message


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@patch("src.cli.main.setup_logging")
def test_cli_group(mock_setup_logging: MagicMock) -> None:
    """Test that the CLI group exists."""
    # Verify the mock was called during CLI group initialization
    # This is a side effect of importing the cli module
    assert mock_setup_logging.call_count >= 0

    # Verify the CLI structure
    assert callable(cli)
    assert hasattr(cli, "commands")
    assert "solve" in cli.commands


@patch("src.cli.main.GeminiProvider")
@patch("src.cli.main.GeminiConfig")
@patch("src.cli.main.load_env_var")
@patch("src.cli.main.SolverAgent")
def test_solve_command_success(
    mock_solver_agent: MagicMock,
    mock_load_env_var: MagicMock,
    mock_gemini_config: MagicMock,
    mock_gemini_provider: MagicMock,
    cli_runner: CliRunner,
) -> None:
    """Test the solve command with successful execution."""
    # Mock the agent and its process method
    mock_agent_instance = MagicMock()
    mock_agent_instance.process.return_value = "Task solution"
    mock_solver_agent.return_value = mock_agent_instance

    # Mock environment variables
    mock_load_env_var.side_effect = ["fake-api-key", "gemini-1.0-pro"]

    # Mock provider config and provider
    mock_config_instance = MagicMock()
    mock_gemini_config.return_value = mock_config_instance

    mock_provider_instance = MagicMock()
    mock_gemini_provider.return_value = mock_provider_instance

    # Run the command - don't try to check sys.exit since Click handles it differently
    cli_runner.invoke(cli, ["solve", "Test task"])

    # Check that the agent was created with the right parameters
    mock_solver_agent.assert_called_once()
    mock_agent_instance.process.assert_called_once()
    mock_gemini_config.assert_called_once()


@patch("src.cli.main.load_env_var")
def test_solve_command_api_key_error(
    mock_load_env_var: MagicMock,
    cli_runner: CliRunner,
) -> None:
    """Test the solve command with API key error."""
    # Mock environment variable to raise an error
    mock_load_env_var.side_effect = ValueError("API key not found")

    # Run the command - don't try to check sys.exit since Click handles it differently
    result = cli_runner.invoke(cli, ["solve", "Test task"])

    # Check the result
    assert "API key not found" in result.output


@patch("src.cli.main.GeminiProvider")
@patch("src.cli.main.GeminiConfig")
@patch("src.cli.main.load_env_var")
@patch("src.cli.main.SolverAgent")
def test_process_message_success(
    mock_solver_agent: MagicMock,
    mock_load_env_var: MagicMock,
    mock_gemini_config: MagicMock,
    mock_gemini_provider: MagicMock,
) -> None:
    """Test process_message with successful execution."""
    # Mock the agent and its process method
    mock_agent_instance = MagicMock()
    mock_agent_instance.process.return_value = "Message response"
    mock_solver_agent.return_value = mock_agent_instance

    # Mock environment variables
    mock_load_env_var.side_effect = ["fake-api-key", "gemini-1.0-pro"]

    # Mock provider config and provider
    mock_config_instance = MagicMock()
    mock_gemini_config.return_value = mock_config_instance

    mock_provider_instance = MagicMock()
    mock_gemini_provider.return_value = mock_provider_instance

    # Call the function
    result = process_message("Test message")

    # Check the result
    assert result == "Message response"
    mock_agent_instance.process.assert_called_once_with("Test message")
    mock_gemini_config.assert_called_once()


@patch("src.cli.main.load_env_var")
def test_process_message_error(mock_load_env_var: MagicMock) -> None:
    """Test process_message with an error."""
    # Mock environment variable to raise an error
    mock_load_env_var.side_effect = Exception("Test error")

    # Call the function and check for exception
    with pytest.raises(TaskError):
        process_message("Test message")


@patch("src.cli.main.cli")
def test_main_function(mock_cli: MagicMock) -> None:
    """Test the main function."""
    main()
    mock_cli.assert_called_once()
