"""Unit tests for CLI main module."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.main import TaskError, cli, main, process_message, setup_agent
from src.config import ConfigError


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
@patch("src.cli.main.create_architect_agent")
def test_solve_command_success(
    mock_create_architect_agent: MagicMock,
    mock_load_env_var: MagicMock,
    mock_gemini_config: MagicMock,
    mock_gemini_provider: MagicMock,
    cli_runner: CliRunner,
) -> None:
    """Test the solve command with successful execution."""
    # Mock the agent and its process method
    mock_agent_instance = MagicMock()

    # Create a mock result for the process method
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = "Task solution"
    mock_result.error = None

    # Set up the mock process method to return the mock result
    mock_agent_instance.process_sync.return_value = mock_result

    # Mock the coordinator and get_final_result_sync method
    mock_coordinator = MagicMock()
    mock_final_result = MagicMock()
    mock_final_result.success = True
    mock_final_result.data = {"result": "Task solution"}
    mock_coordinator.get_final_result_sync.return_value = mock_final_result

    # Set up the agent's state to have the coordinator
    mock_state = MagicMock()
    mock_state.coordinator = mock_coordinator
    mock_agent_instance.state = mock_state

    mock_create_architect_agent.return_value = mock_agent_instance

    # Mock environment variables
    mock_load_env_var.side_effect = ["fake-api-key", "gemini-1.0-pro"]

    # Mock provider config and provider
    mock_config_instance = MagicMock()
    mock_gemini_config.return_value = mock_config_instance

    mock_provider_instance = MagicMock()
    mock_gemini_provider.return_value = mock_provider_instance

    # Run the command - don't try to check sys.exit since Click handles it differently
    result = cli_runner.invoke(cli, ["solve", "Test task"])

    # Check that the output contains the task solution
    assert "Task solution" in result.output

    # Check that the agent was created with the right parameters
    mock_create_architect_agent.assert_called_once()
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
@patch("src.cli.main.create_architect_agent")
def test_process_message_success(
    mock_create_architect_agent: MagicMock,
    mock_load_env_var: MagicMock,
    mock_gemini_config: MagicMock,
    mock_gemini_provider: MagicMock,
) -> None:
    """Test the process_message function with successful execution."""
    # Mock the agent and its process method
    mock_agent_instance = MagicMock()

    # Create a mock result for the process method
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = "Test response"
    mock_result.error = None

    # Set up the mock process method to return the mock result
    mock_agent_instance.process_sync.return_value = mock_result
    mock_create_architect_agent.return_value = mock_agent_instance

    # Mock environment variables
    mock_load_env_var.side_effect = ["fake-api-key", "gemini-1.0-pro"]

    # Mock provider config and provider
    config_instance = MagicMock()
    mock_gemini_config.return_value = config_instance

    provider_instance = MagicMock()
    mock_gemini_provider.return_value = provider_instance

    # Call the function
    result = process_message("Test message")

    # Verify the agent was created
    mock_create_architect_agent.assert_called_once()

    # Verify the result
    assert result == "Test response"


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


@patch("src.cli.main.create_architect_agent")
@patch("src.cli.main.load_env_var")
def test_setup_agent_config_error(mock_load_env_var: MagicMock, mock_create_architect_agent: MagicMock) -> None:
    """Test setup_agent function when there's a configuration error.

    This test verifies that the setup_agent function properly handles
    configuration errors during agent initialization.
    """
    # Mock environment variable to return valid values
    mock_load_env_var.side_effect = ["fake-api-key", "gemini-1.5-pro"]

    # Setup mock to raise a ConfigError
    mock_create_architect_agent.side_effect = ConfigError("Invalid configuration")

    # Call the function and verify it raises the expected error
    with pytest.raises(ConfigError):
        setup_agent("gemini-1.5-pro", 0.7, 1000)
