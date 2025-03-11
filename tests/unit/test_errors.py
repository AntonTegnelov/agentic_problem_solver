"""Test error handling."""

from pathlib import Path

import pytest

from src.agent.errors import AgentError
from src.config.utils import load_config_from_env, load_env_var
from src.exceptions import (
    APIKeyError,
    ConfigError,
    EmptyResponseError,
    InvalidModelError,
    ProcessingError,
    RetryError,
    TemperatureError,
    ValidationError,
)


def test_api_key_error() -> None:
    """Test APIKeyError."""
    # Test error message
    msg = "API key is missing"
    with pytest.raises(APIKeyError, match="API key is missing"):
        raise APIKeyError(msg)

    # Test inheritance
    assert issubclass(APIKeyError, Exception)


def test_config_error() -> None:
    """Test ConfigError."""
    # Test error message
    msg = "Invalid configuration"
    with pytest.raises(ConfigError, match="Invalid configuration"):
        raise ConfigError(msg)

    # Test with agent config
    from src.config.agent import AgentConfig

    with pytest.raises(ConfigError, match="temperature must be less than 1"):
        AgentConfig(temperature=1.5)


def test_empty_response_error() -> None:
    """Test EmptyResponseError."""
    # Test error message
    msg = "Empty response"
    with pytest.raises(EmptyResponseError, match="Empty response"):
        raise EmptyResponseError(msg)

    # Test inheritance
    assert issubclass(EmptyResponseError, Exception)


def test_invalid_model_error() -> None:
    """Test InvalidModelError."""
    # Test error message
    msg = "Invalid model"
    with pytest.raises(InvalidModelError, match="Invalid model"):
        raise InvalidModelError(msg)

    # Test with LLM config
    from src.config.llm import LLMConfig

    with pytest.raises(ValueError, match="Model name cannot be empty"):
        LLMConfig(model="")


def test_retry_error() -> None:
    """Test RetryError."""
    # Test error message
    msg = "Max retries exceeded"
    with pytest.raises(RetryError, match="Max retries exceeded"):
        raise RetryError(msg)

    # Test with agent config
    from src.config.agent import AgentConfig

    with pytest.raises(ConfigError, match="max_retries must be greater than 0"):
        AgentConfig(max_retries=-1)


def test_temperature_error() -> None:
    """Test TemperatureError."""
    # Test error message
    msg = "Invalid temperature"
    with pytest.raises(TemperatureError, match="Invalid temperature"):
        raise TemperatureError(msg)

    # Test inheritance
    assert issubclass(TemperatureError, Exception)


def test_agent_error() -> None:
    """Test AgentError."""
    # Test error message
    msg = "Agent error"
    with pytest.raises(AgentError, match="Agent error"):
        raise AgentError(msg)

    # Test inheritance
    assert issubclass(AgentError, Exception)


def test_validation_error() -> None:
    """Test ValidationError."""
    # Test error message
    msg = "Validation failed"
    with pytest.raises(ValidationError, match="Validation failed"):
        raise ValidationError(msg)

    # Test inheritance
    assert issubclass(ValidationError, Exception)


def test_processing_error() -> None:
    """Test ProcessingError."""
    # Test error message
    msg = "Processing failed"
    with pytest.raises(ProcessingError, match="Processing failed"):
        raise ProcessingError(msg)

    # Test inheritance
    assert issubclass(ProcessingError, Exception)


def _raise_value_error(msg: str) -> None:
    """Raise ValueError with the given message.

    Args:
        msg: Error message.

    Raises:
        ValueError: Always raised with the given message.

    """
    raise ValueError(msg)


def _raise_config_error_from_value_error() -> None:
    """Raise ConfigError from ValueError.

    Raises:
        ConfigError: Raised with a ValueError as the cause.

    """
    try:
        msg = "Original error"
        _raise_value_error(msg)
    except ValueError as e:
        msg = "Configuration error"
        raise ConfigError(msg) from e


def test_error_chaining() -> None:
    """Test error chaining."""
    with pytest.raises(ConfigError) as excinfo:
        _raise_config_error_from_value_error()

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert str(excinfo.value.__cause__) == "Original error"


def test_error_handling_in_env_vars(tmp_path: Path) -> None:
    """Test error handling in environment variable loading."""
    env_file = tmp_path / ".env"

    # Test missing file
    with pytest.raises(ConfigError, match="No .* file found"):
        load_env_var("TEST_VAR", env_file=env_file, required=True)

    # Test missing variable
    env_file.write_text("OTHER_VAR=value\n")
    with pytest.raises(ConfigError, match="TEST_VAR not found in"):
        load_env_var("TEST_VAR", env_file=env_file, required=True)


def test_error_handling_in_config_loading(tmp_path: Path) -> None:
    """Test error handling in configuration loading."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PREFIX_INVALID=value\nPREFIX_NESTED__KEY=value\n",
    )

    # Test empty prefix
    with pytest.raises(ConfigError, match="Prefix cannot be empty"):
        load_config_from_env("")

    # Test invalid prefix
    with pytest.raises(ConfigError, match="Prefix must end with underscore"):
        load_config_from_env("PREFIX")
