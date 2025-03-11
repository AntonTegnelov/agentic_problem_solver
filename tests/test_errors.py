"""Tests for error handling."""

from typing import NoReturn

import pytest

from src.config.agent import AgentConfig
from src.config.llm import LLMConfig
from src.config.utils import load_config_from_env, load_env_var
from src.exceptions import (
    AgentError,
    APIKeyError,
    ConfigError,
    EmptyResponseError,
    InvalidModelError,
    ProcessingError,
    RetryError,
    TemperatureError,
    ValidationError,
)


def test_api_key_error() -> NoReturn:
    """Test APIKeyError."""
    # Test error message
    with pytest.raises(APIKeyError, match="API key is missing"):
        msg = "API key is missing"
        raise APIKeyError(msg)

    # Test inheritance
    with pytest.raises(ValueError):
        msg = "API key is missing"
        raise APIKeyError(msg)


def test_config_error() -> NoReturn:
    """Test ConfigError."""
    # Test error message
    with pytest.raises(ConfigError, match="Invalid configuration"):
        msg = "Invalid configuration"
        raise ConfigError(msg)

    # Test with agent config
    with pytest.raises(ConfigError, match="temperature must be less than 1"):
        AgentConfig(temperature=1.5)

    # Test with environment variables
    with pytest.raises(ConfigError, match="Prefix must end with underscore"):
        load_config_from_env("PREFIX")


def test_empty_response_error() -> NoReturn:
    """Test EmptyResponseError."""
    # Test error message
    with pytest.raises(EmptyResponseError, match="Empty response"):
        msg = "Empty response"
        raise EmptyResponseError(msg)

    # Test inheritance
    with pytest.raises(RuntimeError):
        msg = "Empty response"
        raise EmptyResponseError(msg)


def test_invalid_model_error() -> NoReturn:
    """Test InvalidModelError."""
    # Test error message
    with pytest.raises(InvalidModelError, match="Invalid model"):
        msg = "Invalid model"
        raise InvalidModelError(msg)

    # Test with LLM config
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        LLMConfig(model="")


def test_retry_error() -> NoReturn:
    """Test RetryError."""
    # Test error message
    with pytest.raises(RetryError, match="Max retries exceeded"):
        msg = "Max retries exceeded"
        raise RetryError(msg)

    # Test with agent config
    with pytest.raises(ConfigError, match="max_retries must be greater than 0"):
        AgentConfig(max_retries=-1)


def test_temperature_error() -> NoReturn:
    """Test TemperatureError."""
    # Test error message
    with pytest.raises(TemperatureError, match="Invalid temperature"):
        msg = "Invalid temperature"
        raise TemperatureError(msg)

    # Test with LLM config
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        LLMConfig(temperature=1.5)


def test_agent_error() -> NoReturn:
    """Test AgentError."""
    # Test error message
    with pytest.raises(AgentError, match="Agent error"):
        msg = "Agent error"
        raise AgentError(msg)

    # Test inheritance
    error = AgentError("Agent error")
    assert isinstance(error, Exception)


def test_validation_error() -> NoReturn:
    """Test ValidationError."""
    # Test error message
    with pytest.raises(ValidationError, match="Validation failed"):
        msg = "Validation failed"
        raise ValidationError(msg)

    # Test inheritance
    error = ValidationError("Validation failed")
    assert isinstance(error, Exception)


def test_processing_error() -> NoReturn:
    """Test ProcessingError."""
    # Test error message
    with pytest.raises(ProcessingError, match="Processing failed"):
        msg = "Processing failed"
        raise ProcessingError(msg)

    # Test inheritance
    error = ProcessingError("Processing failed")
    assert isinstance(error, Exception)


def test_error_chaining() -> None:
    """Test error chaining."""
    try:
        try:
            msg = "Original error"
            raise ValueError(msg)
        except ValueError as e:
            msg = "Configuration error"
            raise ConfigError(msg) from e
    except ConfigError as e:
        assert isinstance(e.__cause__, ValueError)
        assert str(e.__cause__) == "Original error"


def test_error_handling_in_env_vars(tmp_path) -> None:
    """Test error handling in environment variable loading."""
    env_file = tmp_path / ".env"

    # Test missing file
    with pytest.raises(ConfigError, match="No .* file found"):
        load_env_var("TEST_VAR", env_file=env_file, required=True)

    # Test missing variable
    env_file.write_text("OTHER_VAR=value\n")
    with pytest.raises(ConfigError, match="TEST_VAR not found in"):
        load_env_var("TEST_VAR", env_file=env_file, required=True)


def test_error_handling_in_config_loading(tmp_path) -> None:
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
