"""Tests for configuration system."""

from dataclasses import dataclass
from typing import Any

import pytest

from src.config.agent import AgentConfig, NumericValidation
from src.config.base import BaseConfig
from src.config.llm import LLMConfig
from src.config.utils import load_config_from_env, load_env_var
from src.exceptions import ConfigError


@dataclass
class TestConfig(BaseConfig):
    """Test configuration class."""

    name: str
    value: int
    nested: dict[str, Any]


def test_base_config_to_dict() -> None:
    """Test BaseConfig.to_dict()."""
    config = TestConfig(name="test", value=42, nested={"key": "value"})
    config_dict = config.to_dict()
    assert config_dict == {
        "name": "test",
        "value": 42,
        "nested": {"key": "value"},
    }


def test_base_config_from_dict() -> None:
    """Test BaseConfig.from_dict()."""
    data = {
        "name": "test",
        "value": 42,
        "nested": {"key": "value"},
    }
    config = TestConfig.from_dict(data)
    assert config.name == "test"
    assert config.value == 42
    assert config.nested == {"key": "value"}


def test_base_config_update() -> None:
    """Test BaseConfig.update()."""
    config = TestConfig(name="test", value=42, nested={"key": "value"})
    config.update({"name": "updated", "value": 100})
    assert config.name == "updated"
    assert config.value == 100
    assert config.nested == {"key": "value"}


def test_agent_config_validation() -> None:
    """Test AgentConfig validation."""
    # Test valid configuration
    config = AgentConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=1000,
        max_retries=3,
        retry_delay=1.0,
        timeout=30.0,
        task_timeout=60,
        max_steps=10,
    )
    assert config.model == "test-model"

    # Test invalid temperature
    with pytest.raises(ConfigError, match="temperature must be less than 1"):
        AgentConfig(temperature=1.5)

    # Test invalid max_retries
    with pytest.raises(ConfigError, match="max_retries must be greater than 0"):
        AgentConfig(max_retries=-1)

    # Test invalid task_timeout
    with pytest.raises(ConfigError, match="task_timeout must be greater than 1"):
        AgentConfig(task_timeout=0)

    # Test invalid max_steps
    with pytest.raises(ConfigError, match="max_steps must be greater than 1"):
        AgentConfig(max_steps=0)


def test_agent_config_numeric_validation() -> None:
    """Test AgentConfig numeric field validation."""
    config = AgentConfig()

    # Test min value validation
    with pytest.raises(ConfigError, match="test must be greater than 5"):
        config._validate_numeric_field("test", 3, min_value=5)

    # Test max value validation
    with pytest.raises(ConfigError, match="test must be less than 10"):
        config._validate_numeric_field("test", 15, max_value=10)

    # Test zero validation
    with pytest.raises(ConfigError, match="test must be non-zero"):
        config._validate_numeric_field("test", 0, validation_type=NumericValidation.DISALLOW_ZERO)

    # Test allow zero
    config._validate_numeric_field("test", 0, validation_type=NumericValidation.ALLOW_ZERO)


def test_llm_config_validation() -> None:
    """Test LLMConfig validation."""
    # Test valid configuration
    config = LLMConfig(
        model="test-model",
        temperature=0.7,
        max_output_tokens=100,
        top_p=0.9,
        top_k=40,
    )
    assert config.model == "test-model"

    # Test invalid temperature
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        LLMConfig(temperature=1.5)

    # Test invalid max_output_tokens
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        LLMConfig(max_output_tokens=0)

    # Test empty model
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        LLMConfig(model="")

    # Test invalid top_p
    with pytest.raises(ValueError, match="Top P must be between 0 and 1"):
        LLMConfig(top_p=1.5)

    # Test invalid top_k
    with pytest.raises(ValueError, match="Top K must be positive"):
        LLMConfig(top_k=0)


def test_llm_config_dict() -> None:
    """Test LLMConfig.dict()."""
    config = LLMConfig(
        model="test-model",
        temperature=0.7,
        max_output_tokens=100,
        top_p=0.9,
        top_k=40,
        extra_params={"custom_param": "value"},
    )
    config_dict = config.dict()
    assert config_dict["model"] == "test-model"
    assert config_dict["temperature"] == 0.7
    assert config_dict["max_output_tokens"] == 100
    assert config_dict["top_p"] == 0.9
    assert config_dict["top_k"] == 40
    assert config_dict["custom_param"] == "value"


def test_load_env_var(tmp_path) -> None:
    """Test load_env_var utility."""
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=test_value\n")

    # Test with existing variable
    assert load_env_var("TEST_VAR", env_file=env_file) == "test_value"

    # Test with default value
    assert load_env_var("NONEXISTENT_VAR", default="default", env_file=env_file) == "default"

    # Test with required variable
    with pytest.raises(ConfigError, match="TEST_REQUIRED not found in"):
        load_env_var("TEST_REQUIRED", required=True, env_file=env_file)


def test_load_config_from_env(tmp_path) -> None:
    """Test load_config_from_env utility."""
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PREFIX_NAME=test\nPREFIX_VALUE=42\nPREFIX_NESTED__KEY=value\n",
    )

    config = load_config_from_env("PREFIX_", env_file=env_file)
    assert config == {
        "name": "test",
        "value": "42",
        "nested": {"key": "value"},
    }

    # Test with empty prefix
    with pytest.raises(ConfigError, match="Prefix cannot be empty"):
        load_config_from_env("")

    # Test with invalid prefix
    with pytest.raises(ConfigError, match="Prefix must end with underscore"):
        load_config_from_env("PREFIX")
