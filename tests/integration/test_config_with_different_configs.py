"""Integration tests for the configuration system with different configurations.

These tests verify that the configuration system can properly handle different
configuration types, validate configurations, and handle configuration updates.
"""

from pathlib import Path

import pytest

from src.common_types import ConfigError
from src.config.agent import AgentConfig
from src.config.base import BaseConfig
from src.config.llm import LLMConfig
from src.config.utils import load_config_from_env


class TestConfigWithDifferentConfigs:
    """Test configuration system with different configurations."""

    def test_config_inheritance(self) -> None:
        """Test configuration inheritance."""
        # Create a base config
        BaseConfig()

        # Create an agent config that inherits from base config
        agent_config = AgentConfig()

        # Verify that agent_config is a BaseConfig
        assert isinstance(agent_config, BaseConfig)

        # Verify that agent_config has all the expected attributes
        assert hasattr(agent_config, "model")
        assert hasattr(agent_config, "temperature")
        assert hasattr(agent_config, "max_tokens")
        assert hasattr(agent_config, "max_retries")
        assert hasattr(agent_config, "retry_delay")
        assert hasattr(agent_config, "timeout")
        assert hasattr(agent_config, "context")
        assert hasattr(agent_config, "task_timeout")
        assert hasattr(agent_config, "max_steps")
        assert hasattr(agent_config, "name")

        # Verify that to_dict method works for inherited configs
        agent_dict = agent_config.to_dict()
        assert "model" in agent_dict
        assert "temperature" in agent_dict
        assert "max_tokens" in agent_dict

    def test_config_update_with_different_types(self) -> None:
        """Test updating configuration with different types."""
        # Create an agent config
        agent_config = AgentConfig()

        # Update with string model
        agent_config.update({"model": "new-model"})
        assert agent_config.model == "new-model"

        # Update with LLMConfig model
        llm_config = LLMConfig(model="llm-model")
        agent_config.update({"model": llm_config})
        assert isinstance(agent_config.model, LLMConfig)
        assert agent_config.model.model == "llm-model"

        # Update nested LLMConfig
        agent_config.update({"model": {"model": "updated-model"}})
        assert isinstance(agent_config.model, LLMConfig)
        assert agent_config.model.model == "updated-model"

    def test_config_validation_with_different_values(self) -> None:
        """Test configuration validation with different values."""
        # Test with valid values
        agent_config = AgentConfig(
            temperature=0.5,
            max_retries=2,
            task_timeout=60,
            max_steps=5,
        )
        assert agent_config.temperature == 0.5
        assert agent_config.max_retries == 2
        assert agent_config.task_timeout == 60
        assert agent_config.max_steps == 5

        # Test with invalid temperature
        with pytest.raises(ConfigError, match="temperature must be less than 1"):
            AgentConfig(temperature=1.5)

        # Test with invalid max_retries
        with pytest.raises(ConfigError, match="max_retries must be greater than 0"):
            AgentConfig(max_retries=-1)

        # Test with invalid task_timeout
        with pytest.raises(ConfigError, match="task_timeout must be greater than 1"):
            AgentConfig(task_timeout=0)

        # Test with invalid max_steps
        with pytest.raises(ConfigError, match="max_steps must be greater than 1"):
            AgentConfig(max_steps=0)

    def test_config_from_env_with_different_prefixes(self, tmp_path: Path) -> None:
        """Test loading configuration from environment with different prefixes."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            "AGENT_MODEL=test-model\n"
            "AGENT_TEMPERATURE=0.8\n"
            "AGENT_MAX_TOKENS=500\n"
            "LLM_MODEL=llm-model\n"
            "LLM_TEMPERATURE=0.6\n"
            "LLM_MAX_OUTPUT_TOKENS=1000\n",
        )

        # Load agent config
        agent_config_dict = load_config_from_env("AGENT_", env_file=env_file)
        assert agent_config_dict == {
            "model": "test-model",
            "temperature": "0.8",
            "max_tokens": "500",
        }

        # Load LLM config
        llm_config_dict = load_config_from_env("LLM_", env_file=env_file)
        assert llm_config_dict == {
            "model": "llm-model",
            "temperature": "0.6",
            "max_output_tokens": "1000",
        }

        # Test with non-existent prefix
        empty_config = load_config_from_env("NONEXISTENT_", env_file=env_file)
        assert empty_config == {}

    def test_config_serialization_with_different_configs(self) -> None:
        """Test configuration serialization with different configurations."""
        # Create an agent config with nested LLM config
        llm_config = LLMConfig(model="nested-model", temperature=0.5)
        agent_config = AgentConfig(
            model=llm_config,
            temperature=0.7,
            max_tokens=1000,
            name="test-agent",
        )

        # Convert to dictionary
        config_dict = agent_config.to_dict()

        # Verify dictionary structure
        assert isinstance(config_dict["model"], dict)
        assert config_dict["model"]["model"] == "nested-model"
        assert config_dict["model"]["temperature"] == 0.5
        assert config_dict["temperature"] == 0.7
        assert config_dict["max_tokens"] == 1000
        assert config_dict["name"] == "test-agent"

        # Create new config from dictionary
        new_config = AgentConfig.from_dict(config_dict)

        # Verify new config
        assert isinstance(new_config.model, LLMConfig)
        assert new_config.model.model == "nested-model"
        assert new_config.model.temperature == 0.5
        assert new_config.temperature == 0.7
        assert new_config.max_tokens == 1000
        assert new_config.name == "test-agent"
