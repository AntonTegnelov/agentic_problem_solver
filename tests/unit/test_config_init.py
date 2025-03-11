"""Tests for config package initialization."""

from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TASK_TIMEOUT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    VERSION,
    AgentConfig,
    BaseConfig,
    ConfigError,
)


def test_config_imports() -> None:
    """Test that all config imports are available."""
    # Test that all imports are available
    assert DEFAULT_MAX_OUTPUT_TOKENS is not None
    assert DEFAULT_MAX_RETRIES is not None
    assert DEFAULT_MAX_STEPS is not None
    assert DEFAULT_MODEL is not None
    assert DEFAULT_TASK_TIMEOUT is not None
    assert DEFAULT_TEMPERATURE is not None
    assert DEFAULT_TOP_K is not None
    assert DEFAULT_TOP_P is not None
    assert VERSION is not None
    assert AgentConfig is not None
    assert BaseConfig is not None
    assert ConfigError is not None

    # Test that constants have expected types
    assert isinstance(DEFAULT_MAX_OUTPUT_TOKENS, int)
    assert isinstance(DEFAULT_MAX_RETRIES, int)
    assert isinstance(DEFAULT_MAX_STEPS, int)
    assert isinstance(DEFAULT_MODEL, str)
    assert isinstance(DEFAULT_TASK_TIMEOUT, int)
    assert isinstance(DEFAULT_TEMPERATURE, float)
    assert isinstance(DEFAULT_TOP_K, int)
    assert isinstance(DEFAULT_TOP_P, float)
    assert isinstance(VERSION, str)

    # Test that classes have expected attributes
    assert hasattr(AgentConfig, "from_dict")
    assert hasattr(BaseConfig, "to_dict")

    # Test that error class is an exception
    assert issubclass(ConfigError, Exception)
