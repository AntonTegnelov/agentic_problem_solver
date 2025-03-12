"""Tests for the provider configuration module."""

from __future__ import annotations

import pytest

from src.config import ConfigError
from src.llm_providers.config.provider_config import GeminiConfig, ProviderConfig
from src.llm_providers.version import ProviderVersion, Version


class TestProviderConfig:
    """Tests for the ProviderConfig class."""

    def test_provider_config_initialization(self) -> None:
        """Test provider config initialization."""
        config = ProviderConfig(
            provider_name="test-provider",
            model="test-model",
            temperature=0.5,
            max_tokens=100,
            api_key="test-key",
            api_base="test-base",
            api_version="1.0.0",
            api_type="test-type",
            deployment_name="test-deployment",
            organization_id="test-org",
            additional_kwargs={"test": "value"},
        )

        assert config.provider_name == "test-provider"
        assert config.model == "test-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 100
        assert config.api_key == "test-key"
        assert config.api_base == "test-base"
        assert config.api_version == "1.0.0"
        assert config.api_type == "test-type"
        assert config.deployment_name == "test-deployment"
        assert config.organization_id == "test-org"
        assert config.additional_kwargs == {"test": "value"}

    def test_provider_config_validate_no_api_key(self) -> None:
        """Test provider config validation with no API key."""
        config = ProviderConfig(
            provider_name="test-provider",
            model="test-model",
            api_key=None,
        )

        with pytest.raises(ConfigError, match="API key is required"):
            config.validate()

    def test_provider_config_validate_no_model(self) -> None:
        """Test provider config validation with no model."""
        config = ProviderConfig(
            provider_name="test-provider",
            model="",
            api_key="test-key",
        )

        with pytest.raises(ConfigError, match="Model name is required"):
            config.validate()

    def test_provider_config_validate_with_api_version(self) -> None:
        """Test provider config validation with API version."""
        config = ProviderConfig(
            provider_name="test-provider",
            model="test-model",
            api_key="test-key",
            api_version="1.0.0",
        )

        # Should not raise an exception
        assert config.validate() is True

    def test_provider_config_get_model_version_no_provider_version(self) -> None:
        """Test get_model_version with no provider version."""
        config = ProviderConfig(
            provider_name="test-provider",
            model="test-model",
            api_key="test-key",
        )

        with pytest.raises(ConfigError, match="Provider version not set"):
            config.get_model_version()

    def test_provider_config_required_keys(self) -> None:
        """Test required_keys method."""
        config = ProviderConfig(
            provider_name="test-provider",
            model="test-model",
            api_key="test-key",
        )

        keys = config.required_keys()
        assert keys == ["TEST-PROVIDER_API_KEY", "TEST-PROVIDER_MODEL"]


class TestGeminiConfig:
    """Tests for the GeminiConfig class."""

    def test_gemini_config_initialization(self) -> None:
        """Test Gemini config initialization."""
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            temperature=0.5,
            max_output_tokens=1000,
            top_p=0.9,
            top_k=30,
        )

        assert config.api_key == "test-key"
        assert config.model == "gemini-2.0-flash-lite"
        assert config.temperature == 0.5
        assert config.max_output_tokens == 1000
        assert config.top_p == 0.9
        assert config.top_k == 30
        assert config.PROVIDER_VERSION == ProviderVersion.GEMINI_V1

    def test_gemini_config_validate_model_version_error(self) -> None:
        """Test Gemini config validation with model version error."""
        # Create a mock model version that requires a higher provider version
        original_get_model = ProviderVersion.GEMINI_V1.get_model

        def mock_get_model(model_name=None):
            model = original_get_model(model_name)
            # Set a higher minimum provider version
            model.min_provider_version = Version(2, 0, 0)
            return model

        # Patch the get_model method
        ProviderVersion.GEMINI_V1.get_model = mock_get_model

        try:
            config = GeminiConfig(
                api_key="test-key",
                model="gemini-2.0-flash-lite",
            )

            with pytest.raises(ConfigError):
                config.validate()
        finally:
            # Restore the original method
            ProviderVersion.GEMINI_V1.get_model = original_get_model

    def test_gemini_config_validate_temperature_error(self) -> None:
        """Test Gemini config validation with temperature error."""
        # Test with temperature < 0
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            temperature=-0.1,
        )

        with pytest.raises(ConfigError):
            config.validate()

        # Test with temperature > 1
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            temperature=1.1,
        )

        with pytest.raises(ConfigError):
            config.validate()

    def test_gemini_config_validate_max_tokens_error(self) -> None:
        """Test Gemini config validation with max tokens error."""
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            max_output_tokens=0,
        )

        with pytest.raises(ConfigError):
            config.validate()

    def test_gemini_config_validate_top_p_error(self) -> None:
        """Test Gemini config validation with top_p error."""
        # Test with top_p < 0
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            top_p=-0.1,
        )

        with pytest.raises(ConfigError):
            config.validate()

        # Test with top_p > 1
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            top_p=1.1,
        )

        with pytest.raises(ConfigError):
            config.validate()

    def test_gemini_config_validate_top_k_error(self) -> None:
        """Test Gemini config validation with top_k error."""
        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            top_k=0,
        )

        with pytest.raises(ConfigError):
            config.validate()

    def test_gemini_config_validate_success(self) -> None:
        """Test Gemini config validation success."""
        # Create a mock model version that doesn't require a higher provider version
        original_get_model = ProviderVersion.GEMINI_V1.get_model

        def mock_get_model(model_name=None):
            model = original_get_model(model_name)
            # Ensure the minimum provider version is compatible
            model.min_provider_version = Version(1, 0, 0)
            return model

        # Patch the get_model method
        ProviderVersion.GEMINI_V1.get_model = mock_get_model

        try:
            config = GeminiConfig(
                api_key="test-key",
                model="gemini-2.0-flash-lite",
                temperature=0.5,
                max_output_tokens=1000,
                top_p=0.9,
                top_k=30,
            )

            # Should not raise an exception
            assert config.validate() is True
        finally:
            # Restore the original method
            ProviderVersion.GEMINI_V1.get_model = original_get_model

    def test_gemini_config_from_env(self) -> None:
        """Test creating Gemini config from environment variables."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-2.0-flash-lite",
            "GEMINI_TEMPERATURE": "0.5",
            "GEMINI_MAX_OUTPUT_TOKENS": "1000",
            "GEMINI_TOP_P": "0.9",
            "GEMINI_TOP_K": "30",
        }

        config = GeminiConfig.from_env(env_vars)

        assert config.api_key == "test-key"
        assert config.model == "gemini-2.0-flash-lite"
        assert config.temperature == 0.5
        assert config.max_output_tokens == 1000
        assert config.top_p == 0.9
        assert config.top_k == 30

    def test_gemini_config_from_env_missing_key(self) -> None:
        """Test creating Gemini config from environment variables with missing key."""
        env_vars = {
            "GEMINI_MODEL": "gemini-2.0-flash-lite",
        }

        # Should use default values for missing keys
        config = GeminiConfig.from_env(env_vars)

        assert config.api_key is None
        assert config.model == "gemini-2.0-flash-lite"
        assert config.temperature == 0.7  # Default value
        assert config.max_output_tokens == 2048  # Default value
        assert config.top_p == 0.95  # Default value
        assert config.top_k == 40  # Default value

    def test_gemini_config_from_env_invalid_value(self) -> None:
        """Test creating Gemini config from environment variables with invalid value."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-2.0-flash-lite",
            "GEMINI_TEMPERATURE": "invalid",  # Not a float
        }

        with pytest.raises(ConfigError):
            GeminiConfig.from_env(env_vars)

    def test_gemini_config_required_keys(self) -> None:
        """Test required_keys method for GeminiConfig."""
        config = GeminiConfig(
            provider_name="gemini",
            model="gemini-2.0-flash-lite",
            api_key="test-key",
        )

        keys = config.required_keys()
        assert "GEMINI_API_KEY" in keys
        assert "GEMINI_MODEL" in keys
        assert "GEMINI_TEMPERATURE" in keys
        assert "GEMINI_MAX_OUTPUT_TOKENS" in keys
        assert "GEMINI_TOP_P" in keys
        assert "GEMINI_TOP_K" in keys
