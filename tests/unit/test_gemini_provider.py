from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common_types.message_types import AIMessage, HumanMessage, SystemMessage, ToolMessage
from src.config import ConfigError
from src.exceptions import APIKeyError, EmptyResponseError, InvalidModelError, RetryError, TemperatureError
from src.llm_providers.config.provider_config import GeminiConfig
from src.llm_providers.providers.gemini import GeminiProvider
from src.llm_providers.version import ProviderVersion


class TestGeminiProvider:
    """Tests for the GeminiProvider class."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        with (
            patch("google.generativeai.configure"),
            patch(
                "google.generativeai.GenerativeModel",
            ),
            patch("src.llm_providers.config.provider_config.GeminiConfig.validate", return_value=True),
        ):
            config = GeminiConfig(api_key="test_key", model="gemini-2.0-flash-lite")
            provider = GeminiProvider(config)
            assert provider.is_initialized
            assert provider._config == config

    def test_get_version(self) -> None:
        """Test get_version method."""
        with (
            patch("google.generativeai.configure"),
            patch(
                "google.generativeai.GenerativeModel",
            ),
            patch("src.llm_providers.config.provider_config.GeminiConfig.validate", return_value=True),
        ):
            config = GeminiConfig(api_key="test_key", model="gemini-2.0-flash-lite")
            provider = GeminiProvider(config)
            assert provider.get_version() == ProviderVersion.GEMINI_V1

    def test_create_config_with_api_key(self) -> None:
        """Test _create_config method with API key."""
        with patch("src.llm_providers.providers.gemini.load_env_var", return_value="gemini-2.0-flash-lite"):
            provider = MagicMock(spec=GeminiProvider)
            config = GeminiProvider._create_config(provider, api_key="test_key")
            assert config.api_key == "test_key"
            assert config.model == "gemini-2.0-flash-lite"

    def test_create_config_without_api_key(self) -> None:
        """Test _create_config method without API key."""
        with patch(
            "src.llm_providers.providers.gemini.load_env_var",
            side_effect=["test_key", "gemini-2.0-flash-lite"],
        ):
            provider = MagicMock(spec=GeminiProvider)
            config = GeminiProvider._create_config(provider)
            assert config.api_key == "test_key"
            assert config.model == "gemini-2.0-flash-lite"

    def test_create_config_error(self) -> None:
        """Test _create_config method with error."""
        with patch(
            "src.llm_providers.providers.gemini.load_env_var",
            side_effect=ConfigError("Config error"),
        ):
            provider = MagicMock(spec=GeminiProvider)
            with pytest.raises(APIKeyError):
                GeminiProvider._create_config(provider)

    def test_initialize_no_config(self) -> None:
        """Test _initialize method with no config."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._config = None
        with pytest.raises(ConfigError, match="Provider not configured"):
            provider._initialize()

    def test_initialize_no_api_key(self) -> None:
        """Test _initialize method with no API key."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._config = GeminiConfig(api_key=None)
        with pytest.raises(ConfigError, match="API key not found"):
            provider._initialize()

    def test_initialize_invalid_model(self) -> None:
        """Test _initialize method with invalid model."""
        with patch("google.generativeai.configure"):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider._config = GeminiConfig(api_key="test_key", model="invalid-model")
            with pytest.raises(InvalidModelError):
                provider._initialize()

    def test_initialize_exception(self) -> None:
        """Test _initialize method with exception."""
        with (
            patch("google.generativeai.configure"),
            patch(
                "google.generativeai.GenerativeModel",
                side_effect=Exception("Test exception"),
            ),
        ):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider._config = GeminiConfig(api_key="test_key", model="gemini-2.0-flash-lite")
            with pytest.raises(ConfigError):
                provider._initialize()

    def test_load_config_with_api_key(self) -> None:
        """Test _load_config method with API key."""
        provider = MagicMock(spec=GeminiProvider)
        config = GeminiProvider._load_config(provider, api_key="test_key")
        assert config.api_key == "test_key"

    def test_load_config_without_api_key(self) -> None:
        """Test _load_config method without API key."""
        with patch(
            "src.llm_providers.providers.gemini.load_env_var",
            return_value="test_key",
        ):
            provider = MagicMock(spec=GeminiProvider)
            config = GeminiProvider._load_config(provider)
            assert config.api_key == "test_key"

    def test_load_config_error(self) -> None:
        """Test _load_config method with error."""
        with patch(
            "src.llm_providers.providers.gemini.load_env_var",
            side_effect=ConfigError("Config error"),
        ):
            provider = MagicMock(spec=GeminiProvider)
            with pytest.raises(APIKeyError):
                GeminiProvider._load_config(provider)

    def test_validate_response_empty(self) -> None:
        """Test _validate_response method with empty response."""
        provider = GeminiProvider.__new__(GeminiProvider)
        response = MagicMock()
        response.text = ""
        with pytest.raises(EmptyResponseError):
            provider._validate_response(response)

    def test_validate_response_valid(self) -> None:
        """Test _validate_response method with valid response."""
        provider = GeminiProvider.__new__(GeminiProvider)
        response = MagicMock()
        response.text = "Test response"
        # Should not raise an exception
        provider._validate_response(response)

    def test_generate_not_initialized(self) -> None:
        """Test generate method when provider is not initialized."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = None
        with pytest.raises(ConfigError, match="Provider not initialized"):
            provider.generate([])

    def test_generate_success(self) -> None:
        """Test generate method success."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = MagicMock()
        response = MagicMock()
        response.text = "Test response"
        provider._model.generate_content.return_value = response

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            SystemMessage(content="System message"),
            ToolMessage(content="Tool message", tool_call_id="test_id"),
        ]

        result = provider.generate(messages)
        assert result == "Test response"
        provider._model.generate_content.assert_called_once()

    def test_generate_exception(self) -> None:
        """Test generate method with exception."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = MagicMock()
        provider._model.generate_content.side_effect = Exception("Test exception")

        with pytest.raises(RetryError):
            provider.generate([HumanMessage(content="Hello")])

    @pytest.mark.asyncio
    async def test_generate_stream_not_initialized(self) -> None:
        """Test generate_stream method when provider is not initialized."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = None
        with pytest.raises(ConfigError, match="Provider not initialized"):
            async for _ in provider.generate_stream([]):
                pass

    @pytest.mark.asyncio
    async def test_generate_stream_success(self) -> None:
        """Test generate_stream method success."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = AsyncMock()

        # Create a mock response that can be iterated asynchronously
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"

        async def mock_stream():
            yield chunk1
            yield chunk2

        provider._model.generate_content_async.return_value = mock_stream()

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            SystemMessage(content="System message"),
            ToolMessage(content="Tool message", tool_call_id="test_id"),
        ]

        chunks = []
        async for chunk in provider.generate_stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]
        provider._model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_exception(self) -> None:
        """Test generate_stream method with exception."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = AsyncMock()
        provider._model.generate_content_async.side_effect = Exception("Test exception")

        with pytest.raises(RetryError):
            async for _ in provider.generate_stream([HumanMessage(content="Hello")]):
                pass

    def test_validate_config_no_model(self) -> None:
        """Test validate_config method with no model."""
        provider = GeminiProvider.__new__(GeminiProvider)
        config = MagicMock()
        config.model = None

        with pytest.raises(InvalidModelError):
            provider.validate_config(config)

    def test_validate_config_invalid_temperature(self) -> None:
        """Test validate_config method with invalid temperature."""
        provider = GeminiProvider.__new__(GeminiProvider)

        # Test temperature < 0
        config = MagicMock()
        config.model = "gemini-2.0-flash-lite"
        config.temperature = -0.1

        with pytest.raises(TemperatureError):
            provider.validate_config(config)

        # Test temperature > 1
        config.temperature = 1.1

        with pytest.raises(TemperatureError):
            provider.validate_config(config)

    def test_validate_config_valid(self) -> None:
        """Test validate_config method with valid config."""
        provider = GeminiProvider.__new__(GeminiProvider)
        config = MagicMock()
        config.model = "gemini-2.0-flash-lite"
        config.temperature = 0.5

        # Should not raise an exception
        provider.validate_config(config)

    def test_count_tokens_not_initialized(self) -> None:
        """Test count_tokens method when provider is not initialized."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = None

        with pytest.raises(ConfigError, match="Provider not initialized"):
            provider.count_tokens("Test text")

    def test_count_tokens_success(self) -> None:
        """Test count_tokens method success."""
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model = MagicMock()
        token_count = MagicMock()
        token_count.total_tokens = 10
        provider._model.count_tokens.return_value = token_count

        result = provider.count_tokens("Test text")
        assert result == 10
        provider._model.count_tokens.assert_called_once_with("Test text")
