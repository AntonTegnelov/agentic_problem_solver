"""Gemini LLM provider implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

import google.generativeai as genai
from google.generativeai.types import AsyncGenerateContentResponse, GenerationConfig

from src.common_types.message_types import AIMessage, HumanMessage, Message, SystemMessage, ToolMessage
from src.config import ConfigError
from src.config.utils import load_env_var
from src.exceptions import (
    APIKeyError,
    EmptyResponseError,
    InvalidModelError,
    RetryError,
    TemperatureError,
)
from src.llm_providers.config.provider_config import GeminiConfig
from src.llm_providers.providers.base import BaseLLMProvider
from src.llm_providers.version import ProviderVersion
from src.utils.log_utils import get_logger

if TYPE_CHECKING:
    from src.agent.agent_types.agent_types import Message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

if TYPE_CHECKING:
    from google.generativeai.types import AsyncGenerateContentResponse

    from src.llm_providers.type_defs import GenerationConfig

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Gemini provider implementation.

    A provider that uses the Google Generative AI API with the Gemini model.
    The default model is 'gemini-2.0-flash-lite', optimized for our use case.

    Attributes:
        _model: The underlying Gemini model instance.
        _config: Provider configuration.
        _default_model: Default model name.
        is_initialized: Indicates whether the provider is initialized.

    """

    _model: Any | None = None
    _config: GeminiConfig | None = None
    _default_model: ClassVar[str] = "gemini-2.0-flash-lite"
    is_initialized: bool = False

    def __init__(self, config: GeminiConfig) -> None:
        """Initialize provider.

        Args:
            config: Provider configuration.

        """
        # Store the config directly
        self.config = config
        self._config = config
        # Validate the config
        self._validate_config()
        # Initialize the provider
        self._initialize()

    def get_version(self) -> ProviderVersion:
        """Get provider version.

        Returns:
            Provider version information.

        """
        return ProviderVersion.GEMINI_V1

    def _create_config(self, api_key: str | None = None) -> GeminiConfig:
        """Create provider configuration.

        Args:
            api_key: Optional API key.

        Returns:
            Provider configuration.

        Raises:
            APIKeyError: If API key is not found.

        """
        try:
            if api_key:
                # Load model from environment
                model = load_env_var("GEMINI_MODEL")
                return GeminiConfig(api_key=api_key, model=model)

            # Load both API key and model from environment
            api_key = load_env_var("GEMINI_API_KEY")
            model = load_env_var("GEMINI_MODEL")
            return GeminiConfig(api_key=api_key, model=model)
        except ConfigError as e:
            raise APIKeyError(str(e)) from e

    def _initialize(self) -> None:
        """Initialize provider."""
        if not self._config:
            msg = "Provider not configured"
            raise ConfigError(msg)

        if not self._config.api_key:
            msg = "API key not found"
            raise ConfigError(msg)

        genai.configure(api_key=self._config.api_key)
        model_name = self._config.model or self._default_model

        # Validate model name
        valid_models = ["gemini-pro", "gemini-pro-vision", "gemini-2.0-flash-lite"]
        if model_name not in valid_models:
            msg = f"Invalid model name: {model_name}. Valid models are: {', '.join(valid_models)}"
            raise InvalidModelError(msg)

        try:
            self._model = genai.GenerativeModel(model_name)
            self.is_initialized = True
        except Exception as e:
            msg = f"Failed to initialize model: {e}"
            raise ConfigError(msg) from e

    def _load_config(self, api_key: str | None = None) -> GeminiConfig:
        """Load configuration.

        Args:
            api_key: API key.

        Returns:
            Provider configuration.

        Raises:
            ConfigError: If .env file is not found.
            APIKeyError: If API key is not found in .env file.

        """
        if api_key:
            return GeminiConfig(api_key=api_key)

        try:
            api_key = load_env_var("GEMINI_API_KEY")
            return GeminiConfig(api_key=api_key)
        except ConfigError as e:
            raise APIKeyError(str(e)) from e

    def _validate_response(self, response: AsyncGenerateContentResponse) -> None:
        """Validate response.

        Args:
            response: Response to validate.

        Raises:
            EmptyResponseError: If response is empty.

        """
        if not response.text:
            msg = "Empty response from model"
            raise EmptyResponseError(msg)

    def generate(
        self,
        messages: list[Message],
    ) -> str:
        """Generate response from messages.

        Args:
            messages: Messages to generate response from.

        Returns:
            Generated response.

        Raises:
            ConfigError: If provider is not initialized.
            RetryError: If generation fails.

        """
        if not self._model:
            msg = "Provider not initialized"
            raise ConfigError(msg)

        try:
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "model"
                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, ToolMessage):
                    role = "function"
                else:
                    role = "user"  # Default fallback

                formatted_messages.append({"role": role, "parts": [msg.content]})

            response = self._model.generate_content(formatted_messages)
        except Exception as e:
            msg = f"Failed to generate response: {e}"
            raise RetryError(msg) from e

        self._validate_response(response)
        return response.text

    async def generate_stream(
        self,
        messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages.

        Args:
            messages: Messages to generate response from.

        Yields:
            Generated response chunks.

        Raises:
            ConfigError: If provider is not initialized.
            RetryError: If generation fails.

        """
        if not self._model:
            msg = "Provider not initialized"
            raise ConfigError(msg)

        try:
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "model"
                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, ToolMessage):
                    role = "function"
                else:
                    role = "user"  # Default fallback

                formatted_messages.append({"role": role, "parts": [msg.content]})

            response = await self._model.generate_content_async(
                formatted_messages,
                stream=True,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            msg = f"Failed to generate response: {e}"
            raise RetryError(msg) from e

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration.

        Args:
            config: Configuration to validate.

        Raises:
            InvalidModelError: If model is invalid.
            TemperatureError: If temperature is invalid.

        """
        if not config.model:
            msg = "Model name is required"
            raise InvalidModelError(msg)

        if config.temperature < 0 or config.temperature > 1:
            msg = "Temperature must be between 0 and 1"
            raise TemperatureError(msg)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Number of tokens.

        Raises:
            ConfigError: If provider is not configured.

        """
        if not self._model:
            msg = "Provider not initialized"
            raise ConfigError(msg)

        return self._model.count_tokens(text).total_tokens
