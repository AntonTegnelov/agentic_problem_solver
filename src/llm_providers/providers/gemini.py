"""Gemini LLM provider implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

import google.generativeai as genai
from google.generativeai.types import AsyncGenerateContentResponse, GenerationConfig

from src.common_types import (
    APIKeyError,
    ConfigError,
    EmptyResponseError,
    InvalidModelError,
    RetryError,
    TemperatureError,
)
from src.common_types.message_types import AIMessage, HumanMessage, Message, SystemMessage, ToolMessage
from src.config.utils import load_env_var
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
        config: Provider configuration.
        _default_model: Default model name.
        is_initialized: Indicates whether the provider is initialized.

    """

    _model: Any | None = None
    _default_model: ClassVar[str] = "gemini-2.0-flash-lite"
    is_initialized: bool = False
    config: GeminiConfig | None = None

    def __init__(self, config: GeminiConfig) -> None:
        """Initialize provider.

        Args:
            config: Provider configuration.

        """
        # Store the config
        self.config = config
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

    @classmethod
    def create_config(cls, api_key: str) -> GeminiConfig:
        """Create provider configuration.

        Args:
            api_key: API key.

        Returns:
            Provider configuration.

        Raises:
            APIKeyError: If API key is not provided.

        """
        if not api_key:
            msg = "API key is required"
            raise APIKeyError(msg)

        return GeminiConfig(
            api_key=api_key,
            model=cls._default_model,
            temperature=0.7,
            max_tokens=None,
        )

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
        if not self.config:
            msg = "Provider not configured"
            raise ConfigError(msg)

        if not self.config.api_key:
            msg = "API key not found"
            raise ConfigError(msg)

        genai.configure(api_key=self.config.api_key)
        model_name = self.config.model or self._default_model

        # Validate model name against the models in the provider version
        valid_models = list(ProviderVersion.GEMINI_V1.supported_models.keys())
        if model_name not in valid_models:
            msg = f"Invalid model name: {model_name}. Valid models are: {', '.join(valid_models)}"
            raise InvalidModelError(msg)

        try:
            self._model = genai.GenerativeModel(model_name)
            self.is_initialized = True
        except Exception as e:
            msg = f"Failed to initialize model: {e}"
            raise ConfigError(msg) from e

    def _validate_config(self) -> None:
        """Validate provider configuration.

        Raises:
            ConfigError: If configuration is invalid.

        """
        if not self.config:
            msg = "Provider not configured"
            raise ConfigError(msg)

        if not self.config.api_key:
            msg = "API key not found"
            raise ConfigError(msg)

        if self.config.temperature is not None and (self.config.temperature < 0.0 or self.config.temperature > 1.0):
            msg = f"Invalid temperature: {self.config.temperature}. Must be between 0.0 and 1.0"
            raise TemperatureError(msg)

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

    async def generate(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages.

        Args:
            messages: Messages to generate response from.
            config: Optional generation configuration.

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

            # Apply config if provided
            if config is not None and self.config is not None:
                # Update generation parameters if provided in config
                generation_params = {}
                if hasattr(config, "temperature") and config.temperature is not None:
                    generation_params["temperature"] = config.temperature
                if hasattr(config, "max_tokens") and config.max_tokens is not None:
                    generation_params["max_output_tokens"] = config.max_tokens

                # Apply the parameters if any were set
                if generation_params and hasattr(self._model, "generation_config"):
                    self._model.generation_config.update(generation_params)

            response = self._model.generate_content(formatted_messages)
            self._validate_response(response)
        except Exception as e:
            msg = f"Failed to generate response: {e}"
            raise RetryError(msg) from e

        return response.text

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages.

        Args:
            messages: Messages to generate response from.
            config: Optional generation configuration.

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

            # Apply config if provided
            if config is not None and self.config is not None:
                # Update generation parameters if provided in config
                generation_params = {}
                if hasattr(config, "temperature") and config.temperature is not None:
                    generation_params["temperature"] = config.temperature
                if hasattr(config, "max_tokens") and config.max_tokens is not None:
                    generation_params["max_output_tokens"] = config.max_tokens

                # Apply the parameters if any were set
                if generation_params and hasattr(self._model, "generation_config"):
                    self._model.generation_config.update(generation_params)

            # Use the async version of generate_content for streaming
            response = await self._model.generate_content_async(formatted_messages, stream=True)

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            msg = f"Failed to generate streaming response: {e}"
            raise RetryError(msg) from e

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration.

        Args:
            config: Configuration to validate.

        Raises:
            InvalidModelError: If model is not specified.
            TemperatureError: If temperature is invalid.

        """
        if not config.model:
            msg = "Model name is required"
            raise InvalidModelError(msg)

        if (
            hasattr(config, "temperature")
            and config.temperature is not None
            and (config.temperature < 0.0 or config.temperature > 1.0)
        ):
            msg = f"Invalid temperature: {config.temperature}. Must be between 0.0 and 1.0"
            raise TemperatureError(msg)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Number of tokens.

        Raises:
            ConfigError: If provider is not initialized.

        """
        if not self._model:
            msg = "Provider not initialized"
            raise ConfigError(msg)

        try:
            return self._model.count_tokens(text).total_tokens
        except Exception as e:
            msg = f"Failed to count tokens: {e}"
            logger.exception(msg)
            return len(text.split())

    def get_config(self) -> GenerationConfig:
        """Get current configuration.

        Returns:
            Current configuration.

        """
        from src.llm_providers.type_defs import GenerationConfig

        return GenerationConfig(
            model=self.config.model or self._default_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            extra_params=self.config.extra_params,
        )

    def update_config(self, config: GenerationConfig) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.

        """
        if hasattr(self.config, "update"):
            self.config.update(config.to_dict())
        else:
            # Manual update
            if config.model:
                self.config.model = config.model
            if hasattr(config, "temperature"):
                self.config.temperature = config.temperature
            if hasattr(config, "max_tokens"):
                self.config.max_tokens = config.max_tokens
            if hasattr(config, "top_k"):
                self.config.top_k = config.top_k
            if hasattr(config, "top_p"):
                self.config.top_p = config.top_p

            # Extra params
            for key, value in config.extra_params.items():
                self.config.extra_params[key] = value
