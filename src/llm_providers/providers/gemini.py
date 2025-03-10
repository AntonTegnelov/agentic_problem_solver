"""Gemini LLM provider implementation."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai.types import AsyncGenerateContentResponse

from src.exceptions import (
    APIKeyError,
    ConfigError,
    EmptyResponseError,
    InvalidModelError,
    RetryError,
    TemperatureError,
)
from src.llm_providers.interface import LLMProvider
from src.llm_providers.type_defs import GenerationConfig
from src.utils.log_utils import get_logger
from src.validation import Message

logger = get_logger(__name__)


@dataclass
class GeminiConfig:
    """Gemini configuration."""

    api_key: str | None = None

    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            msg = "API key is required"
            raise APIKeyError(msg)


class GeminiProvider(LLMProvider):
    """Gemini provider implementation.

    A provider that uses the Google Generative AI API with the Gemini model.
    The default model is 'gemini-2.0-flash-lite', optimized for our use case.

    Attributes:
        _model: The underlying Gemini model instance.
        _config: Provider configuration.
        _default_model: Default model name.

    """

    _model: GenerativeModel | None = None
    _config: GeminiConfig | None = None
    _default_model: ClassVar[str] = "gemini-2.0-flash-lite"

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        api_key: str | None = None,
    ) -> None:
        """Initialize provider.

        Args:
            model: Model name.
            api_key: API key.

        """
        super().__init__()
        self._model_name = model
        self._config = self._load_config(api_key)
        self._initialize()

    def _initialize(self) -> None:
        """Initialize provider."""
        if not self._config:
            msg = "Provider not configured"
            raise ConfigError(msg)

        genai.configure(api_key=self._config.api_key)
        self._model = genai.GenerativeModel(self._model_name)

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

        env_path = Path(".env")
        if not env_path.exists():
            msg = "No .env file found"
            raise ConfigError(msg)

        with env_path.open() as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=")[1].strip()
                    break
            else:
                msg = "GEMINI_API_KEY not found in .env file"
                raise APIKeyError(msg)

        return GeminiConfig(api_key=api_key)

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

    def generate(self, messages: list[Message]) -> str:
        """Generate response from messages.

        Args:
            messages: Messages to generate response from.

        Returns:
            Generated response.

        Raises:
            ConfigError: If provider is not configured.
            RetryError: If generation fails.

        """
        if not self._model:
            msg = "Provider not initialized"
            raise ConfigError(msg)

        try:
            response = self._model.generate_content(
                [{"role": msg.role, "parts": [msg.content]} for msg in messages],
            )
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
            ConfigError: If provider is not configured.
            RetryError: If generation fails.

        """
        if not self._model:
            msg = "Provider not initialized"
            raise ConfigError(msg)

        try:
            response = await self._model.generate_content_async(
                [{"role": msg.role, "parts": [msg.content]} for msg in messages],
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
