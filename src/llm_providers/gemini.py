import asyncio
from collections.abc import AsyncGenerator
from typing import Any, TypedDict

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai import GenerationConfig, GenerativeModel

from src.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)
from src.llm_providers.base import BaseLLMProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Error messages
EMPTY_PROMPT_ERROR = "Prompt cannot be empty"
EMPTY_RESPONSE_ERROR = "Empty response from model"
RETRY_ERROR = "Failed to generate response after retries"
TEMPERATURE_ERROR = "Temperature must be between 0 and 1"
MAX_TOKENS_ERROR = "Max tokens must be positive"

# Constants
MAX_RETRIES = DEFAULT_MAX_RETRIES


class SafetySettings(TypedDict):
    """Safety settings for Gemini."""

    category: str
    threshold: str


class GeminiProvider(BaseLLMProvider):
    """Gemini provider implementation."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        """Initialize provider.

        Args:
            api_key: API key.
            model: Model name.
        """
        self.api_key = api_key
        self.model = model
        self.config = {
            "temperature": DEFAULT_TEMPERATURE,
            "max_output_tokens": DEFAULT_MAX_TOKENS,
        }
        genai.configure(api_key=api_key)
        self._model = GenerativeModel(model)

    async def generate(self, prompt: str) -> str:
        """Generate response.

        Args:
            prompt: Input prompt.

        Returns:
            Generated response.

        Raises:
            ValueError: If prompt is empty.
            RuntimeError: If generation fails after retries.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        retries = 0
        while retries <= MAX_RETRIES:
            try:
                response = await self._model.generate_content_async(
                    prompt,
                    generation_config=self.config,
                )
                return response.text
            except ResourceExhausted:
                retries += 1
                if retries > MAX_RETRIES:
                    raise RuntimeError("Failed to generate response after retries")
                await asyncio.sleep(1)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate response stream.

        Args:
            prompt: Input prompt.

        Yields:
            Response chunks.

        Raises:
            ValueError: If prompt is empty.
            RuntimeError: If generation fails after retries.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        retries = 0
        while retries <= MAX_RETRIES:
            try:
                response = await self._model.generate_content_async(
                    prompt,
                    generation_config=self.config,
                    stream=True,
                )
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
                break
            except ResourceExhausted:
                retries += 1
                if retries > MAX_RETRIES:
                    raise RuntimeError("Failed to generate response after retries")
                await asyncio.sleep(1)

    def get_config(self) -> dict[str, Any]:
        """Get provider config.

        Returns:
            Provider config.
        """
        return self.config

    def update_config(self, config: dict[str, Any]) -> None:
        """Update provider config.

        Args:
            config: New config values.
        """
        self.config.update(config)

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate provider config.

        Args:
            config: Config to validate.

        Raises:
            ValueError: If config is invalid.
        """
        if "temperature" in config and not 0 <= config["temperature"] <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        if "max_output_tokens" in config and config["max_output_tokens"] <= 0:
            raise ValueError("Max tokens must be greater than 0")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in.

        Returns:
            Token count.
        """
        return self._model.count_tokens(text).total_tokens

    def get_config(self) -> dict[str, Any]:
        """Get current configuration.

        Returns:
            Current configuration.
        """
        return self.config.copy()

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.
        """
        self.config.update(config)

    def _get_model(self) -> GenerativeModel:
        """Get Gemini model instance.

        Returns:
            Model instance.
        """
        return self.model

    def _get_generation_config(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationConfig:
        """Get generation configuration.

        Args:
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generation configuration.

        Raises:
            ValueError: If parameters are invalid.
        """
        config = {
            "temperature": temperature or self.config["temperature"],
            "max_output_tokens": max_tokens or self.config["max_output_tokens"],
            "top_p": self.config["top_p"],
            "top_k": self.config["top_k"],
        }

        if not 0 <= config["temperature"] <= 1:
            raise ValueError(TEMPERATURE_ERROR)
        if config["max_output_tokens"] < 1:
            raise ValueError(MAX_TOKENS_ERROR)

        return GenerationConfig(**config)

    def _get_safety_settings(self) -> list[SafetySettings]:
        """Get safety settings.

        Returns:
            Safety settings.
        """
        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
        return [{"category": cat, "threshold": "BLOCK_NONE"} for cat in categories]
