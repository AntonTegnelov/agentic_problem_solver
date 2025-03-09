from typing import Any, Dict, Optional, AsyncGenerator
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core.exceptions import ResourceExhausted
from .base import LLMProvider
import logging
import asyncio
from src.config import DEFAULT_LLM_CONFIG

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini LLM provider implementation."""

    def __init__(self, api_key: str, model: str = DEFAULT_LLM_CONFIG["model"]):
        """Initialize Gemini provider.

        Args:
            api_key: API key for Gemini
            model: Model name to use. Defaults to value from config.
        """
        super().__init__()
        self.api_key = api_key
        self.model = model

        # Configure Gemini
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

        # Default configuration from DEFAULT_LLM_CONFIG
        self.config = {
            "temperature": DEFAULT_LLM_CONFIG["temperature"],
            "max_output_tokens": DEFAULT_LLM_CONFIG["max_tokens"],
            "stop_sequences": DEFAULT_LLM_CONFIG["stop_sequences"],
            "safety_settings": self._get_safety_settings(),
        }

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional arguments

        Returns:
            Generated text

        Raises:
            ResourceExhausted: If rate limit is exceeded
        """
        generation_config = self._get_generation_config(
            max_tokens, temperature, **kwargs
        )

        # Retry with exponential backoff
        for attempt in range(DEFAULT_LLM_CONFIG["retry_attempts"]):
            try:
                response = await self._model.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.config["safety_settings"],
                )
                return response.text
            except ResourceExhausted:
                if attempt == DEFAULT_LLM_CONFIG["retry_attempts"] - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate text from prompt with streaming.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional arguments

        Yields:
            Generated text chunks

        Raises:
            ResourceExhausted: If rate limit is exceeded
        """
        generation_config = self._get_generation_config(
            max_tokens, temperature, **kwargs
        )

        # Retry with exponential backoff
        for attempt in range(DEFAULT_LLM_CONFIG["retry_attempts"]):
            try:
                response = await self._model.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.config["safety_settings"],
                    stream=True,
                )
                async for chunk in response:
                    if chunk.text.strip():  # Skip empty chunks
                        yield chunk.text
                return
            except ResourceExhausted:
                if attempt == DEFAULT_LLM_CONFIG["retry_attempts"] - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

    def get_token_count(self, text: str) -> int:
        """Get token count for text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        # TODO: Implement proper token counting for Gemini
        return len(text.split())

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(config)

    def _get_generation_config(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> GenerationConfig:
        """Get generation config.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional arguments

        Returns:
            Generation config
        """
        config = {}

        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        elif self.config["max_output_tokens"] is not None:
            config["max_output_tokens"] = self.config["max_output_tokens"]

        if temperature is not None:
            config["temperature"] = temperature
        else:
            config["temperature"] = self.config["temperature"]

        if self.config["stop_sequences"]:
            config["stop_sequences"] = self.config["stop_sequences"]

        return GenerationConfig(**config)

    def _get_safety_settings(self) -> list:
        """Get safety settings.

        Returns:
            List of safety settings
        """
        # Set all safety settings to lowest level for development
        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
        return [{"category": cat, "threshold": "BLOCK_NONE"} for cat in categories]
