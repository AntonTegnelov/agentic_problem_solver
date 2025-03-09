from typing import Any, Dict, List, Optional, AsyncGenerator
import google.generativeai as genai
from .base import LLMProvider
import logging
import asyncio
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """Google's Gemini AI provider implementation."""
    
    DEFAULT_MODEL = "gemini-2.0-flash-lite"  # Default to flash-lite for higher rate limits
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize Gemini provider.
        
        Args:
            api_key: Gemini API key
            model: Model name to use. Defaults to gemini-2.0-flash-lite for development.
                  Other options include: gemini-1.5-pro, gemini-1.0-pro, etc.
        """
        logger.info("Initializing Gemini provider")
        logger.debug(f"API key length: {len(api_key)}")
        logger.debug(f"API key prefix: {api_key[:10]}...")
        
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        logger.info(f"Using Gemini model: {self.model}")
        
        self.config = {
            "temperature": 0.7,
            "max_output_tokens": None,
            "top_p": 0.95,
            "top_k": 40,
        }
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model)
        logger.info("Gemini provider initialized successfully")
    
    async def _retry_with_backoff(self, func, *args, max_retries: int = 3, **kwargs):
        """Retry a function with exponential backoff.
        
        Args:
            func: The function to retry
            *args: Positional arguments for the function
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            The last exception encountered
        """
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + 1  # Exponential backoff
                logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Generate text using Gemini."""
        generation_config = self._get_generation_config(max_tokens, temperature, **kwargs)
        
        response = await self._retry_with_backoff(
            self._model.generate_content_async,
            prompt,
            generation_config=generation_config,
            safety_settings=self._get_safety_settings()
        )
        
        return response.text
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream generated text using Gemini."""
        generation_config = self._get_generation_config(max_tokens, temperature, **kwargs)
        
        response = await self._retry_with_backoff(
            self._model.generate_content_async,
            prompt,
            generation_config=generation_config,
            safety_settings=self._get_safety_settings(),
            stream=True
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        return self._model.count_tokens(text).total_tokens
    
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
        **kwargs: Any
    ) -> genai.types.GenerationConfig:
        """Create generation config from parameters."""
        config = self.config.copy()
        
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        
        config.update(kwargs)
        
        # Remove None values as they're not accepted by GenerationConfig
        config = {k: v for k, v in config.items() if v is not None}
        
        return genai.types.GenerationConfig(**config)
    
    def _get_safety_settings(self) -> List[Dict[str, Any]]:
        """Get safety settings for content generation."""
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ] 