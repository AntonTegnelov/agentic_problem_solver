from typing import Any, Dict, List, Optional, AsyncGenerator
import google.generativeai as genai
from .base import LLMProvider

class GeminiProvider(LLMProvider):
    """Google's Gemini AI provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        """Initialize the Gemini provider.
        
        Args:
            api_key: Gemini API key
            model: Model to use (default: gemini-pro)
        """
        self.api_key = api_key
        self.model = model
        self.config = {
            "temperature": 0.7,
            "max_output_tokens": 8192,
            "top_p": 0.95,
            "top_k": 40,
        }
        
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
    
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
        
        response = await self._model.generate_content_async(
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
        
        response = await self._model.generate_content_async(
            prompt,
            generation_config=generation_config,
            safety_settings=self._get_safety_settings(),
            stream=True
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def get_token_count(self, text: str) -> int:
        """Get token count using Gemini's tokenizer."""
        return len(self._model.count_tokens(text).tokens)
    
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