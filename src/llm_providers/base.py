from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Generate text based on the prompt.
        
        Args:
            prompt: The input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation (0.0 to 1.0)
            stop_sequences: List of sequences that will stop generation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream generated text based on the prompt.
        
        Args:
            prompt: The input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation (0.0 to 1.0)
            stop_sequences: List of sequences that will stop generation
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Generated text chunks as they become available
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of the provider.
        
        Returns:
            Dictionary containing provider configuration
        """
        pass

    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the provider configuration.
        
        Args:
            config: Dictionary containing new configuration values
        """
        pass 