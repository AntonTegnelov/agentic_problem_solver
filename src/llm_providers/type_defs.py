"""LLM provider type definitions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationConfig:
    """Base configuration for LLM generation."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_k: int = 40
    top_p: float = 0.95
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary.

        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            **self.extra_params,
        }

    def update(self, config: dict[str, Any]) -> None:
        """Update configuration.

        Args:
            config: Configuration updates.

        """
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra_params[key] = value
