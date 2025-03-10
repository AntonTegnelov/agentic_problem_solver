"""Configuration package initialization.

The default model used throughout the application is 'gemini-2.0-flash-lite'.
This model is chosen for its optimal balance of speed and quality for our use case.
Any changes to this default should be discussed and documented.
"""

from .agent import AgentConfig
from .base import BaseConfig
from .constants import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TASK_TIMEOUT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    VERSION,
)
from .llm import LLMConfig

__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MODEL",
    "DEFAULT_TASK_TIMEOUT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_P",
    "VERSION",
    "AgentConfig",
    "BaseConfig",
    "LLMConfig",
]
