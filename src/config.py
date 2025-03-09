"""Configuration module for default values and constants."""

from enum import Enum

# Version
VERSION = "0.1.0"

# Default values
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MODEL = "gemini-2.0-flash-lite"
DEFAULT_RETRIES = 3
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_TASK_TIMEOUT = 300  # 5 minutes

# LLM configuration
DEFAULT_LLM_CONFIG = {
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "model": DEFAULT_MODEL,
    "retry_attempts": DEFAULT_RETRIES,
    "top_p": DEFAULT_TOP_P,
    "top_k": DEFAULT_TOP_K,
}

# Error messages
TEMPERATURE_ERROR = "Temperature must be between 0 and 1"
MAX_TOKENS_ERROR = "Max tokens must be positive"
MODEL_ERROR = "Model name cannot be empty"
RETRY_ERROR = "Retry attempts must be positive"
TASK_TIMEOUT_ERROR = "Task timeout must be greater than 0"

# Agent configuration defaults
DEFAULT_AGENT_CONFIG = {
    **DEFAULT_LLM_CONFIG,  # Include all LLM defaults
    "task_timeout": DEFAULT_TASK_TIMEOUT,  # Agent-specific timeout for task completion
    "max_retries": DEFAULT_RETRIES,  # Agent-specific retry count for task attempts
    "max_steps": 10,  # Maximum number of steps in agent workflow
}

# Error messages
API_KEY_ERROR = "GEMINI_API_KEY environment variable not set"
UNSUPPORTED_PROVIDER_ERROR = "Unsupported provider: {}"
TASK_ERROR = "Error processing task: {}"

# Supported providers
SUPPORTED_PROVIDERS = ["gemini"]


class MessageRoles(str, Enum):
    """Message role types."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStep(str, Enum):
    """Agent processing steps."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    END = "end"


# Default configuration values
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_STEPS = 10
