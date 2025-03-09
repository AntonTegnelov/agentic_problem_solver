"""Configuration module for default values and constants."""

from enum import Enum

# LLM Provider defaults
DEFAULT_LLM_CONFIG = {
    "model": "gemini-2.0-flash-lite",
    "temperature": 0.7,
    "retry_attempts": 3,
    "timeout_seconds": 30.0,
    "max_tokens": None,  # Added to LLM config since it's a generation parameter
    "stop_sequences": None,  # Added to LLM config since it's a generation parameter
}

# Agent configuration defaults
DEFAULT_AGENT_CONFIG = {
    **DEFAULT_LLM_CONFIG,  # Include all LLM defaults
    "task_timeout": 300.0,  # Agent-specific timeout for task completion
    "max_retries": 3,  # Agent-specific retry count for task attempts
    "max_steps": 10,  # Maximum number of steps in agent workflow
}


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
    EXECUTE = "execute"
    VERIFY = "verify"
    END = "end"


# Default configuration values
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MODEL = "gemini-2.0-flash-lite"
DEFAULT_TASK_TIMEOUT = 300  # 5 minutes
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_STEPS = 10
