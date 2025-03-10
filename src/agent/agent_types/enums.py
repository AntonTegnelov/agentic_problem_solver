"""Agent-related enumerations."""

from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration."""

    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DONE = "done"


class MessageRole(str, Enum):
    """Message role types."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepType(str, Enum):
    """Step types."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
