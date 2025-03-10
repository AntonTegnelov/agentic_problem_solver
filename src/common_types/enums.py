"""Common enumerations used throughout the application."""

from enum import Enum


class AgentStep(str, Enum):
    """Agent execution steps.

    These steps represent the core problem-solving workflow:
    - UNDERSTAND: Analyze and comprehend the task
    - PLAN: Create a strategy to solve the task
    - EXECUTE: Implement the planned solution
    - VERIFY: Test and validate the solution
    """

    UNDERSTAND = "understand"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"


class MessageRole(str, Enum):
    """Message roles in conversations.

    These roles define the different participants in a conversation:
    - SYSTEM: System-level instructions or context
    - USER: Input from the user
    - ASSISTANT: Responses from the AI assistant
    - TOOL: Output from tools or function calls
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LogLevel(str, Enum):
    """Log level enumeration.

    Standard logging levels:
    - DEBUG: Detailed information for debugging
    - INFO: General information about program execution
    - WARNING: Indicate a potential problem
    - ERROR: A more serious problem
    - CRITICAL: A critical problem that may prevent program execution
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
