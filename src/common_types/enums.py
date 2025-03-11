"""Common enumerations used throughout the application."""

from enum import Enum
from typing import Self


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


class AgentStatus(str, Enum):
    """Agent status.

    These statuses represent the current state of an agent:
    - IDLE: Agent is not currently processing any task
    - BUSY: Agent is actively processing a task
    - ERROR: Agent encountered an error during processing
    - COMPLETED: Agent has completed its task
    """

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    COMPLETED = "completed"


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


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: Self) -> bool:
        """Less than comparison.

        Args:
            other: Other priority.

        Returns:
            True if self < other.

        """
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: Self) -> bool:
        """Less than or equal comparison.

        Args:
            other: Other priority.

        Returns:
            True if self <= other.

        """
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: Self) -> bool:
        """Greater than comparison.

        Args:
            other: Other priority.

        Returns:
            True if self > other.

        """
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: Self) -> bool:
        """Greater than or equal comparison.

        Args:
            other: Other priority.

        Returns:
            True if self >= other.

        """
        if not isinstance(other, MessagePriority):
            return NotImplemented
        return self.value >= other.value
