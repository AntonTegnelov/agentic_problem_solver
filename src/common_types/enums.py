"""Common enumerations used throughout the application."""

from enum import Enum
from typing import Self

# Re-export task-related enums for backward compatibility


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
    - BUSY/PROCESSING: Agent is actively processing a task
    - ERROR: Agent encountered an error during processing
    - COMPLETED/DONE: Agent has completed its task
    """

    IDLE = "idle"
    BUSY = "busy"
    PROCESSING = "processing"  # Alias for BUSY for backward compatibility
    ERROR = "error"
    COMPLETED = "completed"
    DONE = "done"  # Alias for COMPLETED for backward compatibility


class AgentRole(str, Enum):
    """Agent roles in the hierarchical system.

    These roles represent the specialized functions in the hierarchical agent system:
    - ARCHITECT: High-level problem decomposition and system design
    - PLANNER: Mid-level task refinement and planning
    - EXECUTOR: Low-level task execution and implementation
    - SOLVER: Legacy/general role for backward compatibility
    """

    ARCHITECT = "architect"
    PLANNER = "planner"
    EXECUTOR = "executor"
    SOLVER = "solver"  # For backward compatibility


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


class TaskComplexity(str, Enum):
    """Task complexity levels.

    These levels indicate the complexity of a task:
    - SIMPLE: Simple, straightforward task requiring minimal effort
    - MODERATE: Moderately complex task requiring some analysis
    - COMPLEX: Complex task requiring significant analysis and planning
    - VERY_COMPLEX: Extremely complex task requiring extensive analysis and planning
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class TaskPriority(str, Enum):
    """Task priority levels.

    These levels indicate the priority of a task:
    - LOW: Task can be completed when convenient
    - MEDIUM: Standard priority task
    - HIGH: Important task that should be prioritized
    - CRITICAL: Urgent task requiring immediate attention
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionStage(str, Enum):
    """Execution stages for task implementation.

    These stages represent the progression of a task through execution:
    - PLANNING: Initial planning and preparation for implementation
    - IMPLEMENTING: Actively implementing the solution
    - TESTING: Testing the implemented solution
    - REFINING: Making improvements based on test results
    - FINALIZING: Completing final adjustments and documentation
    """

    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    REFINING = "refining"
    FINALIZING = "finalizing"


class VerificationStatus(str, Enum):
    """Verification status for task validation.

    These statuses represent the result of verification checks:
    - PENDING: Verification has not yet been performed
    - PASSED: All verification checks have passed
    - FAILED: One or more verification checks have failed
    - PARTIAL: Some verification checks passed, others failed
    - SKIPPED: Verification was skipped for this task
    """

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
