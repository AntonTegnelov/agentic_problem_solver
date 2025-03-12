# Enumerations in APS

## Overview

This document provides guidance on the enumerations used throughout the Agentic Problem Solver (APS) codebase and the proper way to import and use them.

## Consolidated Enumerations

All enumerations in APS are consolidated in the `src.common_types.enums` module to provide a consistent interface and avoid duplication.

### Available Enumerations

The following enumerations are available:

#### AgentStatus

Represents the current state of an agent:

```python
class AgentStatus(str, Enum):
    IDLE = "idle"                # Agent is not currently processing any task
    BUSY = "busy"                # Agent is actively processing a task
    PROCESSING = "processing"    # Alias for BUSY for backward compatibility
    ERROR = "error"              # Agent encountered an error during processing
    COMPLETED = "completed"      # Agent has completed its task
    DONE = "done"                # Alias for COMPLETED for backward compatibility
```

#### AgentStep

Represents the core problem-solving workflow steps:

```python
class AgentStep(str, Enum):
    UNDERSTAND = "understand"    # Analyze and comprehend the task
    PLAN = "plan"                # Create a strategy to solve the task
    EXECUTE = "execute"          # Implement the planned solution
    VERIFY = "verify"            # Test and validate the solution
```

#### MessageRole

Defines the different participants in a conversation:

```python
class MessageRole(str, Enum):
    SYSTEM = "system"            # System-level instructions or context
    USER = "user"                # Input from the user
    ASSISTANT = "assistant"      # Responses from the AI assistant
    TOOL = "tool"                # Output from tools or function calls
```

#### LogLevel

Standard logging levels:

```python
class LogLevel(str, Enum):
    DEBUG = "debug"              # Detailed information for debugging
    INFO = "info"                # General information about program execution
    WARNING = "warning"          # Indicate a potential problem
    ERROR = "error"              # A more serious problem
    CRITICAL = "critical"        # A critical problem that may prevent program execution
```

#### MessagePriority

Message priority levels with comparison methods:

```python
class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
```

## Importing Enumerations

### Recommended Import Pattern

Always import enumerations from `src.common_types.enums`:

```python
from src.common_types.enums import (
    AgentStatus,
    AgentStep,
    LogLevel,
    MessagePriority,
    MessageRole,
)
```

## Backward Compatibility

For backward compatibility, the `AgentStatus` enum includes both the current values (`BUSY`, `COMPLETED`) and the legacy values (`PROCESSING`, `DONE`) as aliases. This ensures that code using either naming convention will continue to work.

## Rationale

Centralizing enumerations in `src.common_types.enums` provides several benefits:

1. **Consistency**: All code uses the same enumeration values
2. **Maintainability**: Easier to update or extend enumerations in one place
3. **Clarity**: Clear documentation of all available options
4. **Type Safety**: Consistent type annotations throughout the codebase

By following these guidelines, we ensure that the codebase remains consistent and maintainable as it evolves.
