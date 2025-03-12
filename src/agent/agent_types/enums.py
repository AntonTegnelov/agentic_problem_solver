"""Agent-related enumerations.

DEPRECATED: This module is deprecated and will be removed in a future version.
All agent-related enumerations have been moved to src.common_types.enums.
Please update your imports to use the new location:

from src.common_types.enums import AgentStatus, MessageRole

Known imports to update:
- tests/unit/test_agent_enums.py
"""

import warnings
from enum import Enum

# Emit a deprecation warning when the module is imported
warnings.warn(
    "The module src.agent.agent_types.enums is deprecated. Use src.common_types.enums instead.",
    DeprecationWarning,
    stacklevel=2,
)


class AgentStatus(str, Enum):
    """Agent status enumeration.

    DEPRECATED: Use src.common_types.enums.AgentStatus instead.

    Note: The values in this enum differ from the common_types version:
    - This version has PROCESSING instead of BUSY
    - This version has DONE instead of COMPLETED
    """

    IDLE = "idle"
    PROCESSING = "processing"  # Different from common_types.enums.AgentStatus.BUSY
    ERROR = "error"
    DONE = "done"  # Different from common_types.enums.AgentStatus.COMPLETED


class MessageRole(str, Enum):
    """Message role types.

    DEPRECATED: Use src.common_types.enums.MessageRole instead.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
