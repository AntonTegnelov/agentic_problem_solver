"""Message types and utilities."""

from __future__ import annotations

from typing import TypeVar

# Import submodules at the top to avoid E402 errors
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    BaseMessage as Message,
)

# Import local modules at the top
from src.common_types.enums import MessagePriority

from .chain import MessageChain
from .creation import (
    create_ai_message,
    create_human_message,
    create_message,
    create_message_chain,
    create_structured_message,
    create_system_message,
    create_tool_message,
)
from .handler import MessageHandler
from .processor import MessageProcessor
from .router import MessageRouter
from .schemas import (
    CompletionReport,
    ExecutionReport,
    ProgressReport,
    ProgressStatus,
)
from .utils import (
    get_message_at_index,
    get_message_metadata,
    get_metadata_at_index,
    parse_structured_content,
    set_message_metadata,
    set_metadata_at_index,
    validate_message_content,
)

T = TypeVar("T")

__all__ = [
    "AIMessage",
    "CompletionReport",
    "ExecutionReport",
    "HumanMessage",
    "Message",
    "MessageChain",
    "MessageHandler",
    "MessagePriority",
    "MessageProcessor",
    "MessageRouter",
    "ProgressReport",
    "ProgressStatus",
    "SystemMessage",
    "ToolMessage",
    "create_ai_message",
    "create_human_message",
    "create_message",
    "create_message_chain",
    "create_structured_message",
    "create_system_message",
    "create_tool_message",
    "get_message_at_index",
    "get_message_metadata",
    "get_metadata_at_index",
    "parse_structured_content",
    "set_message_metadata",
    "set_metadata_at_index",
    "validate_message_content",
]
