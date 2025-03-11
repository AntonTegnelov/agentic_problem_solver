"""Message handling module."""

# This file is a compatibility layer for src/messages/chain.py. Issue: #123
# The project is consolidating these implementations to avoid confusion and bugs.
# The plan is:
# 1. Ensure all tests pass with both implementations
# 2. Gradually migrate all code to use the modular implementation in src/messages/chain.py
# 3. Eventually remove this compatibility layer

from __future__ import annotations

# Re-export from original sources to avoid circular imports
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    BaseMessage as Message,
)

from src.common_types.enums import MessagePriority
from src.messages.chain import MessageChain, create_message_chain
from src.messages.creation import (
    create_ai_message,
    create_human_message,
    create_message,
    create_structured_message,
    create_system_message,
    create_tool_message,
)
from src.messages.handler import MessageHandler
from src.messages.processor import MessageProcessor
from src.messages.router import MessageRouter
from src.messages.utils import (
    get_message_at_index,
    get_message_metadata,
    get_metadata_at_index,
    parse_structured_content,
    set_message_metadata,
    set_metadata_at_index,
    validate_message_content,
)

# For backward compatibility
get_metadata = get_message_metadata
set_metadata = set_message_metadata

__all__ = [
    "AIMessage",
    "HumanMessage",
    "Message",
    "MessageChain",
    "MessageHandler",
    "MessagePriority",
    "MessageProcessor",
    "MessageRouter",
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
    "get_metadata",
    "get_metadata_at_index",
    "parse_structured_content",
    "set_message_metadata",
    "set_metadata",
    "set_metadata_at_index",
    "validate_message_content",
]
