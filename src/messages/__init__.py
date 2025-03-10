"""Message handling package."""

from .chain import MessageChain, MessagePriority, create_message_chain
from .creation import (
    create_ai_message,
    create_human_message,
    create_structured_message,
    create_system_message,
    create_tool_message,
    get_message_at_index,
)
from .handler import MessageHandler
from .processor import (
    MessageProcessor,
    create_message_from_dict,
    get_message_metadata,
    get_metadata_at_index,
    parse_structured_content,
    set_message_metadata,
    set_metadata_at_index,
    validate_message_content,
)
from .routing import MessageRouter

__all__ = [
    "MessageChain",
    "MessageHandler",
    "MessagePriority",
    "MessageProcessor",
    "MessageRouter",
    "create_ai_message",
    "create_human_message",
    "create_message_chain",
    "create_message_from_dict",
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
