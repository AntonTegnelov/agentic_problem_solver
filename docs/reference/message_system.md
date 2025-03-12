# Message System Reference

## Overview

The message system is a core component of the Agentic Problem Solver (APS) framework, providing a robust infrastructure for communication between agents. It handles message creation, validation, routing, and history tracking, enabling complex multi-agent interactions.

## Key Components

### Message Types

The message system uses the following message types imported from `langchain_core.messages`:

- **SystemMessage**: System-level instructions or context
- **HumanMessage**: Input from the user
- **AIMessage**: Responses from AI agents
- **ToolMessage**: Output from tools or function calls

These types are re-exported through `src.common_types.message_types` for convenience.

### Message Chain

The `MessageChain` class manages a sequence of messages, providing:

- Message validation
- Filtering by type, priority, or custom criteria
- Search functionality
- History tracking with metadata

### Message Router

The `MessageRouter` class handles message routing between agents, with features like:

- Agent registration
- Message routing to specific agents
- Message broadcasting to multiple agents
- Retry mechanisms for failed message processing
- Stream-based message processing

### Message Handler

The `MessageHandler` class provides a high-level interface for message processing, including:

- Message handling with priority
- Message history tracking
- Message chain validation
- Message routing to agents

## Usage Examples

### Creating Messages

```python
from src.messages.creation import (
    create_human_message,
    create_ai_message,
    create_system_message,
    create_tool_message,
)

# Create a system message
system_msg = create_system_message("You are a helpful assistant.")

# Create a human message
human_msg = create_human_message("What's the weather like today?")

# Create an AI message
ai_msg = create_ai_message("The weather is sunny with a high of 75°F.")

# Create a tool message
tool_msg = create_tool_message(
    content="Weather data retrieved successfully.",
    tool_call_id="weather_tool",
)

# Create a message with additional metadata
from datetime import datetime, UTC
human_msg_with_metadata = create_human_message(
    content="What's the weather forecast?",
    metadata={
        "timestamp": datetime.now(UTC).isoformat(),
        "location": "New York",
    },
)
```

### Creating Structured Messages

```python
from src.messages.creation import create_structured_message
from src.messages.utils import parse_structured_content

# Create a structured message with JSON content
structured_msg = create_structured_message(
    role="system",
    content={
        "action": "search",
        "query": "weather in New York",
        "parameters": {
            "date": "2023-03-10",
            "location": "New York, NY"
        }
    },
    metadata={
        "source": "weather_api",
        "timestamp": "2023-03-10T12:00:00Z"
    }
)

# Parse structured content from a message
content = parse_structured_content(structured_msg)
print(content["action"])  # "search"
print(content["query"])   # "weather in New York"
```

### Working with Message Chains

```python
from src.messages.chain import create_message_chain
from src.common_types.enums import MessagePriority

# Create a message chain
chain = create_message_chain()

# Add messages with different priorities
chain.add_message(human_msg, MessagePriority.NORMAL)
chain.add_message(ai_msg, MessagePriority.HIGH)

# Validate the chain
chain.validate_chain()

# Filter messages by criteria
high_priority = chain.filter_messages(criteria={"priority": MessagePriority.HIGH.value})

# Get message history
history = chain.get_message_history(limit=10)

# Search messages
results = chain.search_messages("weather", field=None)

# Filter messages with a custom function
def is_question(message):
    return isinstance(message.content, str) and message.content.endswith("?")

questions = chain.filter_messages(filter_fn=is_question)

# Clear the chain
chain.clear()
```

### Routing Messages Between Agents

```python
from src.messages.router import MessageRouter
from src.agent.agent_types.agent_types import MockAgent

# Create a router
router = MessageRouter(max_retries=3, retry_delay=0.1)

# Create and register agents
math_agent = MockAgent("math_agent", ["math"])
text_agent = MockAgent("text_agent", ["text"])

# Register agents
router.register_agent("math_agent", math_agent)
router.register_agent("text_agent", text_agent)

# Route a message to a specific agent
from src.messages.creation import create_human_message
message = create_human_message("Calculate 2 + 2")
result = await router.route_message(message, "math_agent")

# Broadcast a message to all agents
results = await router.broadcast_message(message)

# Stream results from an agent
async for chunk in router.route_message_stream(message, "text_agent"):
    print(chunk)

# Add a route between agents
router.add_route("math_agent", "text_agent")
```

### Using the Message Handler

```python
from src.messages.handler import MessageHandler
from src.agent.agent_types.agent_types import MockAgent

# Create a handler
handler = MessageHandler()

# Create and register agents
math_agent = MockAgent("math_agent", ["math"])
text_agent = MockAgent("text_agent", ["text"])

# Register agents
handler.register_agent("math_agent", math_agent)
handler.register_agent("text_agent", text_agent)

# Register message handlers
def handle_human_message(message):
    print(f"Handling human message: {message.content}")

handler.register_handler("human", handle_human_message)

# Handle a message
from src.messages.creation import create_human_message
message = create_human_message("What's 2 + 2?")
handler.handle_message(message)

# Route a message to an agent
result = await handler.route_to_agent(message, "math_agent")

# Handle a message with retry
result = await handler.handle_message_with_retry(message, "text_agent", max_retries=3)
```

## API Reference

### Message Creation Functions

- `create_message(role: str, content: str, metadata: dict[str, Any] | None = None) -> Message`: Create a message with the specified role
- `create_system_message(content: str, metadata: dict[str, Any] | None = None, **kwargs: object) -> SystemMessage`: Create a system message
- `create_human_message(content: str, metadata: dict[str, Any] | None = None, **kwargs: object) -> HumanMessage`: Create a human message
- `create_ai_message(content: str, metadata: dict[str, Any] | None = None, **kwargs: object) -> AIMessage`: Create an AI message
- `create_tool_message(content: str, tool_call_id: str, metadata: dict[str, Any] | None = None, **kwargs: object) -> ToolMessage`: Create a tool message
- `create_structured_message(role: str, content: str | dict[str, Any], metadata: dict[str, Any] | None = None) -> Message`: Create a structured message
- `create_message_chain() -> MessageChain`: Create a new message chain

### Message Metadata Functions

- `get_message_metadata(message: Message, key: str, default: object = None) -> object`: Get message metadata
- `set_message_metadata(message: Message, key: str, value: object) -> None`: Set message metadata
- `get_message_at_index(messages: list[Message], index: int) -> Message`: Get message at index
- `get_metadata_at_index(messages: list[Message], index: int, key: str, default: T | None = None) -> T | None`: Get metadata at index
- `set_metadata_at_index(messages: list[Message], index: int, key: str, value: object) -> None`: Set metadata at index
- `parse_structured_content(message: Message, default: dict[str, object] | None = None) -> dict[str, object]`: Parse structured content
- `validate_message_content(message: Message, required_fields: list[str] | None = None) -> bool`: Validate message content

### MessageChain Class

- `add_message(message: Message, priority: MessagePriority = MessagePriority.NORMAL) -> None`: Add a message to the chain
- `validate_chain() -> bool`: Validate the message chain
- `validate_message_chain() -> bool`: Validate the message chain (alias for validate_chain)
- `get_messages_by_type(msg_type: type[Message]) -> Iterator[Message]`: Get messages by type
- `get_messages_by_priority(min_priority: MessagePriority = MessagePriority.LOW) -> Iterator[Message]`: Get messages by priority
- `search_messages(query: str, metadata_key: str | None = None) -> Iterator[Message]`: Search messages by content or metadata
- `filter_messages(criteria: dict[str, Any] | None = None, filter_fn: Callable[[Message], bool] | None = None) -> list[Message]`: Filter messages by criteria or function
- `get_message_history(limit: int | None = None, include_metadata: bool = False) -> list[dict[str, Any]]`: Get message history
- `clear() -> None`: Clear all messages from the chain

### MessageRouter Class

- `__init__(max_retries: int = 3, retry_delay: float = 0.1) -> None`: Initialize the router
- `register_agent(name: str, agent: Agent) -> None`: Register an agent
- `unregister_agent(name: str) -> None`: Unregister an agent
- `add_route(source: str, target: str) -> None`: Add a route between agents
- `remove_route(source: str, target: str) -> None`: Remove a route between agents
- `route_message(message: Message, target_agent: str) -> Awaitable[Result]`: Route a message to an agent
- `route_message_stream(message: Message, target_agent: str) -> AsyncGenerator[str, None]`: Stream results from an agent
- `broadcast_message(message: Message) -> Awaitable[dict[str, Result]]`: Broadcast a message to all agents
- `get_agent_names() -> list[str]`: Get all registered agent names

### MessageHandler Class

- `__init__() -> None`: Initialize the handler
- `register_handler(message_type: str, handler: Callable[[Message], None]) -> None`: Register a message handler
- `register_agent(name: str, agent: Agent) -> None`: Register an agent
- `handle_message(message: Message) -> None`: Handle a message
- `track_message_history(message: Message) -> None`: Track message history
- `validate_message_chain() -> bool`: Validate the message chain
- `filter_messages(**criteria) -> list[Message]`: Filter messages by criteria
- `get_message_history(limit: int | None = None) -> list[dict[str, Any]]`: Get message history
- `route_to_agent(message: Message, agent_name: str) -> Awaitable[Result]`: Route a message to an agent
- `route_to_agent_stream(message: Message, agent_name: str) -> AsyncGenerator[str, None]`: Stream results from an agent
- `handle_message_with_retry(message: Message, agent_name: str, max_retries: int = 3) -> Awaitable[Result]`: Handle a message with retry

## Error Handling

The message system uses the following error types from `exceptions.py`:

- `ConfigError`: For configuration and validation errors
- `RetryError`: For retry-related errors

## Best Practices

1. **Message Validation**: Always validate message chains before processing to ensure proper structure.
2. **Metadata Usage**: Use metadata for tracking message properties like timestamps, priorities, and sources.
3. **Error Handling**: Implement proper error handling with retry mechanisms for robust agent communication.
4. **Structured Content**: Use structured messages for complex data exchange between agents.
5. **Message Priority**: Assign appropriate priorities to messages based on their importance.
6. **Message History**: Use message history tracking for debugging and auditing purposes.
