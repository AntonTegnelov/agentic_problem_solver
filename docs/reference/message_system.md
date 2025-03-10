# Message System Reference

## Overview

The message system is a core component of the Agentic Problem Solver (APS) framework, providing a robust infrastructure for communication between agents. It handles message creation, validation, routing, and history tracking, enabling complex multi-agent interactions.

## Key Components

### Message Types

- **SystemMessage**: System-level instructions or context
- **HumanMessage**: Input from the user
- **AIMessage**: Responses from AI agents
- **ToolMessage**: Output from tools or function calls

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
from src.messages import create_human_message, create_ai_message, create_system_message, create_tool_message

# Create a system message
system_msg = create_system_message("You are a helpful assistant.")

# Create a human message
human_msg = create_human_message("What's the weather like today?")

# Create an AI message
ai_msg = create_ai_message("The weather is sunny with a high of 75°F.")

# Create a tool message
tool_msg = create_tool_message("Weather data retrieved successfully.", "weather_tool")
```

### Creating Structured Messages

```python
from src.messages import create_structured_message

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
from src.messages import parse_structured_content
content = parse_structured_content(structured_msg)
print(content["action"])  # "search"
print(content["query"])   # "weather in New York"
```

### Working with Message Chains

```python
from src.messages import create_message_chain, MessagePriority

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
history = chain.get_message_history(limit=10, include_metadata=True)
```

### Routing Messages Between Agents

```python
from src.messages import MessageRouter
from src.agent.agent_types.agent_types import Agent

# Create a router
router = MessageRouter()

# Register agents
router.register_agent("math_agent", math_agent)
router.register_agent("text_agent", text_agent)

# Route a message to a specific agent
result = router.route_message(message, "math_agent")

# Broadcast a message to all agents
results = router.broadcast_message(message)

# Stream results from an agent
async for chunk in router.route_message_stream(message, "text_agent"):
    print(chunk)
```

### Using the Message Handler

```python
from src.messages import MessageHandler

# Create a handler
handler = MessageHandler()

# Register agents
handler.register_agent("math_agent", math_agent)
handler.register_agent("text_agent", text_agent)

# Handle a message
handler.handle_message(message, MessagePriority.HIGH)

# Route a message to an agent
result = handler.route_to_agent(message, "math_agent")

# Handle a message with retry
result = handler.handle_message_with_retry(message, "text_agent", max_retries=3)
```

## API Reference

### Message Creation Functions

- `create_system_message(content: str, metadata: dict[str, object] | None = None) -> SystemMessage`
- `create_human_message(content: str, metadata: dict[str, object] | None = None) -> HumanMessage`
- `create_ai_message(content: str, metadata: dict[str, object] | None = None) -> AIMessage`
- `create_tool_message(content: str, tool_call_id: str, metadata: dict[str, object] | None = None) -> ToolMessage`
- `create_structured_message(role: str, content: MessageValue, metadata: dict[str, MessageValue] | None = None) -> Message`
- `create_message_chain() -> MessageChain`

### Message Metadata Functions

- `get_message_metadata(message: Message, key: str, default: T | None = None) -> T | None`
- `set_message_metadata(message: Message, key: str, value: MessageValue) -> None`
- `get_message_at_index(messages: list[Message], index: int) -> Message`
- `get_metadata_at_index(messages: list[Message], index: int, key: str, default: T | None = None) -> T | None`
- `set_metadata_at_index(messages: list[Message], index: int, key: str, value: MessageValue) -> None`

### Message Validation Functions

- `validate_message_content(message: Message, required_fields: list[str] | None = None) -> bool`
- `parse_structured_content(message: Message, default: T | None = None) -> dict[str, MessageValue] | T`

### MessageChain Class

- `add_message(message: Message, priority: MessagePriority = MessagePriority.NORMAL) -> None`
- `validate_chain() -> bool`
- `validate_message_chain() -> bool`
- `get_messages_by_type(msg_type: type[Message]) -> Iterator[Message]`
- `get_messages_by_priority(min_priority: MessagePriority = MessagePriority.LOW) -> Iterator[Message]`
- `search_messages(query: str, metadata_key: str | None = None) -> Iterator[Message]`
- `filter_messages(criteria: dict[str, Any] | None = None, filter_fn: Callable[[Message], bool] | None = None) -> list[Message]`
- `get_message_history(limit: int | None = None, include_metadata: bool = False) -> list[dict[str, Any]]`

### MessageRouter Class

- `register_agent(name: str, agent: Agent[T, U]) -> None`
- `route_message(message: Message, target_agent: str, priority: MessagePriority = MessagePriority.NORMAL) -> StepResult[U]`
- `route_message_stream(message: Message, target_agent: str, priority: MessagePriority = MessagePriority.NORMAL) -> AsyncGenerator[str, None]`
- `broadcast_message(message: Message, priority: MessagePriority = MessagePriority.NORMAL) -> dict[str, StepResult[U]]`
- `get_agent_names() -> list[str]`
- `get_message_history(limit: int | None = None, include_metadata: bool = False) -> list[dict[str, Any]]`

### MessageHandler Class

- `handle_message(message: Message, priority: MessagePriority = MessagePriority.NORMAL) -> None`
- `track_message_history(message: Message) -> None`
- `validate_message_chain() -> bool`
- `filter_messages(**criteria) -> list[Message]`
- `get_message_history(limit: int | None = None, include_metadata: bool = False) -> list[dict[str, Any]]`
- `route_to_agent(message: Message, agent_name: str, priority: MessagePriority = MessagePriority.NORMAL) -> StepResult[Any]`
- `route_to_agent_stream(message: Message, agent_name: str, priority: MessagePriority = MessagePriority.NORMAL) -> AsyncGenerator[str, None]`
- `register_agent(name: str, agent: Agent[Any, Any]) -> None`
- `handle_message_with_retry(message: Message, agent_name: str, max_retries: int = 3, priority: MessagePriority = MessagePriority.NORMAL) -> StepResult[Any]`

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
