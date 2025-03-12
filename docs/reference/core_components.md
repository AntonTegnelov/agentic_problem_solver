# Core Components Reference

## Agent System

### Agent Protocol

```python
from typing import TypeVar, Protocol, Any, AsyncGenerator

from src.common_types.message_types import Message
from src.common_types.result_types import Result

T = TypeVar("T")

class Agent(Protocol[T]):
    """Agent protocol."""

    def process(self, message: Message) -> Result[T]:
        """Process a message."""
        ...

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message and stream results."""
        ...

    def get_agent_id(self) -> str:
        """Get agent ID."""
        ...

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities."""
        ...

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task."""
        ...

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent."""
        ...

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent."""
        ...
```

### AgentState

```python
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, list, dict

from src.common_types.enums import AgentStep
from src.common_types.message_types import Message
from src.common_types.result_types import Result

@dataclass
class AgentState:
    """Agent state management."""

    messages: list[Message] = field(default_factory=list)
    context: Context = field(default_factory=Context)
    execution_result: str = ""
    current_step: AgentStep = field(default=AgentStep.UNDERSTAND)
    step_count: int = field(default=0)
    task_completed: bool = field(default=False)
    error: str | None = field(default=None)
    step_results: dict[str, Result[Any]] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    agent_id: str = field(default="")
    parent_agent_id: str | None = field(default=None)
    _agents: dict[str, Agent] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the state."""
        ...

    def get_context(self, key: str, default: T | None = None) -> T | None:
        """Get context value."""
        ...

    def set_context(self, key: str, value: T) -> None:
        """Set context value."""
        ...

    def clear(self) -> None:
        """Clear state."""
        ...

    def validate(self) -> bool:
        """Validate state."""
        ...

    def record_step_result(self, step: AgentStep, result: Result[Any]) -> None:
        """Record result for a step."""
        ...

    def get_step_result(self, step: AgentStep) -> Result[Any] | None:
        """Get result for a step."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Create state from dictionary."""
        ...

    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """Register agent."""
        ...

    def get_agent_for_step(self, step: AgentStep) -> Agent:
        """Get agent for step."""
        ...
```

### StateManager Protocol

```python
from typing import Protocol

class StateManager(Protocol):
    """State manager protocol."""

    def get_state(self) -> AgentState:
        """Get current state."""
        ...

    def set_state(self, state: AgentState) -> None:
        """Set current state."""
        ...

    def clear_state(self) -> None:
        """Clear current state."""
        ...

    def save_state(self, path: str | None = None) -> str:
        """Save state to file."""
        ...

    def load_state(self, path: str) -> AgentState:
        """Load state from file."""
        ...
```

## Provider System

### LLMProvider Protocol

```python
from typing import Protocol, runtime_checkable, AsyncGenerator
from src.common_types.message_types import Message
from src.llm_providers.type_defs import GenerationConfig

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    def generate(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate response from messages."""
        ...

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response stream from messages."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def validate_config(self, config: GenerationConfig) -> None:
        """Validate configuration."""
        ...
```

### BaseLLMProvider

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize provider."""
        self.config = self._create_config(api_key)
        self._validate_config()

    @abstractmethod
    def _create_config(self, api_key: str | None = None) -> ProviderConfig:
        """Create provider configuration."""
        ...

    def _validate_config(self) -> None:
        """Validate provider configuration."""
        ...

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate text from prompt as a stream."""
        ...

    def update_config(self, config: dict[str, str]) -> None:
        """Update provider configuration."""
        ...

    def get_config(self) -> dict[str, str]:
        """Get current configuration."""
        ...
```

### LLMProviderFactory

```python
class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_cls: type[BaseLLMProvider],
        version: ProviderVersion | None = None,
    ) -> None:
        """Register a provider."""
        ...

    @classmethod
    def get_provider(cls, name: str) -> type[BaseLLMProvider]:
        """Get a provider class."""
        ...

    def get_provider_instance(
        self,
        name: str | None = None,
        capabilities: list[str] | None = None,
        temperature: float | None = None,
    ) -> BaseLLMProvider:
        """Get provider instance."""
        ...
```

## Message System

### Message Types

```python
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage as Message,
)

# Type aliases
MessageValue = Union[str, int, float, bool, dict[str, Any], list[Any], None]
CriteriaValue = Union[str, int, float, bool, None]
CriteriaDict = dict[str, CriteriaValue]
```

### MessageChain

```python
class MessageChain:
    """Message chain for tracking conversation history."""

    def __init__(self, messages: list[Message] | None = None) -> None:
        """Initialize message chain."""
        self._messages: list[Message] = messages or []

    def add_message(self, message: Message, priority: MessagePriority = MessagePriority.NORMAL) -> None:
        """Add a message to the chain."""
        ...

    def validate_chain(self) -> bool:
        """Validate the message chain."""
        ...

    def get_messages_by_type(self, msg_type: type[Message]) -> Iterator[Message]:
        """Get messages by type."""
        ...

    def get_messages_by_priority(self, min_priority: MessagePriority = MessagePriority.LOW) -> Iterator[Message]:
        """Get messages by priority."""
        ...

    def search_messages(self, query: str, metadata_key: str | None = None) -> Iterator[Message]:
        """Search messages by content or metadata."""
        ...

    def filter_messages(self, criteria: dict[str, Any] | None = None, filter_fn: Callable[[Message], bool] | None = None) -> list[Message]:
        """Filter messages by criteria or function."""
        ...

    def get_message_history(self, limit: int | None = None, include_metadata: bool = False) -> list[dict[str, Any]]:
        """Get message history."""
        ...

    def clear(self) -> None:
        """Clear all messages from the chain."""
        ...
```

### MessageRouter

```python
class MessageRouter:
    """Message router for directing messages between agents."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> None:
        """Initialize router."""
        ...

    def register_agent(self, name: str, agent: Agent) -> None:
        """Register an agent."""
        ...

    def unregister_agent(self, name: str) -> None:
        """Unregister an agent."""
        ...

    def add_route(self, source: str, target: str) -> None:
        """Add a route between agents."""
        ...

    def remove_route(self, source: str, target: str) -> None:
        """Remove a route between agents."""
        ...

    async def route_message(self, message: Message, target_agent: str) -> Result:
        """Route a message to an agent."""
        ...

    async def route_message_stream(self, message: Message, target_agent: str) -> AsyncGenerator[str, None]:
        """Stream results from an agent."""
        ...

    async def broadcast_message(self, message: Message) -> dict[str, Result]:
        """Broadcast a message to all agents."""
        ...

    def get_agent_names(self) -> list[str]:
        """Get all registered agent names."""
        ...
```

### MessageHandler

```python
@dataclass
class MessageHandler:
    """Message handler."""

    handlers: dict[str, Callable[[Message], None]] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    message_chain: MessageChain = field(default_factory=MessageChain)
    _sequence: int = field(default=0)
    router: MessageRouter = field(default_factory=MessageRouter)

    def register_handler(self, message_type: str, handler: Callable[[Message], None]) -> None:
        """Register message handler."""
        ...

    def register_agent(self, name: str, agent: Agent) -> None:
        """Register an agent."""
        ...

    def handle_message(self, message: Message) -> None:
        """Handle a message."""
        ...

    def track_message_history(self, message: Message) -> None:
        """Track message history."""
        ...

    def validate_message_chain(self) -> bool:
        """Validate the message chain."""
        ...

    def filter_messages(self, **criteria) -> list[Message]:
        """Filter messages by criteria."""
        ...

    def get_message_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get message history."""
        ...

    async def route_to_agent(self, message: Message, agent_name: str) -> Result:
        """Route a message to an agent."""
        ...

    async def route_to_agent_stream(self, message: Message, agent_name: str) -> AsyncGenerator[str, None]:
        """Stream results from an agent."""
        ...

    async def handle_message_with_retry(self, message: Message, agent_name: str, max_retries: int = 3) -> Result:
        """Handle a message with retry."""
        ...
```

## Configuration System

### ProviderConfig

```python
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

@dataclass
class ProviderConfig:
    """Provider configuration."""

    api_key: str
    api_base: str | None = None
    model: str = "gemini-2.0-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 40
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        ...

    def update(self, config: dict[str, str]) -> None:
        """Update configuration."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...
```

### AgentConfig

```python
@dataclass
class AgentConfig:
    """Agent configuration."""

    name: str = "agent"
    model: str = "gemini-2.0-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 1024
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        ...

    def update(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...
```

## CLI System

### Commands

```python
@click.group()
def cli():
    """APS command line interface."""
    ...

@cli.command()
def solve(prompt: str):
    """Solve a programming task."""
    ...

@cli.command()
def config(key: str, value: str):
    """Configure APS settings."""
    ...
```

## Error Handling

### Exception Types

```python
class ConfigError(Exception):
    """Configuration error."""
    ...

class AgentNotFoundError(Exception):
    """Agent not found error."""
    ...

class RetryError(Exception):
    """Retry error."""
    ...

class InvalidModelError(Exception):
    """Invalid model error."""
    ...

class ProviderNotFoundError(Exception):
    """Provider not found error."""
    ...

class EmptyResponseError(Exception):
    """Empty response error."""
    ...
```

## Type Definitions

### Enums

```python
class AgentStatus(str, Enum):
    """Agent status."""
    IDLE = "idle"                # Agent is not currently processing any task
    BUSY = "busy"                # Agent is actively processing a task
    PROCESSING = "processing"    # Alias for BUSY for backward compatibility
    ERROR = "error"              # Agent encountered an error during processing
    COMPLETED = "completed"      # Agent has completed its task
    DONE = "done"                # Alias for COMPLETED for backward compatibility

class AgentStep(str, Enum):
    """Agent steps."""
    UNDERSTAND = "understand"    # Analyze and comprehend the task
    PLAN = "plan"                # Create a strategy to solve the task
    EXECUTE = "execute"          # Implement the planned solution
    VERIFY = "verify"            # Test and validate the solution

class MessageRole(str, Enum):
    """Message roles."""
    SYSTEM = "system"            # System-level instructions or context
    USER = "user"                # Input from the user
    ASSISTANT = "assistant"      # Responses from the AI assistant
    TOOL = "tool"                # Output from tools or function calls

class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "debug"              # Detailed information for debugging
    INFO = "info"                # General information about program execution
    WARNING = "warning"          # Indicate a potential problem
    ERROR = "error"              # A more serious problem
    CRITICAL = "critical"        # A critical problem that may prevent program execution

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
```

### Result Type

```python
@dataclass
class Result(Generic[T]):
    """Result type for operations."""

    success: bool
    data: T | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```
