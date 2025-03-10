# Core Components Reference

## Agent System

### BaseAgent

```python
class BaseAgent(Protocol):
    """Base agent interface."""

    def process(self, input_data: str) -> str:
        """Process input and return result."""
        ...

    def add_step(self, step: Step) -> None:
        """Add a processing step."""
        ...

    def clear_steps(self) -> None:
        """Clear all processing steps."""
        ...
```

### AgentState

```python
class AgentState:
    """Agent state management."""

    messages: List[Message]
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    current_step: Optional[Step]
    error: Optional[Exception]
```

### Step

```python
class Step(Protocol):
    """Step interface."""

    name: str
    required_keys: List[str]
    optional_keys: Optional[List[str]]

    def execute(self, state: AgentState) -> StepResult:
        """Execute the step."""
        ...
```

## Provider System

### BaseProvider

```python
class BaseProvider(Protocol):
    """Base LLM provider interface."""

    def generate(self, messages: List[Message], **kwargs) -> str:
        """Generate response from messages."""
        ...

    def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Stream response from messages."""
        ...
```

### ProviderFactory

```python
class ProviderFactory:
    """Provider factory."""

    @classmethod
    def create(cls, provider_type: str, **config) -> BaseProvider:
        """Create provider instance."""
        ...
```

## Message System

### Message

```python
class Message:
    """Message structure."""

    role: MessageRole
    content: str
    type: Optional[str]
    tool_call_id: Optional[str]
    additional_kwargs: Dict[str, Any]
```

### MessageHandler

```python
class MessageHandler(Protocol):
    """Message handler interface."""

    def process(self, message: Message) -> Message:
        """Process a message."""
        ...
```

## Configuration System

### BaseConfig

```python
class BaseConfig:
    """Base configuration."""

    model: str
    temperature: float
    max_tokens: int
    timeout: int
    retry_count: int
```

### AgentConfig

```python
class AgentConfig(BaseConfig):
    """Agent configuration."""

    max_steps: int
    memory_size: int
    step_timeout: int
```

### ProviderConfig

```python
class ProviderConfig(BaseConfig):
    """Provider configuration."""

    api_key: str
    api_base: Optional[str]
    organization: Optional[str]
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

### Exceptions

```python
class APSError(Exception):
    """Base error for APS."""
    ...

class ProviderError(APSError):
    """Provider-related errors."""
    ...

class ConfigError(APSError):
    """Configuration-related errors."""
    ...

class AgentError(APSError):
    """Agent-related errors."""
    ...
```

## Type Definitions

### Enums

```python
class MessageRole(str, Enum):
    """Message role types."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class StepType(str, Enum):
    """Step types."""
    UNDERSTAND = "understand"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
```

### Types

```python
StepResult = TypedDict("StepResult", {
    "success": bool,
    "message": str,
    "data": Optional[Dict[str, Any]],
    "error": Optional[Exception]
})

Context = Dict[str, Any]
StateKwargs = TypedDict("StateKwargs", {
    "messages": List[Message],
    "context": Context
})
```
