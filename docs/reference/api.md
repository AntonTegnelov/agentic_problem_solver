# API Reference

This document provides detailed information about the key classes and functions in the Agentic Problem Solver.

## Agents

### ArchitectAgent

Top-level agent in the hierarchical system responsible for high-level task decomposition and system design.

```python
class ArchitectAgent:
    """Agent that handles high-level task decomposition and system design."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        state_manager: AgentState | StateManager | None = None,
        config: AgentConfig | None = None,
    ):
        """Initialize agent.

        Args:
            provider: LLM provider.
            state_manager: State manager or agent state.
            config: Agent configuration.
        """
```

#### Methods

##### process

```python
async def process(self, message: Message) -> Result[str]:
    """Process a message.

    Args:
        message: Message to process.

    Returns:
        Result containing the processed message.
    """
```

##### process_stream

```python
async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
    """Process a message and stream results.

    Args:
        message: Message to process.

    Yields:
        Processed output chunks.
    """
```

##### get_agent_id

```python
def get_agent_id(self) -> str:
    """Get agent ID.

    Returns:
        Agent ID.
    """
```

### PlannerAgent

Mid-level agent in the hierarchical system responsible for detailed task planning and refinement.

See the [Hierarchical Agent System](../explanation/hierarchical_agents.md) documentation for details.

### ExecutorAgent

Bottom-level agent in the hierarchical system responsible for implementing specific tasks.

See the [Hierarchical Agent System](../explanation/hierarchical_agents.md) documentation for details.

## LLM Providers

### LLMProviderFactory

Factory class for creating and managing LLM providers.

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
        """Register a provider.

        Args:
            name: Provider name.
            provider_cls: Provider class.
            version: Optional provider version info.
        """
```

#### Methods

##### get_provider

```python
@classmethod
def get_provider(
    cls,
    name: str | None = None,
    config: LLMConfig | None = None,
) -> BaseLLMProvider:
    """Get a provider instance.

    Args:
        name: Provider name.
        config: Provider configuration.

    Returns:
        Provider instance.
    """
```

##### create_provider

```python
@classmethod
def create_provider(
    cls,
    name: str,
    config: LLMConfig | None = None,
) -> BaseLLMProvider:
    """Create a new provider instance.

    Args:
        name: Provider name.
        config: Provider configuration.

    Returns:
        Provider instance.
    """
```

## Message System

### Message

Base message class for communication between components.

```python
class Message:
    """Message class for communication between components."""

    def __init__(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
    ):
        """Initialize message.

        Args:
            role: Message role.
            content: Message content.
            metadata: Optional message metadata.
        """
```

### MessageChain

Container for a sequence of related messages.

```python
class MessageChain:
    """Container for a sequence of related messages."""

    def __init__(
        self,
        messages: list[Message] | None = None,
    ):
        """Initialize message chain.

        Args:
            messages: Optional initial messages.
        """
```

#### Methods

##### add_message

```python
def add_message(self, message: Message) -> None:
    """Add a message to the chain.

    Args:
        message: Message to add.
    """
```

##### get_messages

```python
def get_messages(self) -> list[Message]:
    """Get all messages in the chain.

    Returns:
        List of messages.
    """
```

## Configuration

### AgentConfig

Configuration for agent behavior.

```python
class AgentConfig(BaseConfig):
    """Configuration for agent behavior."""

    def __init__(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retry_count: int | None = None,
        retry_delay: float | None = None,
        **kwargs,
    ):
        """Initialize agent configuration.

        Args:
            max_tokens: Maximum tokens to generate.
            temperature: Temperature for generation.
            retry_count: Number of retries for failed operations.
            retry_delay: Delay between retries in seconds.
            **kwargs: Additional configuration options.
        """
```

### LLMConfig

Configuration for LLM providers.

```python
class LLMConfig(BaseConfig):
    """Configuration for LLM providers."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        """Initialize LLM configuration.

        Args:
            api_key: API key for the provider.
            model: Model name.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional configuration options.
        """
```
