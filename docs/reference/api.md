# API Reference

This document provides detailed information about the key classes and functions in the Agentic Problem Solver.

## Agents

### SolverAgent

The main agent class that processes tasks and generates solutions.

```python
class SolverAgent:
    """Agent that solves programming problems."""

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
def process(self, message: Message) -> Result[str]:
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
def get_provider(cls, name: str) -> type[BaseLLMProvider]:
    """Get a provider class.

    Args:
        name: Provider name.

    Returns:
        Provider class.

    Raises:
        ProviderNotFoundError: If provider not found.
    """
```

##### get_provider_instance

```python
def get_provider_instance(
    self,
    name: str | None = None,
    capabilities: list[str] | None = None,
    temperature: float | None = None,
) -> BaseLLMProvider:
    """Get provider instance.

    Args:
        name: Provider name.
        capabilities: Required capabilities.
        temperature: Temperature setting.

    Returns:
        Provider instance.

    Raises:
        ConfigError: If provider creation fails.
    """
```

### BaseLLMProvider

Base class for LLM providers.

```python
class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize provider.

        Args:
            api_key: Optional API key.
        """
        self.config = self._create_config(api_key)
        self._validate_config()

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.
        """

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate text from prompt as a stream.

        Args:
            prompt: Input prompt.

        Yields:
            Generated text chunks.
        """
```

## Message System

### Message Creation

#### create_human_message

```python
def create_human_message(
    content: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: object
) -> HumanMessage:
    """Create a human message.

    Args:
        content: Message content.
        metadata: Optional metadata.
        **kwargs: Additional keyword arguments.

    Returns:
        Human message.
    """
```

#### create_ai_message

```python
def create_ai_message(
    content: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: object
) -> AIMessage:
    """Create an AI message.

    Args:
        content: Message content.
        metadata: Optional metadata.
        **kwargs: Additional keyword arguments.

    Returns:
        AI message.
    """
```

#### create_system_message

```python
def create_system_message(
    content: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: object
) -> SystemMessage:
    """Create a system message.

    Args:
        content: Message content.
        metadata: Optional metadata.
        **kwargs: Additional keyword arguments.

    Returns:
        System message.
    """
```

## Constants

### AgentStep

Enum defining the steps in the agent's workflow.

```python
class AgentStep(str, Enum):
    UNDERSTAND = "understand"    # Analyze and comprehend the task
    PLAN = "plan"                # Create a strategy to solve the task
    EXECUTE = "execute"          # Implement the planned solution
    VERIFY = "verify"            # Test and validate the solution
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: API key for Google's Gemini model
- `PROVIDER`: LLM provider to use (default: "gemini")
- `LOG_LEVEL`: Logging level (default: "INFO")
- `LOG_FILE`: Path to log file (optional)
