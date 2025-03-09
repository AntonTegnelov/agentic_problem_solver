# API Reference

This document provides detailed information about the key classes and functions in the Agentic Problem Solver.

## Agents

### SolverAgent

The main agent class that processes tasks and generates solutions.

```python
class SolverAgent(BaseAgent):
    def __init__(self, name: str = "solver", config: Optional[Dict[str, Any]] = None):
        """Initialize the solver agent.

        Args:
            name: Name of the agent (default: solver)
            config: Optional configuration dictionary with settings like:
                   - temperature: float (0.0 to 1.0)
                   - max_tokens: int
                   - model: str
        """
```

#### Methods

##### \_process_message_impl

```python
async def _process_message_impl(self, message: HumanMessage) -> AIMessage:
    """Process a message through the agent graph.

    Args:
        message: The human message to process

    Returns:
        AIMessage: The agent's response
    """
```

##### clear_state

```python
def clear_state(self) -> None:
    """Clear the agent's state.

    Resets all state variables to their initial values.
    """
```

## LLM Providers

### LLMProviderFactory

Factory class for creating and managing LLM providers.

```python
class LLMProviderFactory:
    def __init__(self):
        """Initialize the LLM provider factory."""
```

#### Methods

##### get_provider

```python
def get_provider(self) -> BaseLLMProvider:
    """Get the configured LLM provider instance.

    Returns:
        BaseLLMProvider: The configured provider
    """
```

##### set_provider

```python
def set_provider(self, provider_name: str) -> None:
    """Set the active LLM provider.

    Args:
        provider_name: Name of the provider to use
    """
```

### BaseLLMProvider

Base class for LLM providers.

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> Union[str, AIMessage]:
        """Generate a response for the given prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated response as string or AIMessage
        """
```

## Utilities

### Logging

#### setup_logging

```python
def setup_logging(
    log_file: Optional[str] = None,
    level: Union[str, int] = logging.INFO,
    format_str: str = "%(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s",
) -> None:
    """Set up logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level (can be string or int)
        format_str: Log message format string
    """
```

#### get_logger

```python
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
```

### Messages

#### create_system_message

```python
def create_system_message(
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> SystemMessage:
    """Create a SystemMessage with proper initialization.

    Args:
        content: Message content
        metadata: Optional metadata dictionary

    Returns:
        Initialized SystemMessage
    """
```

## Constants

### AgentStep

Enum defining the steps in the agent's workflow.

```python
class AgentStep(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    END = "END"
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: API key for Google's Gemini model
- `PROVIDER`: LLM provider to use (default: "gemini")
- `LOG_LEVEL`: Logging level (default: "INFO")
- `LOG_FILE`: Path to log file (optional)
