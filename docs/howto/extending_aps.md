# How to Extend APS

This guide shows you how to extend APS with new functionality.

## Adding a New LLM Provider

1. Create a new provider class:

```python
from src.llm_providers.base import BaseProvider
from src.messages import Message

class NewProvider(BaseProvider):
    """Implementation for a new LLM provider."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        # Initialize provider-specific client

    def generate(self, messages: List[Message], **kwargs) -> str:
        # Implement generation logic
        ...

    def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        # Implement streaming logic
        ...
```

2. Register the provider:

```python
# src/llm_providers/factory.py
from src.llm_providers.new_provider import NewProvider

PROVIDER_MAP = {
    "new": NewProvider,
    # ... existing providers ...
}
```

## Creating Custom Agent Steps

1. Define a new step:

```python
from src.agent.steps.base import Step
from src.agent.state import AgentState

class CustomStep(Step):
    """Custom processing step."""

    name = "custom"
    required_keys = ["input_data"]
    optional_keys = ["extra_param"]

    def execute(self, state: AgentState) -> StepResult:
        # Implement step logic
        input_data = state.context["input_data"]
        result = self._process(input_data)
        return {
            "success": True,
            "message": "Custom step completed",
            "data": {"result": result}
        }

    def _process(self, data: str) -> str:
        # Internal processing logic
        ...
```

2. Use the step:

```python
from src.agent.steps.custom import CustomStep

agent = BaseAgent()
agent.add_step(CustomStep())
```

## Adding Message Handlers

1. Create a handler:

```python
from src.messages.base import MessageHandler
from src.messages import Message

class CustomHandler(MessageHandler):
    """Custom message processing."""

    def process(self, message: Message) -> Message:
        # Implement message transformation
        new_content = self._transform(message.content)
        return Message(
            role=message.role,
            content=new_content,
            type=message.type
        )

    def _transform(self, content: str) -> str:
        # Transform logic
        ...
```

2. Register the handler:

```python
# src/messages/registry.py
from src.messages.handlers.custom import CustomHandler

HANDLERS = {
    "custom": CustomHandler,
    # ... existing handlers ...
}
```

## Creating Custom Configurations

1. Define configuration:

```python
from src.config.base import BaseConfig
from pydantic import Field

class CustomConfig(BaseConfig):
    """Custom component configuration."""

    special_param: str = Field(
        default="default_value",
        description="Special parameter for custom component"
    )

    retry_limit: int = Field(
        default=3,
        ge=1,
        description="Number of retries"
    )
```

2. Use configuration:

```python
from src.config.custom import CustomConfig

config = CustomConfig(
    special_param="custom_value",
    retry_limit=5
)
```

## Adding CLI Commands

1. Create command:

```python
# src/cli/commands/custom.py
import click
from src.cli.base import cli

@cli.command()
@click.argument("input_data")
@click.option("--param", default="default")
def custom(input_data: str, param: str):
    """Custom command description."""
    # Implement command logic
    ...
```

2. Register command:

```python
# src/cli/__init__.py
from src.cli.commands import custom
```

## Creating Custom Error Types

1. Define errors:

```python
from src.errors import APSError

class CustomError(APSError):
    """Custom error type."""

    def __init__(self, message: str, details: Dict[str, Any]):
        self.details = details
        super().__init__(f"Custom error: {message}")
```

2. Use errors:

```python
try:
    # Some operation
    raise CustomError("Operation failed", {"reason": "Invalid input"})
except CustomError as e:
    logger.error(f"Error occurred: {e}, Details: {e.details}")
```

## Adding Tests

1. Unit tests:

```python
# tests/test_custom.py
import pytest
from src.custom import CustomComponent

def test_custom_component():
    component = CustomComponent()
    result = component.process("test input")
    assert result == "expected output"

@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2")
])
def test_multiple_cases(input, expected):
    component = CustomComponent()
    assert component.process(input) == expected
```

2. Integration tests:

```python
# tests/integration/test_custom.py
import pytest
from src.agent import Agent
from src.custom import CustomComponent

@pytest.mark.integration
def test_custom_integration():
    agent = Agent()
    component = CustomComponent()
    agent.add_component(component)

    result = agent.process("test input")
    assert result.status == "success"
```

## Best Practices

1. **Documentation**

   - Add docstrings to all classes and methods
   - Include usage examples
   - Document configuration options

2. **Error Handling**

   - Use custom error types
   - Include detailed error messages
   - Provide recovery suggestions

3. **Testing**

   - Write unit tests for new components
   - Add integration tests
   - Test edge cases

4. **Type Safety**

   - Use type hints
   - Add runtime type checking
   - Document type requirements

5. **Configuration**
   - Use Pydantic models
   - Include validation
   - Provide sensible defaults
