from typing import Any
from unittest.mock import AsyncMock, Mock

from langchain.schema import AIMessage

from src.agents.solver_agent import SolverAgent


def create_test_agent(
    name: str = "test_agent", config: dict[str, Any] | None = None
) -> SolverAgent:
    """Create a test agent with a mock LLM provider."""
    mock_provider = Mock()

    async def mock_generate(prompt: str) -> AIMessage:
        if "Analyze this programming request" in prompt:
            return AIMessage(
                content=(
                    "Need to implement a simple hello_world function that prints "
                    "'Hello, world!' to the console."
                )
            )
        if "Outline a brief implementation plan" in prompt:
            return AIMessage(
                content=(
                    "1. Create hello_world function\n"
                    "2. Add print statement\n"
                    "3. Add example usage"
                )
            )
        if "Write clean, well-documented code" in prompt:
            return AIMessage(
                content="""[CODE]
def hello_world():
    \"\"\"Print 'Hello, world!' to the console.\"\"\"
    print("Hello, world!")

if __name__ == "__main__":
    hello_world()
[/CODE]"""
            )
        if "Reviewing the implementation" in prompt:
            return AIMessage(
                content="""Here's your solution:

```python
def hello_world():
    \"\"\"Print 'Hello, world!' to the console.\"\"\"
    print("Hello, world!")

if __name__ == "__main__":
    hello_world()
```

Verification:
1. Code works as expected
2. No error handling needed for this simple function
3. Documentation is clear and concise"""
            )
        return AIMessage(content="[CODE]No response[/CODE]")

    mock_provider.generate = AsyncMock(side_effect=mock_generate)
    mock_provider.generate_stream = AsyncMock(side_effect=mock_generate)
    mock_factory = Mock()
    mock_factory.get_provider = Mock(return_value=mock_provider)
    return SolverAgent(name=name, config=config, llm_factory=mock_factory)
