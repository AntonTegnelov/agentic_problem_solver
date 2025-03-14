# API Reference

This document provides detailed information about the key classes and functions in the Agentic Problem Solver.

## Agents

### SolverAgent (DEPRECATED)

> **IMPORTANT DEPRECATION NOTICE**:
>
> The `SolverAgent` class is deprecated and will be removed in a future version of the APS framework.
> It is maintained only for backward compatibility.
>
> Please migrate to the hierarchical agent system using:
>
> - `ArchitectAgent` for high-level task decomposition
> - `PlannerAgent` for mid-level task refinement
> - `ExecutorAgent` for low-level task execution
>
> For more information, see the [Hierarchical Agent System](../explanation/hierarchical_agents.md) documentation.

The original agent class that processes tasks and generates solutions.

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

### ArchitectAgent

Top-level agent in the hierarchical system responsible for high-level task decomposition and system design.

See the [Hierarchical Agent System](../explanation/hierarchical_agents.md) documentation for details.

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

```

```
