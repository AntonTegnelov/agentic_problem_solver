# Migrating from Agent ABC to Agent Protocol

This guide provides step-by-step instructions for migrating from the abstract base class (ABC) implementation of Agent to the Protocol-based approach.

## Background

The Agentic Problem Solver codebase currently has two different Agent definitions:

1. **Agent ABC** (`src/agent/base.py`): A traditional abstract base class using Python's `ABC` module
2. **Agent Protocol** (`src/agent/agent_types/agent_types.py`): A modern Protocol-based interface using Python's typing module

We are standardizing on the Protocol-based approach because it:

- Provides more flexibility without requiring inheritance
- Supports generic type parameters for better type checking
- Aligns better with the architecture's emphasis on clean interfaces
- Encourages composition over inheritance
- Is more consistent with modern Python typing practices

## Migration Timeline

The migration will happen in phases to minimize disruption:

1. **Phase 1: Deprecation and Warning** (Current)

   - Deprecation warnings added to all ABC methods
   - Usage tracking in logs
   - Backward compatibility maintained

2. **Phase 2: Dual Support**

   - SolverAgent updated to implement Protocol
   - Adapter classes available for legacy code
   - Tests updated to use Protocol-based approach

3. **Phase 3: Protocol Dominance**

   - ABC becomes a wrapper around Protocol
   - All remaining direct usages of ABC updated

4. **Phase 4: ABC Removal**
   - ABC implementation removed entirely
   - All code uses Protocol-based approach

## How to Migrate Your Code

### For Agent Implementations

If you have a class that inherits from `Agent` ABC:

```python
# Before
from src.agent.base import Agent

class MyAgent(Agent):
    def get_agent_id(self) -> str:
        return "my_agent"

    # ... other method implementations
```

Change it to implement the Protocol:

```python
# After
from src.agent.agent_types import Agent

class MyAgent:  # No inheritance needed
    def get_agent_id(self) -> str:
        return "my_agent"

    # ... other method implementations
```

### For Code That Uses Agents

If your code expects an Agent ABC:

```python
# Before
from src.agent.base import Agent

def process_with_agent(agent: Agent) -> None:
    # ...
```

Change it to use the Protocol:

```python
# After
from src.agent.agent_types import Agent

def process_with_agent(agent: Agent[Any]) -> None:
    # ...
```

### Using Adapter Classes During Transition

If you need to use a Protocol-based Agent with code that expects an ABC:

```python
from src.agent.adapters import ProtocolToABCAdapter
from src.agent.agent_types import Agent as AgentProtocol

# Your Protocol-based agent
protocol_agent: AgentProtocol[Any] = MyProtocolAgent()

# Adapt it to work with code expecting ABC
abc_compatible_agent = ProtocolToABCAdapter(protocol_agent)
```

Or if you need to use an ABC-based Agent with code that expects a Protocol:

```python
from src.agent.adapters import ABCToProtocolAdapter
from src.agent.base import Agent as AgentABC

# Your ABC-based agent
abc_agent: AgentABC = MyABCAgent()

# Adapt it to work with code expecting Protocol
protocol_compatible_agent = ABCToProtocolAdapter(abc_agent)
```

## Best Practices

1. **For New Code**:

   - Always use the Protocol-based approach
   - Import from `src.agent.agent_types`

2. **For Existing Code**:

   - Gradually migrate to the Protocol-based approach
   - Use adapter classes for complex migrations
   - Ensure test coverage during migration

3. **Testing**:
   - Update tests to use Protocol-based approach
   - Test with both approaches during transition
   - Verify behavior is identical

## Troubleshooting

### Deprecation Warnings

If you see deprecation warnings:

```
DeprecationWarning: Call to deprecated method get_agent_id. Use src.agent.agent_types.agent_types.Agent Protocol instead.
```

This indicates you're still using the ABC implementation. Follow this guide to migrate to the Protocol approach.

### Type Checking Errors

If you encounter type checking errors:

```
error: "MyAgent" has no attribute "get_agent_id" (not a protocol member)
```

Ensure your class implements all required methods of the Agent Protocol.

## Getting Help

If you encounter issues during migration, please:

1. Check the detailed migration plan in `src/agent/base.py`
2. Review the adapter classes in `src/agent/adapters.py`
3. Consult the architecture documentation in `docs/explanation/architecture.md`
4. Reach out to the development team for assistance

## References

- [PEP 544 – Protocols: Structural subtyping (static duck typing)](https://peps.python.org/pep-0544/)
- [Python typing documentation](https://docs.python.org/3/library/typing.html#protocols)
- [Agent Protocol definition](src/agent/agent_types/agent_types.py)
- [Migration plan](src/agent/base.py)
