# Agent System Reference

## Overview

The Agent System is a foundational component of the Agentic Problem Solver (APS) framework, providing a flexible and extensible architecture for creating, managing, and coordinating intelligent agents. It enables agents to process information, make decisions, and communicate with each other to solve complex problems.

## Key Components

### Agent Protocol

The `Agent` protocol defines the interface that all agents must implement, providing a consistent way to interact with agents regardless of their specific implementation.

### Agent State

The `AgentState` class manages the state of an agent during execution, including:

- Messages received and sent
- Context data
- Execution results
- Current step in the problem-solving process
- Step results
- Error information

### State Management

The framework provides two state manager implementations:

- `InMemoryStateManager`: Stores agent state in memory
- `FileStateManager`: Persists agent state to disk

### Agent Steps

The agent system follows a structured problem-solving approach with four main steps:

1. **UNDERSTAND**: Analyze and comprehend the task
2. **PLAN**: Create a strategy to solve the task
3. **EXECUTE**: Implement the planned solution
4. **VERIFY**: Test and validate the solution

### Agent Coordination

The agent coordination system enables multiple agents to work together:

- `AgentRegistry`: Manages agent registration and discovery
- `AgentCoordinator`: Coordinates task delegation and message routing between agents

## Usage Examples

### Creating an Agent

```python
from src.agent.agent_types.agent_types import Agent, Result
from src.common_types.message_types import Message
from typing import Any, AsyncGenerator

class MyAgent(Agent[str]):
    """Custom agent implementation."""

    def __init__(self, agent_id: str):
        """Initialize agent."""
        self.agent_id = agent_id

    def process(self, message: Message) -> Result[str]:
        """Process a message.

        Args:
            message: Message to process

        Returns:
            Result containing the processed message
        """
        # Process the message
        return Result(success=True, data=f"Processed: {message.content}")

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message and stream results.

        Args:
            message: Message to process

        Yields:
            Processed output chunks
        """
        yield f"Processing: {message.content}"
        yield f"Completed processing"

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID
        """
        return self.agent_id

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities
        """
        return ["text_processing", "analysis"]

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check

        Returns:
            True if agent can handle task
        """
        return "text" in task.lower()

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send

        Returns:
            Result of message processing
        """
        return self.process(message)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive

        Returns:
            Result of message processing
        """
        return self.process(message)
```

### Managing Agent State

```python
from src.agent.state.base import AgentState, Context, FileStateManager
from src.common_types.enums import AgentStep
from src.common_types.result_types import Result
from src.messages.creation import create_human_message
from datetime import datetime, UTC

# Create agent state
state = AgentState(
    agent_id="test-agent",
    parent_agent_id="parent-agent"
)

# Add a message to the state
message = create_human_message("Solve this problem: 2 + 2")
state.add_message(message)

# Work with context data
state.context.data["task"] = "Solve a math problem"
state.set_context("difficulty", "easy")
difficulty = state.get_context("difficulty")

# Update the current step
state.current_step = AgentStep.PLAN
state.step_count += 1

# Record step result
result = Result(success=True, data="Planning to solve 2 + 2", error=None)
state.record_step_result(AgentStep.PLAN, result)

# Get step result
plan_result = state.get_step_result(AgentStep.PLAN)

# Convert state to dictionary
state_dict = state.to_dict()

# Create state from dictionary
new_state = AgentState.from_dict(state_dict)

# Create a file-based state manager
file_manager = FileStateManager("./state")

# Save state to file
state_path = file_manager.save_state(state)

# Load state from file
loaded_state = file_manager.load_state(state_path)

# Register an agent with the state
from src.agent.agent_types.agent_types import MockAgent
agent = MockAgent("test-agent", ["math"])
state.register_agent("test-agent", agent)

# Get agent for a step
step_agent = state.get_agent_for_step(AgentStep.EXECUTE)
```

### Processing Agent Steps

```python
from src.agent.state.base import AgentState
from src.common_types.enums import AgentStep
from src.common_types.result_types import Result
from src.agent.steps import (
    get_step_prompt,
    execute_step_with_retry,
    validate_step_result,
    get_next_step,
)

# Create agent state
state = AgentState()
state.set_context("task", "Create a Python function to calculate factorial")

# Get prompt for current step
state.current_step = AgentStep.UNDERSTAND
prompt = get_step_prompt(state)

# Execute step with retry
def execute_step():
    # Simulate LLM call
    return Result(
        success=True,
        data="I need to create a recursive function to calculate factorial.",
        error=None,
    )

result = execute_step_with_retry(state, execute_step, max_retries=2)

# Validate step result
validate_step_result(state, AgentStep.UNDERSTAND, result)

# Record step result
state.record_step_result(AgentStep.UNDERSTAND, result)

# Move to next step
next_step = get_next_step(state.current_step)
state.current_step = next_step
```

### Coordinating Multiple Agents

```python
from src.agent.agent_types.agent_types import Agent, SimpleAgentCoordinator
from src.agent.coordination import InMemoryAgentRegistry
from src.common_types import AgentInfo
from src.common_types.message_types import Message

# Create registry and coordinator
registry = InMemoryAgentRegistry()
coordinator = SimpleAgentCoordinator(registry)

# Create and register agents
math_agent = MyAgent("math_agent")
text_agent = MyAgent("text_agent")

math_info = AgentInfo(
    agent_id="math_agent",
    name="Math Agent",
    description="Handles mathematical tasks",
    capabilities=["math", "calculation"],
)

text_info = AgentInfo(
    agent_id="text_agent",
    name="Text Agent",
    description="Handles text processing tasks",
    capabilities=["text", "nlp"],
)

registry.register_agent(math_agent, math_info)
registry.register_agent(text_agent, text_info)

# Delegate tasks to agents
math_result = coordinator.delegate_task("Calculate 2 + 2", "math_agent")
text_result = coordinator.delegate_task("Summarize this text", "text_agent")

# Find agents by capability
math_agents = registry.find_agents_by_capability("math")

# Route messages between agents
from src.messages.creation import create_human_message
from src.messages.utils import set_message_metadata

message = create_human_message("Process this data")
set_message_metadata(message, "receiver_id", "text_agent")
result = coordinator.route_message(message)

# Broadcast task to agents with a specific capability
results = coordinator.broadcast_task("Analyze this data", "analysis")
```

## API Reference

### Agent Protocol

- `process(message: Message) -> Result[T]`: Process a message
- `process_stream(message: Message) -> AsyncGenerator[str, None]`: Process message and stream results
- `get_agent_id() -> str`: Get agent ID
- `get_capabilities() -> list[str]`: Get agent capabilities
- `can_handle(task: str) -> bool`: Check if agent can handle task
- `send_message(message: Message) -> Result[Any]`: Send message to agent
- `receive_message(message: Message) -> Result[Any]`: Receive message from another agent

### AgentState Class

- `add_message(message: Message) -> None`: Add message to state
- `get_message(index: int) -> Message`: Get message at index
- `get_metadata_at_index(index: int, key: str, default: T | None = None) -> T | None`: Get message metadata
- `set_metadata_at_index(index: int, key: str, value: Any) -> None`: Set message metadata
- `get_context(key: str, default: T | None = None) -> T | None`: Get context value
- `set_context(key: str, value: T) -> None`: Set context value
- `clear() -> None`: Clear state
- `validate() -> bool`: Validate state
- `record_step_result(step: AgentStep, result: Result[Any]) -> None`: Record step result
- `get_step_result(step: AgentStep) -> Result[Any] | None`: Get step result
- `to_dict() -> dict[str, Any]`: Convert state to dictionary
- `from_dict(data: dict[str, Any]) -> AgentState`: Create state from dictionary
- `register_agent(agent_id: str, agent: Agent) -> None`: Register agent
- `get_agent_for_step(step: AgentStep) -> Agent`: Get agent for step

### StateManager Protocol

- `get_state() -> AgentState`: Get current state
- `set_state(state: AgentState) -> None`: Set current state
- `clear_state() -> None`: Clear current state
- `save_state(path: str | None = None) -> str`: Save state to file
- `load_state(path: str) -> AgentState`: Load state from file

### FileStateManager Class

- `list_states() -> list[str]`: List available state files
- `get_state_by_id(agent_id: str) -> AgentState`: Get state by agent ID

### AgentRegistry Protocol

- `register_agent(agent: Agent[Any], info: AgentInfo) -> None`: Register an agent
- `unregister_agent(agent_id: str) -> None`: Unregister an agent
- `get_agent(agent_id: str) -> Agent[Any]`: Get agent by ID
- `get_agent_info(agent_id: str) -> AgentInfo`: Get agent information
- `list_agents() -> list[AgentInfo]`: List all registered agents
- `find_agents_by_capability(capability: str) -> list[AgentInfo]`: Find agents by capability
- `find_agents_by_parent(parent_id: str) -> list[AgentInfo]`: Find agents by parent ID

### InMemoryAgentRegistry Implementation

- `register_agent(agent: Agent, info: AgentInfo | None = None) -> None`: Register an agent
- `unregister_agent(agent_id: str) -> None`: Unregister an agent
- `get_agent(agent_id: str) -> Agent`: Get agent by ID
- `get_agent_info(agent_id: str) -> AgentInfo | None`: Get agent information
- `list_agents() -> list[Agent]`: List all registered agents
- `get_agents() -> dict[str, Agent]`: Get all agents as a dictionary
- `find_agents_by_capability(capability: str) -> list[Agent]`: Find agents by capability
- `find_agents_by_parent(parent_id: str) -> list[Agent]`: Find agents by parent ID

### SimpleAgentCoordinator

- `__init__(registry: AgentRegistry) -> None`: Initialize coordinator
- `register_agent_factory(agent_type: str, factory: callable) -> None`: Register agent factory
- `create_agent(agent_type: str, config: dict[str, Any]) -> Agent`: Create a new agent
- `delegate_task(task: str, agent_id: str) -> Result[Any]`: Delegate task to agent
- `broadcast_task(task: str, capability: str) -> dict[str, Result[Any]]`: Broadcast task to agents with capability
- `route_message(message: Message) -> Result[Any]`: Route message to target agent
- `get_agent_status(agent_id: str) -> str`: Get agent status
- `set_agent_status(agent_id: str, status: str) -> None`: Set agent status

### Prompt Functions

- `get_step_prompt(state: AgentState) -> str`: Get prompt for current step
- `get_retry_prompt(state: AgentState, error: str) -> str`: Get retry prompt for current step
- `validate_step_result(state: AgentState, step: AgentStep, result: StepResult[Any]) -> bool`: Validate step result
- `execute_step_with_retry(state: AgentState, execute_fn: callable, max_retries: int = 3) -> StepResult[Any]`: Execute step with retry
- `get_next_step(current_step: AgentStep) -> AgentStep`: Get next step in sequence
- `get_step_description(step: AgentStep) -> str`: Get description for step

## Error Handling

The agent system uses the following error types from `exceptions.py`:

- `ConfigError`: For configuration and validation errors
- `RetryError`: For retry-related errors

## Best Practices

1. **State Management**: Use appropriate state managers based on your needs (in-memory for simple cases, file-based for persistence).
2. **Step Validation**: Always validate step results to ensure quality outputs.
3. **Error Handling**: Implement proper error handling with retry mechanisms for robust agent execution.
4. **Agent Coordination**: Use the agent registry and coordinator for managing complex multi-agent systems.
5. **Context Tracking**: Use the context system to store and retrieve agent-specific data.
6. **Step Sequencing**: Follow the UNDERSTAND → PLAN → EXECUTE → VERIFY sequence for structured problem-solving.
