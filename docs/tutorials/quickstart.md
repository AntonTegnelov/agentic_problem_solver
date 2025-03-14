# Quick Start Guide

## Installation

```bash
pip install aps-solver
```

## Basic Usage

1. Set up your API key:

```bash
export GEMINI_API_KEY=your_api_key_here
```

2. Run a simple task:

```bash
APS solve "Create a simple calculator in Python"
```

3. The system will:
   - Break down the task into steps
   - Generate a solution
   - Verify the code works
   - Show you the result

## Configuration

1. View current configuration:

```bash
APS config show
```

2. Change settings:

```bash
APS config set temperature 0.8
APS config set model "gemini-pro"
```

## Advanced Usage

### 1. Multi-step Projects

For complex projects, APS will break down the task into manageable steps:

```bash
APS solve "Create a web application with user authentication and a database"
```

### 2. Streaming Output

Watch the solution being generated in real-time:

```bash
APS solve --stream "Explain how to implement binary search"
```

### 3. Custom Models and Parameters

```bash
APS solve --model "gemini-2.0-pro" --temperature 0.9 "Generate creative poetry"
```

## Direct API Usage

If you want to use the API programmatically instead of the CLI, you should use the hierarchical agent system directly:

```python
from src.agent.agent_types.architect import ArchitectAgent
from src.agent.state.base import InMemoryStateManager
from src.llm_providers.providers.gemini import GeminiProvider
from src.llm_providers.config.provider_config import GeminiConfig

# Set up the provider
provider_config = GeminiConfig(
    api_key="your_api_key_here",
    model="gemini-2.0-pro"
)
provider = GeminiProvider(config=provider_config)

# Create the state manager and architect agent
state_manager = InMemoryStateManager()
agent = ArchitectAgent(
    provider=provider,
    state_manager=state_manager
)

# Process a task
result = agent.process("Create a simple calculator in Python")
print(result.data)
```

## Best Practices

1. **Clear Requirements**

   ```bash
   # Good
   APS solve "Create a Python REST API with FastAPI, including user authentication and PostgreSQL database"

   # Less Clear
   APS solve "Make me an API"
   ```

2. **Use Context**

   ```bash
   # With context
   APS solve "Add unit tests for the user authentication module" --context ./src/auth/

   # Without context
   APS solve "Add tests"
   ```

3. **Iterative Development**

   ```bash
   # Step 1: Basic setup
   APS solve "Create a basic Flask application structure"

   # Step 2: Add features
   APS solve "Add user registration and login to the Flask app"

   # Step 3: Enhance
   APS solve "Add password reset functionality to the user system"
   ```

## Troubleshooting

1. If a solution isn't working:

```bash
APS verify "Check why the login system isn't working"
```

2. Get detailed logs:

```bash
APS solve "Create a React component" --verbose
```

3. Debug mode:

```bash
APS solve "Fix the API endpoint" --debug
```

## Next Steps

- Check out the [Getting Started](./getting_started.md) guide for a more detailed walkthrough
- Explore the [API Reference](../reference/api.md) for advanced usage
- Learn more about the [hierarchical agent system](../explanation/hierarchical_agents.md) architecture
