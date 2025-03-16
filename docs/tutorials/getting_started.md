# Getting Started with Agentic Problem Solver

This tutorial will guide you through setting up and using the Agentic Problem Solver to solve your first programming task.

> See the [Hierarchical Agent System](../explanation/hierarchical_agents.md) documentation for more information on the agent architecture.

## Prerequisites

Before you begin, make sure you have:

1. Python 3.12 or later installed
2. A Google Gemini API key (you can get one at https://makersuite.google.com/app/apikey)
3. Git installed (optional, for cloning the repository)

## Installation

1. Clone the repository (or download and extract the ZIP):

   ```bash
   git clone https://github.com/yourusername/Agentic_problem_solver.git
   cd Agentic_problem_solver
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   # On Windows:
   set GEMINI_API_KEY=your_api_key_here
   # On Unix/MacOS:
   export GEMINI_API_KEY=your_api_key_here
   ```

## Your First Task

Let's solve a simple programming task using the agent. We'll create a function to calculate the factorial of a number.

1. Start the CLI:

   ```bash
   python -m src.cli.main solve "Write a function to calculate the factorial of a number in Python."
   ```

2. The agent will generate a solution, explaining the factorial function and providing the implementation.

## Using the Hierarchical Agent System Programmatically

You can use the agent system directly in your code:

```python
from src.agent.agent_types import create_architect_agent
from src.messages.creation import create_human_message
from src.llm_providers.factory import create_provider

# Create an LLM provider
provider = create_provider("gemini")  # or "openai", etc.

# Create an architect agent (top-level agent)
architect = create_architect_agent(provider=provider)

# Create a message with your task
message = create_human_message("Write a function to calculate the factorial of a number in Python.")

# Process the message (synchronously)
result = architect.process_sync(message)

# Print the result
print(result.data)
```

## Understanding the Agent Hierarchy

The system uses a hierarchical approach with three agent types:

1. **ArchitectAgent**: For high-level design and task decomposition
2. **PlannerAgent**: For detailed planning and task refinement
3. **ExecutorAgent**: For implementing specific tasks

For simple tasks, you can use an ExecutorAgent directly:

```python
from src.agent.agent_types import create_executor_agent
from src.messages.creation import create_human_message

executor = create_executor_agent(provider=provider)
message = create_human_message("Write a function to calculate the factorial of a number in Python.")
result = executor.process_sync(message)
print(result.data)
```

## Next Steps

- Try solving more complex programming tasks
- Explore the [API documentation](../reference/api.md) for advanced usage
- Learn about the [agent roles](../explanation/hierarchical_agents.md) in the system
- Check out the [examples](../../examples/) directory for more usage patterns

## Troubleshooting

- **API Key Issues**: Ensure your API key is set correctly in the environment variables
- **Dependency Errors**: Make sure all requirements are installed correctly
- **Provider Selection**: Try a different LLM provider if you encounter quality issues
