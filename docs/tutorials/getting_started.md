# Getting Started with Agentic Problem Solver

> **⚠️ DEPRECATION NOTICE:**
>
> The underlying implementation of this CLI currently uses the deprecated `SolverAgent` class.
> In future versions, it will be updated to use the hierarchical agent system.
> The CLI interface will remain stable, but if you're using the API directly,
> you should migrate to using the hierarchical agent system.
>
> See the [Hierarchical Agent System](../explanation/hierarchical_agents.md) documentation for more information.

This tutorial will guide you through setting up and using the Agentic Problem Solver to solve your first programming task.

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

## Using the Hierarchical Agent System Directly

If you want to use the agent system programmatically, you should use the hierarchical agent system directly rather than the deprecated `SolverAgent`:

```python
import os
from src.agent.agent_types.architect import ArchitectAgent
from src.agent.state.base import InMemoryStateManager
from src.llm_providers.providers.gemini import GeminiProvider
from src.llm_providers.config.provider_config import GeminiConfig

# Set up the provider with your API key
api_key = os.environ.get("GEMINI_API_KEY")
provider_config = GeminiConfig(
    api_key=api_key,
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
result = agent.process("Write a function to calculate the factorial of a number in Python.")
print(result.data)
```

## Next Steps

- Explore more complex tasks using the hierarchical agent system
- Learn about the [agent roles](../explanation/hierarchical_agents.md) in the system
- Check out the [task breakdown](../howto/task_breakdown.md) capabilities
- See the [API Reference](../reference/api.md) for complete documentation
