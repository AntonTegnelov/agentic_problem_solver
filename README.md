#Agentic Problem Solver

A powerful AI-powered problem-solving system that breaks down complex programming tasks into manageable steps and generates solutions.

## Features

- Intelligent task breakdown and analysis
- Step-by-step solution generation
- Multiple LLM provider support
- Comprehensive error handling
- Extensive logging and debugging capabilities
- Modular and extensible architecture

## Installation

1. Clone the repository:

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

3. Install the package in development mode:

   ```bash
   pip install -e .
   ```

4. Set up environment variables:

   ```bash
   # Required:
   export GEMINI_API_KEY=your_api_key_here

   # Optional:
   export PROVIDER=gemini  # Default LLM provider
   export LOG_LEVEL=INFO  # Logging level
   export LOG_FILE=agent.log  # Log file path
   ```

## Usage

### Command Line Interface

Use the `APS` command to solve problems:

```bash
# Basic usage
APS solve "Create a calculator app in TypeScript"

# With options
APS solve --temperature 0.7 --max-tokens 2000 "Write a Python web scraper"

# Stream output
APS solve --stream "Create a REST API in FastAPI"

# Show version
APS version
```

Available options:

- `--temperature, -t`: Control creativity (0.0 to 1.0)
- `--max-tokens, -m`: Maximum tokens to generate
- `--stream, -s`: Stream output as it's generated

### Python API

```python
from src.agent.solver import SolverAgent
from src.llm_providers.providers.gemini import GeminiProvider

# Create provider and agent
provider = GeminiProvider()
agent = SolverAgent(provider=provider)

# Process a task
result = agent.process("Create a simple calculator in Python")
print(result)
```

## Documentation

- [Tutorials](docs/tutorials/): Step-by-step guides for beginners
- [How-To Guides](docs/howto/): Solutions to common problems
- [Reference](docs/reference/): API documentation
- [Explanation](docs/explanation/): Architecture and design decisions
- [Examples](docs/examples.py): Code examples with doctests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Status

This project is under active development. Check the [Issues](https://github.com/yourusername/Agentic_problem_solver/issues) page for current tasks and planned features.
