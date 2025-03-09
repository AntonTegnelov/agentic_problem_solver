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
from src.agents.solver_agent import SolverAgent
from langchain_core.messages import HumanMessage

# Create an agent
agent = SolverAgent()

# Process a task
message = HumanMessage(content="Write a Python function to calculate factorial")
response = await agent._process_message_impl(message)
print(response.content)
```

## Documentation

- [Tutorials](docs/tutorials/): Step-by-step guides for beginners
- [How-To Guides](docs/howto/): Solutions to common problems
- [Reference](docs/reference/): API documentation
- [Explanation](docs/explanation/): Architecture and design decisions
- [Examples](docs/examples.py): Code examples with doctests

## Development Status

### Completed Features

- [x] LLM Provider System

  - Multiple provider support
  - Configuration management
  - Error handling
  - Unit and integration tests

- [x] Core Agent System

  - Task processing pipeline
  - State management
  - Logging system
  - Configuration options
  - Unit and integration tests

- [x] Documentation
  - Getting started guide
  - How-to guides
  - API reference
  - Architecture explanation
  - Code examples

### In Progress

- [ ] CI/CD Pipeline

  - Automated testing (pytest and doctest)
  - Test coverage monitoring (75% minimum)
  - Documentation completeness checks
  - Strict typing and linting (ruff)

- [ ] Task Breakdown System
  - Standardized TODO format
  - Task prioritization
  - Dependency tracking
  - Validation system

### Planned Features

- [ ] Multi-Agent System

  - Dynamic agent creation
  - Agent lifecycle management
  - Inter-agent communication
  - Resource management

- [ ] Task Delegation

  - Hierarchical problem solving
  - Progress tracking
  - Result aggregation
  - Error recovery

- [ ] Vector Database Integration

  - Codebase storage
  - RAG implementation
  - Query optimization
  - Automatic updates

- [ ] End-to-End Testing
  - Comprehensive test scenarios
  - Performance benchmarks
  - Reliability testing
  - Stress testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Run doctests: `python -m doctest docs/examples.py -v`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Status

This project is under active development. Check the [Issues](https://github.com/yourusername/Agentic_problem_solver/issues) page for current tasks and planned features.
