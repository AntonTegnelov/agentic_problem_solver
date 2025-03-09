# Getting Started with Agentic Problem Solver

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
   python cli.py
   ```

2. Enter your task:

   ```
   Write a Python function to calculate the factorial of a number
   ```

3. Watch as the agent:

   - Analyzes the task requirements
   - Creates a plan
   - Implements the solution
   - Verifies the result

4. Review the generated solution and test it in your own code.

## Understanding the Output

The agent works through several steps:

1. **Understanding**: Breaks down the task requirements
2. **Planning**: Creates a step-by-step plan
3. **Execution**: Implements the solution
4. **Verification**: Tests and validates the solution

Each step's output is clearly marked, and you can follow the agent's reasoning process.

## Next Steps

Now that you've solved your first task, try:

1. More complex tasks like implementing classes
2. Tasks that require error handling
3. Tasks that involve multiple functions or modules

See the [How-To Guides](../howto/) for more specific use cases and advanced features.
