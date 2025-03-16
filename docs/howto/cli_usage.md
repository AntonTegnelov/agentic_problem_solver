# CLI Usage Guide

This guide provides detailed information on how to use the Agentic Problem Solver CLI effectively.

## Basic Commands

### Solve Command

The `solve` command is the primary way to interact with the agent system. It takes a task description and returns a solution.

```bash
APS solve "Your task description here"
```

#### Options

- `--model`: Specify the model to use for generation (default: depends on configuration)
- `--temperature`: Set the temperature for generation (default: 0.7)
- `--max-tokens`: Set the maximum tokens to generate (default: 1000)
- `--verbose`: Enable verbose logging and show delegation messages
- `--debug`: Enable debug mode with maximum logging

#### Examples

Simple calculator:

```bash
APS solve "Make a simple calculator in Python"
```

Web application:

```bash
APS solve "Create a Flask web application with user authentication"
```

### Solution Output

By default, the `solve` command will only display the code portion of the solution, without any explanations, instructions, or internal delegation messages. This provides a clean output that you can directly use or copy.

Example default output:

```python
def add(x, y):
  """Adds two numbers."""
  return x + y

def subtract(x, y):
  """Subtracts two numbers."""
  return x - y

# ... rest of the calculator code
```

If you want to see the full solution including explanations, instructions, and internal delegation messages, use the `--verbose` flag:

```bash
APS solve --verbose "Make a simple calculator in Python"
```

Verbose output includes:

1. Delegation messages (showing how the task was processed)
2. A separator line
3. The complete solution content with explanations and code

Example verbose output:

````
2025-03-16 10:37:35,457 - agent.architect.architect_2425584351952 - INFO - Delegation decision: {"source_agent_id": "architect_2425584351952", "target_agent_id": "self", "task": "make a simple calculator in python", "reason": "Task complexity analysis: SIMPLE", "additional_info": {"task_complexity": "SIMPLE", "analysis_method": "rule_based", "decision_type": "complexity_analysis"}}
2025-03-16 10:37:35,458 - agent.architect.architect_2425584351952 - INFO - Delegation decision: {"source_agent_id": "architect_2425584351952", "target_agent_id": "executor_2425579146560", "task": "make a simple calculator in python", "reason": "Direct delegation to executor due to SIMPLE complexity", "additional_info": {"task_complexity": "SIMPLE"}}

--------------------------------------------------------------------------------

Here's a simple calculator in Python:

```python
def add(x, y):
  """Adds two numbers."""
  return x + y

def subtract(x, y):
  """Subtracts two numbers."""
  return x - y

# ... rest of the calculator code
````

Save this as calculator.py and run it with python calculator.py

````

## Troubleshooting

If you encounter issues with the CLI, try the following:

1. Use the `--verbose` flag to see more detailed logging and understand the internal delegation process:

```bash
APS solve "Your task" --verbose
````

2. Use the `--debug` flag for maximum logging:

```bash
APS solve "Your task" --debug
```

3. Check that your API keys are correctly set in your environment variables.

## Advanced Usage

### Customizing Model Parameters

You can customize the model parameters to get different results:

```bash
APS solve "Create a Python web scraper" --temperature 0.9 --max-tokens 2000
```

Higher temperature values (e.g., 0.9) will produce more creative but potentially less focused results, while lower values (e.g., 0.2) will produce more deterministic and focused results.

### Working with Complex Tasks

For complex tasks, the agent system will automatically break down the task into smaller components and delegate them appropriately. You can see this delegation process in the output when using the `--verbose` flag.

Example:

```bash
APS solve "Create a full-stack web application with React frontend and Django backend" --verbose
```

This will show how the architect agent breaks down the task and delegates to planner and executor agents.
