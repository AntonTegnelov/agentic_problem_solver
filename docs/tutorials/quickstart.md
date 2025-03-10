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
APS solve "Create a React todo app with TypeScript and deploy it to Vercel"
```

The system will:

1. Create a project plan
2. Set up the development environment
3. Implement each component
4. Add testing
5. Handle deployment

### 2. Using Different Providers

Switch between different LLM providers:

```bash
# Use OpenAI
APS config set provider openai
export OPENAI_API_KEY=your_key_here

# Use Anthropic
APS config set provider anthropic
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Customizing Behavior

Control how APS approaches tasks:

```bash
# More creative solutions
APS config set temperature 0.9

# More focused solutions
APS config set temperature 0.1

# Longer responses
APS config set max_tokens 2000
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

- Read the [How-To Guides](../howto/) for detailed instructions
- Check the [Reference](../reference/) for complete API documentation
- See [Examples](../examples.py) for more use cases
- Read [Explanations](../explanation/) for architecture details
