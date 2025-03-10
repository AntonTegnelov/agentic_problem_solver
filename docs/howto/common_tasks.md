# Common Tasks and Use Cases

This guide provides step-by-step instructions for common use cases with the Agentic Problem Solver.

## Switching LLM Providers

To switch between different LLM providers:

1. Set the provider in your environment:

   ```bash
   # For Gemini (default)
   export PROVIDER=gemini
   # For future providers
   export PROVIDER=other_provider
   ```

2. Set the corresponding API key:
   ```bash
   export GEMINI_API_KEY=your_key_here
   # or for other providers
   export OTHER_PROVIDER_KEY=your_key_here
   ```

## Customizing Agent Behavior

### Adjusting Temperature

Control the agent's creativity vs determinism:

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

### Setting Token Limits

Control the length of generated responses:

```python
agent = SolverAgent(config={"max_tokens": 1000})
```

## Error Handling

### Handling Invalid Input

The agent automatically validates input and handles errors:

```python
from langchain_core.messages import HumanMessage

# Invalid input will be caught and handled
message = HumanMessage(content="")  # Empty input
response = await agent._process_message_impl(message)
print(response.content)  # Will contain error message
```

### Recovering from Errors

If an error occurs, you can:

1. Clear the agent's state:

   ```python
   agent.clear_state()
   ```

2. Try again with valid input:
   ```python
   message = HumanMessage(content="Write a valid Python function")
   response = await agent._process_message_impl(message)
   ```

## Logging and Debugging

### Setting Log Level

Control the verbosity of logs:

```python
import logging
from src.utils.logging import setup_logging

setup_logging(level=logging.DEBUG)  # For detailed logs
setup_logging(level=logging.INFO)   # For normal operation
```

### Logging to File

Save logs to a file for debugging:

```python
setup_logging(log_file="agent.log")
```

## Testing Generated Code

### Running Unit Tests

Test the generated code using pytest:

```python
# Save the generated code to a file
with open("generated_code.py", "w") as f:
    f.write(generated_code)

# Run tests
pytest test_generated_code.py
```

### Manual Testing

1. Copy the generated code to a Python file
2. Import and test the functionality:

   ```python
   from generated_code import generated_function

   result = generated_function(test_input)
   assert result == expected_output
   ```

## Best Practices

1. Always validate generated code before using in production
2. Keep tasks focused and specific
3. Provide clear requirements in your prompts
4. Use appropriate error handling
5. Monitor and log agent behavior
6. Test generated code thoroughly

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**

   - Verify the key is set correctly
   - Check for environment variable typos
   - Ensure the key has necessary permissions

2. **Generation Failures**

   - Check your internet connection
   - Verify API quotas and limits
   - Try adjusting the temperature or max tokens

3. **Invalid Generated Code**
   - Make task requirements more specific
   - Adjust the temperature for more deterministic output
   - Use the verification step to catch issues
