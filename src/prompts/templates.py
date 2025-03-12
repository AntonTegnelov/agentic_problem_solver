"""Prompt templates for the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.common_types.enums import AgentRole, AgentStep
from src.common_types.error_types import ConfigError
from src.messages.creation import create_human_message

if TYPE_CHECKING:
    from src.agent.state.base import AgentState
    from src.common_types.result_types import Result as StepResult

# Constants for validation
MIN_UNDERSTANDING_LENGTH = 100  # Minimum length for understanding step
MIN_PLAN_LENGTH = 100  # Minimum length for plan step
MIN_VERIFY_LENGTH = 100

# System prompts
SYSTEM_PROMPT = """You are an AI agent tasked with solving programming problems.
Your goal is to understand the problem, create a plan, and implement a solution.
Follow best practices and provide clear explanations."""

# Step prompts
UNDERSTAND_PROMPT = """You are tasked with understanding and breaking down a
programming task.

Please analyze the task and provide:
1. A clear problem statement
2. Key requirements and constraints
3. Any assumptions or clarifications needed
4. Initial thoughts on potential approaches
5. Identify any dependencies or prerequisites
6. Break down complex requirements into simpler components
7. Highlight any potential challenges or edge cases
8. Consider the context and scope of the problem

Task: {task}
{context}
"""

PLAN_PROMPT = """Based on our understanding of the task, create a detailed plan:
1. Break down the solution into clear steps
2. Identify potential challenges
3. List required resources or dependencies
4. Consider alternative approaches
5. Outline testing strategy
6. Prioritize tasks in a logical sequence
7. Estimate complexity of each component
8. Define interfaces between components
9. Consider error handling and edge cases
10. Plan for extensibility and maintainability

Understanding:
{understanding}

Task: {task}
{context}
"""

EXECUTE_PROMPT = """Now let's implement the solution according to our plan:
1. Write clean, well-documented code
2. Follow best practices and patterns
3. Consider error handling
4. Add appropriate comments
5. Ensure maintainability
6. Implement each component according to the plan
7. Follow consistent coding style
8. Use appropriate data structures and algorithms
9. Optimize for readability and performance
10. Include logging and debugging information

Plan:
{plan}

Task: {task}
{context}
"""

VERIFY_PROMPT = """Let's verify our implementation:
1. Review the code for correctness
2. Check against requirements
3. Test edge cases
4. Identify potential improvements
5. Document any limitations
6. Verify error handling
7. Check for performance issues
8. Ensure all requirements are met
9. Validate against test cases
10. Consider security implications

Implementation:
{implementation}

Task: {task}
{context}
"""

RETRY_PROMPT = """The previous attempt at the {step} step encountered an issue:

Error: {error}

Let's try again with a different approach:
1. Review what went wrong
2. Consider alternative strategies
3. Address the specific error
4. Be more careful about edge cases
5. Provide a more robust solution

Previous attempt:
{previous_result}

Task: {task}
{context}
"""

# New hierarchical agent prompts

# Architectural breakdown prompt for the Architect agent
ARCHITECTURAL_BREAKDOWN_PROMPT = """You are an ARCHITECT agent responsible for high-level system \
design and task decomposition.

Task Description: {task_description}

Please analyze this task and break it down into major architectural components following these guidelines:

1. Identify the key components or subsystems needed
2. Define clear interfaces between components
3. Consider system-level design patterns and principles
4. Establish data flow between components
5. Identify potential technical challenges
6. Consider scalability, maintainability, and extensibility
7. Determine appropriate technologies and frameworks
8. Establish naming conventions and architectural standards

For each component, provide:
- A clear description
- Its purpose and responsibilities
- Key interfaces with other components
- Estimated complexity (simple, moderate, complex, very_complex)
- Priority (low, medium, high, critical)
- Any dependencies on other components

Format your response as a JSON array of components with the following structure:

```json
[
  {{
    "description": "Component description",
    "purpose": "Component purpose and responsibilities",
    "interfaces": ["Interface 1", "Interface 2"],
    "complexity": "simple|moderate|complex|very_complex",
    "priority": "low|medium|high|critical",
    "dependencies": [
      {{
        "component_index": 0,
        "description": "Dependency description",
        "is_blocking": true|false
      }}
    ]
  }},
  // Additional components...
]
```

Ensure that the dependencies reference other components in the list by their index (0-based).
{context}
"""

# Planning prompt for the Planner agent
PLANNING_PROMPT = """You are a PLANNER agent responsible for mid-level task refinement and planning.

Component to Implement: {component_description}
Component Purpose: {component_purpose}
Component Interfaces: {component_interfaces}
Component Complexity: {component_complexity}
Component Priority: {component_priority}

Please break down this component into specific implementation tasks following these guidelines:

1. Create a logical sequence of development tasks
2. Define clear acceptance criteria for each task
3. Identify dependencies between tasks
4. Estimate complexity for each task
5. Assign appropriate priority to each task
6. Consider testing requirements
7. Plan for error handling and edge cases
8. Include documentation tasks

For each task, provide:
- A clear description
- Specific acceptance criteria
- Estimated complexity (simple, moderate, complex, very_complex)
- Priority (low, medium, high, critical)
- Any dependencies on other tasks

Format your response as a JSON array of tasks with the following structure:

```json
[
  {{
    "description": "Task description",
    "acceptance_criteria": ["Criterion 1", "Criterion 2"],
    "complexity": "simple|moderate|complex|very_complex",
    "priority": "low|medium|high|critical",
    "dependencies": [
      {{
        "task_index": 0,
        "description": "Dependency description",
        "is_blocking": true|false
      }}
    ]
  }},
  // Additional tasks...
]
```

Ensure that the dependencies reference other tasks in the list by their index (0-based).
{context}
"""

# Execution prompt for the Executor agent
EXECUTION_PROMPT = """You are an EXECUTOR agent responsible for low-level task implementation.

Task to Implement: {task_description}
Acceptance Criteria: {acceptance_criteria}
Task Complexity: {task_complexity}
Task Priority: {task_priority}

Please implement this task following these guidelines:

1. Write clean, well-documented code
2. Follow best practices and patterns
3. Implement proper error handling
4. Add appropriate comments
5. Ensure the code meets all acceptance criteria
6. Follow consistent coding style
7. Use appropriate data structures and algorithms
8. Optimize for readability and performance
9. Include unit tests where appropriate
10. Document any assumptions or limitations

Your implementation should be complete and ready for review.

{context}
"""

# Role-specific prompts dictionary
ROLE_PROMPTS = {
    AgentRole.ARCHITECT: ARCHITECTURAL_BREAKDOWN_PROMPT,
    AgentRole.PLANNER: PLANNING_PROMPT,
    AgentRole.EXECUTOR: EXECUTION_PROMPT,
}

STEP_PROMPTS = {
    AgentStep.UNDERSTAND: UNDERSTAND_PROMPT,
    AgentStep.PLAN: PLAN_PROMPT,
    AgentStep.EXECUTE: EXECUTE_PROMPT,
    AgentStep.VERIFY: VERIFY_PROMPT,
}


def get_step_prompt(state: AgentState) -> str:
    """Get prompt for current step.

    Args:
        state: Current agent state.

    Returns:
        Prompt for current step.

    Raises:
        ConfigError: If step is invalid.

    """
    step = state.current_step

    # Validate step
    if step not in STEP_PROMPTS:
        msg = f"Invalid step: {step}"
        raise ConfigError(msg)

    prompt = STEP_PROMPTS[step]

    # Get context for prompt
    task = state.get_context("task", "")
    understanding = state.get_context("understanding", "")
    plan = state.get_context("plan", "")
    implementation = state.get_context("implementation", "")

    # Get additional context
    context_data = state.get_context("additional_context", {})
    context_str = ""
    if context_data:
        context_str = "\nAdditional Context:\n"
        for key, value in context_data.items():
            context_str += f"- {key}: {value}\n"

    return prompt.format(
        task=task,
        understanding=understanding,
        plan=plan,
        implementation=implementation,
        context=context_str,
    )


def get_retry_prompt(state: AgentState, error: str) -> str:
    """Get retry prompt for current step.

    Args:
        state: Current agent state.
        error: Error message.

    Returns:
        Retry prompt for current step.

    Raises:
        ConfigError: If step is invalid.

    """
    step = state.current_step

    # Validate step
    if step not in AgentStep:
        msg = f"Invalid step: {step}"
        raise ConfigError(msg)

    # Get context for prompt
    task = state.get_context("task", "")
    previous_result = state.get_step_result(step)
    previous_result_str = ""
    if previous_result:
        previous_result_str = str(previous_result.data)

    # Get additional context
    context_data = state.get_context("additional_context", {})
    context_str = ""
    if context_data:
        context_str = "\nAdditional Context:\n"
        for key, value in context_data.items():
            context_str += f"- {key}: {value}\n"

    return RETRY_PROMPT.format(
        step=step.value,
        error=error,
        previous_result=previous_result_str,
        task=task,
        context=context_str,
    )


def get_role_prompt(role: AgentRole, **kwargs: dict[str, Any]) -> str:
    """Get prompt for specific agent role.

    Args:
        role: Agent role.
        **kwargs: Additional context for the prompt.

    Returns:
        Role-specific prompt.

    Raises:
        ConfigError: If role is invalid.

    """
    # Validate role
    if role not in ROLE_PROMPTS:
        msg = f"Invalid role: {role}"
        raise ConfigError(msg)

    prompt = ROLE_PROMPTS[role]

    # Get additional context
    context_data = kwargs.pop("context", {})
    context_str = ""
    if context_data:
        context_str = "\nAdditional Context:\n"
        for key, value in context_data.items():
            context_str += f"- {key}: {value}\n"

    # Format the prompt with provided kwargs and context
    return prompt.format(context=context_str, **kwargs)


def validate_step_result(
    step: AgentStep,
    result: StepResult[Any],
) -> None:
    """Validate step result.

    Args:
        step: Step to validate.
        result: Result to validate.

    Raises:
        ConfigError: If result is invalid.

    """
    # Check for failed result
    if not result.success:
        msg = f"Step failed: {result.error}"
        raise ConfigError(msg)

    # Check for empty result
    if not result.data:
        msg = "Empty result"
        raise ConfigError(msg)

    # Validate step-specific requirements
    if step == AgentStep.UNDERSTAND:
        if len(str(result.data)) < MIN_UNDERSTANDING_LENGTH:
            msg = "Understanding is too brief"
            raise ConfigError(msg)
    elif step == AgentStep.PLAN:
        if len(str(result.data)) < MIN_PLAN_LENGTH:
            msg = "Plan is too brief"
            raise ConfigError(msg)
    elif step == AgentStep.VERIFY:
        if not isinstance(result.data, bool):
            msg = "Verification result must be boolean"
            raise ConfigError(msg)
    elif step == AgentStep.EXECUTE and not result.data:
        msg = "Missing execution result"
        raise ConfigError(msg)


def execute_step(state: AgentState, step: AgentStep, prompt: str) -> StepResult:
    """Execute a step with the given prompt.

    Args:
        state: Current agent state
        step: Step to execute
        prompt: Prompt to use for execution

    Returns:
        StepResult containing the execution result

    """
    # Get agent for step
    agent = state.get_agent_for_step(step)
    if not agent:
        msg = f"No agent configured for step {step.value}"
        raise ConfigError(msg)

    # Create message with prompt
    message = create_human_message(prompt)

    # Process with agent
    return agent.process(message)


def _try_execute_step_once(
    state: AgentState,
    step: AgentStep,
    prompt: str,
) -> StepResult:
    """Try to execute a step once.

    Args:
        state: Current agent state.
        step: Step to execute.
        prompt: Prompt to use.

    Returns:
        Step result.

    Raises:
        ConfigError: If step execution fails.
        ValueError: If step execution fails.
        AttributeError: If step execution fails.

    """
    # Execute step
    result = execute_step(state, step, prompt)

    # Record result
    state.record_step_result(step, result)

    return result


def _try_execute_step_once_with_retry(
    state: AgentState,
    step: AgentStep,
    prompt: str,
    retries: int,
    max_retries: int,
) -> tuple[bool, StepResult | None, Exception | None]:
    """Try to execute a step once with retry handling.

    Args:
        state: Current agent state.
        step: Step to execute.
        prompt: Prompt to use.
        retries: Current retry count.
        max_retries: Maximum number of retries.

    Returns:
        Tuple of (success, result, error).

    """
    try:
        result = _try_execute_step_once(state, step, prompt)
    except ConfigError as e:
        if retries > max_retries:
            error_msg = f"Max retries exceeded ({max_retries}) for step {step.value}. Last error: {e}"
            raise ConfigError(error_msg) from e
        return False, None, e
    except (ValueError, AttributeError) as e:
        error_msg = f"Error executing step {step.value}: {e}"
        raise ConfigError(error_msg) from e
    else:
        return True, result, None


def execute_step_with_retry(
    state: AgentState,
    step: AgentStep,
    max_retries: int = 3,
) -> StepResult:
    """Execute step with retry on failure.

    Args:
        state: Current agent state.
        step: Step to execute.
        max_retries: Maximum number of retries.

    Returns:
        Step result.

    Raises:
        ConfigError: If step execution fails.

    """
    last_error = None
    retries = 0

    while retries <= max_retries:
        # Get step prompt
        prompt = get_step_prompt(state) if retries == 0 else get_retry_prompt(state, str(last_error))

        # Get agent for step
        agent = state.get_agent_for_step(step)
        if not agent:
            msg = f"No agent configured for step {step.value}"
            raise ConfigError(msg)

        # Create message with prompt
        message = create_human_message(prompt)

        # Process with agent
        result = agent.process(message)

        # If successful, return the result
        if result.success:
            return result

        # Update retry state
        retries += 1
        last_error = result.error

    # Return the last result after all retries are exhausted
    return result


def get_next_step(current_step: AgentStep) -> AgentStep:
    """Get next step in sequence.

    Args:
        current_step: Current step.

    Returns:
        Next step.

    Raises:
        ConfigError: If current step is invalid.

    """
    if current_step == AgentStep.UNDERSTAND:
        return AgentStep.PLAN
    if current_step == AgentStep.PLAN:
        return AgentStep.EXECUTE
    if current_step == AgentStep.EXECUTE:
        return AgentStep.VERIFY
    if current_step == AgentStep.VERIFY:
        return AgentStep.UNDERSTAND
    msg = f"Invalid step: {current_step}"
    raise ConfigError(msg)


def get_step_description(step: AgentStep) -> str:
    """Get description for step.

    Args:
        step: Step to describe.

    Returns:
        Step description.

    Raises:
        ConfigError: If step is invalid.

    """
    if step == AgentStep.UNDERSTAND:
        return "Analyzing and comprehending the task"
    if step == AgentStep.PLAN:
        return "Creating a strategy to solve the task"
    if step == AgentStep.EXECUTE:
        return "Implementing the planned solution"
    if step == AgentStep.VERIFY:
        return "Testing and validating the solution"
    msg = f"Invalid step: {step}"
    raise ConfigError(msg)
