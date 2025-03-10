"""Prompt templates for the agent."""

from src.agent.state.base import AgentState
from src.common_types.enums import AgentStep

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

Task: {task}
"""

PLAN_PROMPT = """Based on our understanding of the task, create a detailed plan:
1. Break down the solution into clear steps
2. Identify potential challenges
3. List required resources or dependencies
4. Consider alternative approaches
5. Outline testing strategy

Understanding:
{understanding}

Task: {task}
"""

EXECUTE_PROMPT = """Now let's implement the solution according to our plan:
1. Write clean, well-documented code
2. Follow best practices and patterns
3. Consider error handling
4. Add appropriate comments
5. Ensure maintainability

Plan:
{plan}

Task: {task}
"""

VERIFY_PROMPT = """Let's verify our implementation:
1. Review the code for correctness
2. Check against requirements
3. Test edge cases
4. Identify potential improvements
5. Document any limitations

Implementation:
{implementation}

Task: {task}
"""

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

    """
    step = state.current_step
    prompt = STEP_PROMPTS[step]

    # Get context for prompt
    task = state.get_context("task", "")
    understanding = state.get_context("understanding", "")
    plan = state.get_context("plan", "")
    implementation = state.get_context("implementation", "")

    return prompt.format(
        task=task,
        understanding=understanding,
        plan=plan,
        implementation=implementation,
    )
