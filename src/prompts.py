"""Prompt templates for the agent."""

from src.validation import ProcessingStep

prompts = {
    "understand": (
        "Please analyze the following task and identify key requirements:\n{task}"
    ),
    "plan": (
        "Based on the task:\n{task}\nCreate a step-by-step plan to solve this task."
    ),
    "implement": (
        "Based on the plan:\n{task}\nProvide the implementation details and results."
    ),
    "verify": (
        "Verify the following result:\n{result}\n"
        "Verify if it meets the requirements and suggest improvements if needed."
    ),
    "end": "Task completed. Final summary for:\n{task}",
}


def get_step_prompt(step: ProcessingStep, context: dict[str, str] | None = None) -> str:
    """Get the prompt template for a given step.

    Args:
        step: The current step in the agent's workflow
        context: Optional context variables for the prompt template

    Returns:
        Formatted prompt string
    """
    if not context:
        context = {}
    return prompts[step.value].format(**context)
