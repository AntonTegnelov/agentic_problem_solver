"""Prompt templates for agent steps."""

from typing import Dict, Optional

from src.config import AgentStep


def get_step_prompt(step: AgentStep, context: Optional[Dict] = None) -> str:
    """Get the prompt template for a given step.

    Args:
        step: The agent step
        context: Optional context variables

    Returns:
        The prompt template
    """
    context = context or {}
    prompts = {
        AgentStep.UNDERSTAND: "Analyze the task: {task}",
        AgentStep.PLAN: "Create a plan based on task analysis: {task_analysis}",
        AgentStep.EXECUTE: "Execute the plan: {plan}",
        AgentStep.VERIFY: "Verify the result: {result}",
        AgentStep.END: "Summarize the task completion: {task}",
    }
    return prompts[step].format(**context)
