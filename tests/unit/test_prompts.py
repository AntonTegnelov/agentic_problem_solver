"""Tests for prompt templates."""

import pytest

from src.common_types.enums import AgentRole
from src.common_types.error_types import ConfigError
from src.prompts.templates import (
    ARCHITECTURAL_BREAKDOWN_PROMPT,
    EXECUTION_PROMPT,
    PLANNING_PROMPT,
    ROLE_PROMPTS,
    get_role_prompt,
)


class TestRolePrompts:
    """Test role-specific prompts."""

    def test_role_prompts_dictionary(self) -> None:
        """Test that the ROLE_PROMPTS dictionary contains all expected roles."""
        assert AgentRole.ARCHITECT in ROLE_PROMPTS
        assert AgentRole.PLANNER in ROLE_PROMPTS
        assert AgentRole.EXECUTOR in ROLE_PROMPTS
        assert ROLE_PROMPTS[AgentRole.ARCHITECT] == ARCHITECTURAL_BREAKDOWN_PROMPT
        assert ROLE_PROMPTS[AgentRole.PLANNER] == PLANNING_PROMPT
        assert ROLE_PROMPTS[AgentRole.EXECUTOR] == EXECUTION_PROMPT

    def test_get_role_prompt_architect(self) -> None:
        """Test get_role_prompt for ARCHITECT role."""
        prompt = get_role_prompt(
            AgentRole.ARCHITECT,
            task_description="Build a web application",
        )
        assert "ARCHITECT agent" in prompt
        assert "Build a web application" in prompt
        assert "architectural components" in prompt
        assert "JSON array" in prompt

    def test_get_role_prompt_planner(self) -> None:
        """Test get_role_prompt for PLANNER role."""
        prompt = get_role_prompt(
            AgentRole.PLANNER,
            component_description="User Authentication Module",
            component_purpose="Handle user login and registration",
            component_interfaces=["API", "Database"],
            component_complexity="complex",
            component_priority="high",
        )
        assert "PLANNER agent" in prompt
        assert "User Authentication Module" in prompt
        assert "Handle user login and registration" in prompt
        assert "API" in prompt
        assert "Database" in prompt
        assert "complex" in prompt
        assert "high" in prompt
        assert "implementation tasks" in prompt

    def test_get_role_prompt_executor(self) -> None:
        """Test get_role_prompt for EXECUTOR role."""
        prompt = get_role_prompt(
            AgentRole.EXECUTOR,
            task_description="Implement login form validation",
            acceptance_criteria=["Validate email format", "Check password strength"],
            task_complexity="simple",
            task_priority="medium",
        )
        assert "EXECUTOR agent" in prompt
        assert "Implement login form validation" in prompt
        assert "Validate email format" in prompt
        assert "Check password strength" in prompt
        assert "simple" in prompt
        assert "medium" in prompt
        assert "implementation" in prompt

    def test_get_role_prompt_with_context(self) -> None:
        """Test get_role_prompt with additional context."""
        prompt = get_role_prompt(
            AgentRole.ARCHITECT,
            task_description="Build a web application",
            context={
                "Framework": "React",
                "Database": "PostgreSQL",
                "Deployment": "Docker",
            },
        )
        assert "ARCHITECT agent" in prompt
        assert "Build a web application" in prompt
        assert "Additional Context" in prompt
        assert "Framework: React" in prompt
        assert "Database: PostgreSQL" in prompt
        assert "Deployment: Docker" in prompt

    def test_get_role_prompt_invalid_role(self) -> None:
        """Test get_role_prompt with invalid role."""
        with pytest.raises(ConfigError, match="Invalid role: AgentRole.SOLVER"):
            get_role_prompt(AgentRole.SOLVER, task_description="Test")
