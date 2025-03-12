"""Tests for prompt templates."""

import pytest

from src.common_types.enums import AgentRole
from src.common_types.error_types import ConfigError
from src.prompts.templates import (
    ARCHITECT_SYSTEM_DESIGN_PROMPT,
    ARCHITECTURAL_BREAKDOWN_PROMPT,
    EXECUTION_PROMPT,
    PLANNER_TASK_REFINEMENT_PROMPT,
    PLANNING_PROMPT,
    ROLE_PROMPTS,
    SPECIALIZED_ROLE_PROMPTS,
    get_role_prompt,
    get_specialized_role_prompt,
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


class TestSpecializedRolePrompts:
    """Test specialized role-specific prompts."""

    def test_specialized_role_prompts_dictionary(self) -> None:
        """Test that the SPECIALIZED_ROLE_PROMPTS dictionary contains expected entries."""
        assert AgentRole.ARCHITECT in SPECIALIZED_ROLE_PROMPTS
        assert "system_design" in SPECIALIZED_ROLE_PROMPTS[AgentRole.ARCHITECT]
        assert "breakdown" in SPECIALIZED_ROLE_PROMPTS[AgentRole.ARCHITECT]
        assert AgentRole.PLANNER in SPECIALIZED_ROLE_PROMPTS
        assert "task_refinement" in SPECIALIZED_ROLE_PROMPTS[AgentRole.PLANNER]
        assert "planning" in SPECIALIZED_ROLE_PROMPTS[AgentRole.PLANNER]
        assert SPECIALIZED_ROLE_PROMPTS[AgentRole.ARCHITECT]["system_design"] == ARCHITECT_SYSTEM_DESIGN_PROMPT
        assert SPECIALIZED_ROLE_PROMPTS[AgentRole.ARCHITECT]["breakdown"] == ARCHITECTURAL_BREAKDOWN_PROMPT
        assert SPECIALIZED_ROLE_PROMPTS[AgentRole.PLANNER]["task_refinement"] == PLANNER_TASK_REFINEMENT_PROMPT
        assert SPECIALIZED_ROLE_PROMPTS[AgentRole.PLANNER]["planning"] == PLANNING_PROMPT

    def test_get_specialized_role_prompt_architect_system_design(self) -> None:
        """Test get_specialized_role_prompt for ARCHITECT role with system_design type."""
        prompt = get_specialized_role_prompt(
            AgentRole.ARCHITECT,
            "system_design",
            task_description="Build a scalable e-commerce platform",
        )
        assert "ARCHITECT agent specializing in system design" in prompt
        assert "Build a scalable e-commerce platform" in prompt
        assert "System Overview" in prompt
        assert "Architectural Style and Patterns" in prompt
        assert "Component Breakdown" in prompt
        assert "Data Flow and Processing" in prompt
        assert "Non-Functional Requirements" in prompt
        assert "Technology Stack" in prompt
        assert "Implementation Roadmap" in prompt
        assert "JSON array" in prompt

    def test_get_specialized_role_prompt_architect_breakdown(self) -> None:
        """Test get_specialized_role_prompt for ARCHITECT role with breakdown type."""
        prompt = get_specialized_role_prompt(
            AgentRole.ARCHITECT,
            "breakdown",
            task_description="Build a user authentication system",
        )
        assert "ARCHITECT agent responsible for high-level system" in prompt
        assert "Build a user authentication system" in prompt
        assert "architectural components" in prompt
        assert "JSON array" in prompt

    def test_get_specialized_role_prompt_with_context(self) -> None:
        """Test get_specialized_role_prompt with additional context."""
        prompt = get_specialized_role_prompt(
            AgentRole.ARCHITECT,
            "system_design",
            task_description="Build a distributed database system",
            context={
                "Performance": "High throughput required",
                "Consistency": "Eventually consistent",
                "Availability": "99.99% uptime",
            },
        )
        assert "ARCHITECT agent specializing in system design" in prompt
        assert "Build a distributed database system" in prompt
        assert "Additional Context" in prompt
        assert "Performance: High throughput required" in prompt
        assert "Consistency: Eventually consistent" in prompt
        assert "Availability: 99.99% uptime" in prompt

    def test_get_specialized_role_prompt_invalid_role(self) -> None:
        """Test get_specialized_role_prompt with invalid role."""
        with pytest.raises(ConfigError, match="Invalid role for specialized prompt: AgentRole.SOLVER"):
            get_specialized_role_prompt(AgentRole.SOLVER, "system_design", task_description="Test")

    def test_get_specialized_role_prompt_invalid_prompt_type(self) -> None:
        """Test get_specialized_role_prompt with invalid prompt type."""
        with pytest.raises(ConfigError, match="Invalid prompt type 'invalid_type' for role AgentRole.ARCHITECT"):
            get_specialized_role_prompt(AgentRole.ARCHITECT, "invalid_type", task_description="Test")

    def test_get_specialized_role_prompt_planner_task_refinement(self) -> None:
        """Test get_specialized_role_prompt for PLANNER role with task_refinement type."""
        prompt = get_specialized_role_prompt(
            AgentRole.PLANNER,
            "task_refinement",
            component_description="User Authentication Module",
            component_purpose="Handle user login and registration",
            component_interfaces=["API", "Database"],
            component_complexity="complex",
            component_priority="high",
        )
        assert "PLANNER agent specializing in task refinement" in prompt
        assert "User Authentication Module" in prompt
        assert "Handle user login and registration" in prompt
        assert "API" in prompt
        assert "Database" in prompt
        assert "complex" in prompt
        assert "high" in prompt
        assert "Task Breakdown" in prompt
        assert "Task Sequencing" in prompt
        assert "Implementation Strategy" in prompt
        assert "Risk Assessment" in prompt
        assert "Acceptance Criteria" in prompt
        assert "Resource Planning" in prompt
        assert "JSON array" in prompt

    def test_get_specialized_role_prompt_planner_planning(self) -> None:
        """Test get_specialized_role_prompt for PLANNER role with planning type."""
        prompt = get_specialized_role_prompt(
            AgentRole.PLANNER,
            "planning",
            component_description="User Authentication Module",
            component_purpose="Handle user login and registration",
            component_interfaces=["API", "Database"],
            component_complexity="complex",
            component_priority="high",
        )
        assert "PLANNER agent responsible for mid-level task refinement" in prompt
        assert "User Authentication Module" in prompt
        assert "Handle user login and registration" in prompt
        assert "API" in prompt
        assert "Database" in prompt
        assert "complex" in prompt
        assert "high" in prompt
        assert "implementation tasks" in prompt
        assert "JSON array" in prompt
