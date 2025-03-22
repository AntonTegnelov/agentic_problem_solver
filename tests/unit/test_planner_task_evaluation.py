"""Unit tests for PlannerAgent task evaluation features.

This module contains tests for the task evaluation capabilities of the PlannerAgent,
including prioritization, dependency analysis, and completion time estimation.
"""

from unittest.mock import patch

import pytest

from src.agent.agent_types.planner import PlannerAgent
from src.common_types.task_types import Task, TaskComplexity, TaskPriority


class TestPlannerTaskEvaluation:
    """Test class for PlannerAgent task evaluation methods."""

    @pytest.fixture
    def planner_agent(self) -> PlannerAgent:
        """Create a planner agent for testing."""
        return PlannerAgent()

    def test_evaluate_task_priority(self, planner_agent: PlannerAgent) -> None:
        """Test evaluating task priority based on description."""
        # Test high priority task
        high_priority_desc = "URGENT: Fix critical security vulnerability in authentication system"
        assert planner_agent.evaluate_task_priority(high_priority_desc) == TaskPriority.HIGH

        # Test medium priority task
        medium_priority_desc = "Implement new feature for user dashboard"
        assert planner_agent.evaluate_task_priority(medium_priority_desc) == TaskPriority.MEDIUM

        # Test low priority task
        low_priority_desc = "Minor UI improvement: adjust button padding"
        assert planner_agent.evaluate_task_priority(low_priority_desc) == TaskPriority.LOW

    def test_evaluate_task_priority_with_explicit_markers(self, planner_agent: PlannerAgent) -> None:
        """Test evaluating task priority with explicit priority markers."""
        # Test with explicit HIGH marker
        assert planner_agent.evaluate_task_priority("[HIGH] Update user profile page") == TaskPriority.HIGH
        assert planner_agent.evaluate_task_priority("(Priority: HIGH) Update user profile page") == TaskPriority.HIGH

        # Test with explicit MEDIUM marker
        assert planner_agent.evaluate_task_priority("[MEDIUM] Add form validation") == TaskPriority.MEDIUM
        assert planner_agent.evaluate_task_priority("(Priority: MEDIUM) Add form validation") == TaskPriority.MEDIUM

        # Test with explicit LOW marker
        assert planner_agent.evaluate_task_priority("[LOW] Update documentation") == TaskPriority.LOW
        assert planner_agent.evaluate_task_priority("(Priority: LOW) Update documentation") == TaskPriority.LOW

    def test_evaluate_task_priority_with_keywords(self, planner_agent: PlannerAgent) -> None:
        """Test evaluating task priority based on keywords in description."""
        # Test high priority keywords
        assert planner_agent.evaluate_task_priority("Critical bug in payment system") == TaskPriority.HIGH
        assert planner_agent.evaluate_task_priority("Urgent fix needed for login page") == TaskPriority.HIGH
        assert planner_agent.evaluate_task_priority("Security vulnerability in user data") == TaskPriority.HIGH

        # Test medium priority keywords
        assert planner_agent.evaluate_task_priority("Enhance performance of search function") == TaskPriority.MEDIUM
        assert planner_agent.evaluate_task_priority("Improve user experience on mobile") == TaskPriority.MEDIUM

        # Test low priority keywords
        assert planner_agent.evaluate_task_priority("Minor styling issue on about page") == TaskPriority.LOW
        assert planner_agent.evaluate_task_priority("Small typo in footer text") == TaskPriority.LOW

    def test_evaluate_task_priority_edge_cases(self, planner_agent: PlannerAgent) -> None:
        """Test evaluating task priority with edge cases."""
        # Empty description
        assert planner_agent.evaluate_task_priority("") == TaskPriority.MEDIUM

        # Very short description
        assert planner_agent.evaluate_task_priority("Fix bug") == TaskPriority.MEDIUM

        # Mixed priority signals
        mixed_desc = "Minor update to critical system component"
        # The mixed description contains both "minor" (low) and "critical" (high)
        # The implementation should prioritize one signal over the other
        priority = planner_agent.evaluate_task_priority(mixed_desc)
        # Just verify it's one of the valid priorities (not checking which one specifically)
        assert priority in (TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH)

    def test_analyze_task_dependencies(self, planner_agent: PlannerAgent) -> None:
        """Test analyzing task dependencies."""
        # Create test tasks
        tasks = [
            Task(task_id="task1", description="Setup database schema"),
            Task(task_id="task2", description="Implement user authentication"),
            Task(task_id="task3", description="Create user profile page"),
        ]

        # Mock the LLM call for dependency analysis
        with patch.object(
            planner_agent,
            "_get_llm_response",
            return_value={
                "dependencies": [
                    {"task_id": "task1", "dependent_task_ids": ["task2"]},
                    {"task_id": "task2", "dependent_task_ids": ["task3"]},
                ],
            },
        ):
            dependencies = planner_agent.analyze_task_dependencies(tasks)

            # Verify dependencies
            assert len(dependencies) == 2
            assert dependencies[0]["task_id"] == "task1"
            assert dependencies[0]["dependent_task_ids"] == ["task2"]
            assert dependencies[1]["task_id"] == "task2"
            assert dependencies[1]["dependent_task_ids"] == ["task3"]

    def test_analyze_task_dependencies_no_dependencies(self, planner_agent: PlannerAgent) -> None:
        """Test analyzing task dependencies when there are no dependencies."""
        # Create test tasks
        tasks = [
            Task(task_id="task1", description="Setup database schema"),
            Task(task_id="task2", description="Implement user authentication"),
        ]

        # Mock the LLM call for dependency analysis
        with patch.object(
            planner_agent,
            "_get_llm_response",
            return_value={"dependencies": []},
        ):
            dependencies = planner_agent.analyze_task_dependencies(tasks)

            # Verify no dependencies
            assert dependencies == []

    def test_analyze_task_dependencies_empty_tasks(self, planner_agent: PlannerAgent) -> None:
        """Test analyzing task dependencies with empty task list."""
        dependencies = planner_agent.analyze_task_dependencies([])
        assert len(dependencies) == 0

    def test_estimate_task_completion_time(self, planner_agent: PlannerAgent) -> None:
        """Test estimating task completion time."""
        # Test with different complexity levels
        simple_task = Task(description="Add a simple button to the UI", complexity=TaskComplexity.SIMPLE)
        moderate_task = Task(description="Implement form validation", complexity=TaskComplexity.MODERATE)
        complex_task = Task(description="Create authentication system", complexity=TaskComplexity.COMPLEX)
        very_complex_task = Task(
            description="Implement real-time collaboration feature",
            complexity=TaskComplexity.VERY_COMPLEX,
        )

        # Verify time estimates
        assert planner_agent.estimate_task_completion_time(simple_task) <= 60  # Simple tasks: up to 1 hour
        assert 60 <= planner_agent.estimate_task_completion_time(moderate_task) <= 240  # Moderate tasks: 1-4 hours
        assert 240 <= planner_agent.estimate_task_completion_time(complex_task) <= 480  # Complex tasks: 4-8 hours
        assert planner_agent.estimate_task_completion_time(very_complex_task) >= 480  # Very complex tasks: 8+ hours

    def test_estimate_task_completion_time_with_factors(self, planner_agent: PlannerAgent) -> None:
        """Test estimating task completion time with additional factors."""
        # Create tasks with different factors
        task_with_dependencies = Task(
            description="Implement user profile page",
            complexity=TaskComplexity.MODERATE,
            dependencies=[{"task_id": "auth_task", "description": "Implement authentication"}],
        )

        task_with_subtasks = Task(
            description="Create dashboard",
            complexity=TaskComplexity.COMPLEX,
            subtasks=[
                {"task_id": "subtask1", "description": "Create charts component"},
                {"task_id": "subtask2", "description": "Implement data filtering"},
            ],
        )

        # Verify that tasks with dependencies or subtasks have appropriate time estimates
        # The actual values may vary based on implementation, but we're testing that they return valid estimates
        assert planner_agent.estimate_task_completion_time(task_with_dependencies) > 0
        assert planner_agent.estimate_task_completion_time(task_with_subtasks) > 0
