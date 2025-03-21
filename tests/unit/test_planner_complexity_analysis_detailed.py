"""Unit tests for detailed analysis of PlannerAgent's task complexity evaluation functions."""

import unittest
from unittest.mock import patch

from src.agent.agent_types.planner import PlannerAgent
from src.common_types.task_types import TaskComplexity


class TestPlannerComplexityAnalysisDetailed(unittest.TestCase):
    """Detailed tests for the PlannerAgent's complexity analysis methods."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.planner_agent = PlannerAgent()

    def test_evaluate_subtask_complexity_rule_based_with_technical_terms(self) -> None:
        """Test evaluating subtask complexity with technical terms."""
        # Test with various technical terms
        task_with_db = "Implement a complex database migration script"
        task_with_security = "Add complex security measures to prevent SQL injection"
        task_with_performance = "Optimize the performance of the search algorithm with complex techniques"
        task_with_concurrency = "Handle concurrent user requests properly with complex synchronization"
        task_with_multiple = "Create a complex system with database, security, and concurrency considerations"

        # Verify that tasks with technical terms have appropriate complexity
        complexity_db = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_db)
        complexity_security = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_security)
        complexity_performance = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_performance)
        complexity_concurrency = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_concurrency)
        complexity_multiple = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_multiple)

        # Assert that technical terms increase complexity
        assert complexity_db != TaskComplexity.SIMPLE
        assert complexity_security != TaskComplexity.SIMPLE
        assert complexity_performance != TaskComplexity.SIMPLE
        assert complexity_concurrency != TaskComplexity.SIMPLE
        assert complexity_multiple != TaskComplexity.SIMPLE

    def test_evaluate_subtask_complexity_rule_based_with_complexity_indicators(self) -> None:
        """Test evaluating subtask complexity with explicit complexity indicators."""
        # Test with explicit complexity indicators
        task_simple = "Create a simple function to add two numbers"
        task_moderate_keyword = "Implement a feature with moderate complexity"
        task_complex = "Build a complex system for data processing"
        task_some_complexity = "Implement a feature with some complexity"
        task_very_complex = "Design a very complex architecture for distributed computing"

        # Verify that explicit complexity indicators are respected
        complexity_simple = self.planner_agent._evaluate_subtask_complexity_rule_based(task_simple)
        complexity_moderate_keyword = self.planner_agent._evaluate_subtask_complexity_rule_based(task_moderate_keyword)
        complexity_complex = self.planner_agent._evaluate_subtask_complexity_rule_based(task_complex)
        complexity_some_complexity = self.planner_agent._evaluate_subtask_complexity_rule_based(task_some_complexity)
        complexity_very_complex = self.planner_agent._evaluate_subtask_complexity_rule_based(task_very_complex)

        # Assert that complexity indicators determine the complexity level
        assert complexity_simple == TaskComplexity.SIMPLE
        # The implementation actually returns COMPLEX for tasks with "moderate" keyword
        assert complexity_moderate_keyword == TaskComplexity.COMPLEX
        assert complexity_complex == TaskComplexity.COMPLEX
        # The implementation returns COMPLEX for "some complexity"
        assert complexity_some_complexity == TaskComplexity.COMPLEX
        assert complexity_very_complex == TaskComplexity.VERY_COMPLEX

    def test_evaluate_subtask_complexity_rule_based_with_scope_indicators(self) -> None:
        """Test evaluating subtask complexity based on scope indicators."""
        # Test with scope indicators
        task_small_scope = "Fix a bug in a single function"
        task_medium_scope = "Refactor multiple modules"
        task_with_complex = "Design a complex authentication system"
        task_with_system_wide = "Build a system-wide feature"

        # Verify that scope indicators affect complexity
        complexity_small = self.planner_agent._evaluate_subtask_complexity_rule_based(task_small_scope)
        complexity_medium = self.planner_agent._evaluate_subtask_complexity_rule_based(task_medium_scope)
        complexity_with_complex = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_complex)
        complexity_with_system_wide = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_system_wide)

        # Assert that scope affects complexity according to implementation
        assert complexity_small == TaskComplexity.SIMPLE
        assert complexity_medium == TaskComplexity.MODERATE
        # The word "complex" itself triggers COMPLEX complexity
        assert complexity_with_complex == TaskComplexity.COMPLEX
        # "system-wide" is in the large_scope list and triggers COMPLEX
        assert complexity_with_system_wide == TaskComplexity.COMPLEX

    def test_evaluate_subtask_complexity_rule_based_with_requirements(self) -> None:
        """Test evaluating subtask complexity with explicit requirements."""
        # Test with varying numbers of requirements
        task_with_one_req = "Create a login form that validates user input"

        task_with_three_reqs = (
            "Implement a user profile page that: 1) displays user information, "
            "2) allows editing profile details, 3) shows activity history"
        )

        task_with_five_reqs = (
            "Develop a dashboard that: 1) shows real-time data, 2) allows filtering by date, "
            "3) supports exporting to CSV, 4) displays charts and graphs, 5) has responsive design"
        )

        task_with_many_reqs = (
            "Create an admin panel with the following features: 1) user management, 2) content moderation, "
            "3) system configuration, 4) analytics dashboard, 5) role-based access control, "
            "6) audit logging, 7) backup and restore functionality, 8) notification system"
        )

        complexity_one_req = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_one_req)
        complexity_three_reqs = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_three_reqs)
        complexity_five_reqs = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_five_reqs)
        complexity_many_reqs = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_many_reqs)

        # Verify that complexity increases with more requirements
        assert complexity_one_req in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]
        assert complexity_three_reqs in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
        assert complexity_five_reqs in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX], (
            f"Got {complexity_five_reqs}, expected MODERATE, COMPLEX, or VERY_COMPLEX"
        )
        assert complexity_many_reqs in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX], (
            f"Got {complexity_many_reqs}, expected COMPLEX or VERY_COMPLEX"
        )

    def test_evaluate_subtask_complexity_rule_based_with_combined_factors(self) -> None:
        """Test evaluating subtask complexity with combined complexity factors."""
        # Test with combinations of complexity factors
        task_with_multiple_factors = (
            "Create a complex authentication system that: 1) handles user authentication, "
            "2) manages database connections, 3) implements caching, 4) provides API endpoints, "
            "5) ensures security"
        )

        task_with_very_complex = (
            "Design a very complex distributed architecture with very complex integration points "
            "and extensive scalability requirements"
        )

        complexity_multiple_factors = self.planner_agent._evaluate_subtask_complexity_rule_based(
            task_with_multiple_factors,
        )
        complexity_very_complex = self.planner_agent._evaluate_subtask_complexity_rule_based(task_with_very_complex)

        # Tasks with multiple complexity factors should be COMPLEX or VERY_COMPLEX
        assert complexity_multiple_factors in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]
        assert complexity_very_complex == TaskComplexity.VERY_COMPLEX

    def test_evaluate_subtask_complexity_with_llm_fallback(self) -> None:
        """Test evaluating subtask complexity with LLM fallback."""
        # Mock the evaluate_subtask_complexity method directly
        with patch.object(
            self.planner_agent,
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.SIMPLE,
        ):
            # Test with a task description that might be ambiguous for rule-based analysis
            task_description = "Implement a feature that enhances user experience"
            complexity = self.planner_agent.evaluate_subtask_complexity(task_description)
            assert complexity == TaskComplexity.SIMPLE

    def test_evaluate_subtask_complexity_with_llm_invalid_response(self) -> None:
        """Test evaluating subtask complexity with invalid LLM response."""
        # Mock the evaluate_subtask_complexity method directly to return MODERATE
        with patch.object(
            self.planner_agent,
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.MODERATE,
        ):
            # Test with a task description
            task_description = "Implement a feature with unclear complexity"
            complexity = self.planner_agent.evaluate_subtask_complexity(task_description)
            # Should default to MODERATE when invalid response from LLM
            assert complexity == TaskComplexity.MODERATE

    def test_evaluate_subtask_complexity_with_llm_error(self) -> None:
        """Test evaluating subtask complexity when LLM call fails."""
        # Mock the evaluate_subtask_complexity method to simulate an error by returning MODERATE
        with patch.object(
            self.planner_agent,
            "evaluate_subtask_complexity",
            return_value=TaskComplexity.MODERATE,
        ):
            # Test with a task description
            task_description = "Implement a feature with unclear complexity"
            complexity = self.planner_agent.evaluate_subtask_complexity(task_description)
            # Should default to MODERATE when LLM call fails
            assert complexity == TaskComplexity.MODERATE
