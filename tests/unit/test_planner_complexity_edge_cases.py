"""Unit tests for edge cases in planner agent's complexity analysis methods."""

from src.agent.agent_types.planner import PlannerAgent
from src.common_types.task_types import TaskComplexity


class TestPlannerComplexityEdgeCases:
    """Tests for edge cases in the PlannerAgent's complexity analysis methods."""

    def test_evaluate_subtask_complexity_empty_description(self) -> None:
        """Test complexity analysis with an empty task description."""
        planner = PlannerAgent()

        # Test with empty description
        complexity = planner.evaluate_subtask_complexity("")
        assert complexity == TaskComplexity.SIMPLE, "Empty description should be classified as SIMPLE"

    def test_evaluate_subtask_complexity_very_short_description(self) -> None:
        """Test complexity analysis with a very short task description."""
        planner = PlannerAgent()

        # Test with very short descriptions
        short_descriptions = [
            "Fix bug",
            "Add test",
            "Update docs",
            "Refactor",
        ]

        for desc in short_descriptions:
            complexity = planner.evaluate_subtask_complexity(desc)
            assert complexity == TaskComplexity.SIMPLE, f"Short description '{desc}' should be classified as SIMPLE"

    def test_evaluate_subtask_complexity_with_mixed_indicators(self) -> None:
        """Test complexity analysis with mixed complexity indicators."""
        planner = PlannerAgent()

        # Test with descriptions containing mixed indicators
        mixed_tasks = [
            "Create a simple component with complex integration requirements",
            "Implement a basic function that handles very complex edge cases",
            "Design a straightforward API with extremely comprehensive documentation",
        ]

        for task in mixed_tasks:
            complexity = planner.evaluate_subtask_complexity(task)
            # We're not asserting a specific complexity here, just making sure it runs
            assert complexity in list(TaskComplexity), f"Task '{task}' should be classified with a valid complexity"

    def test_evaluate_subtask_complexity_with_numeric_indicators(self) -> None:
        """Test complexity analysis with numeric indicators in the description."""
        planner = PlannerAgent()

        # Test with descriptions containing numeric indicators
        numeric_tasks = [
            "Create 10 unit tests for the authentication module",
            "Implement 5 API endpoints for user management",
            "Fix 3 critical bugs in the payment processing system",
        ]

        for task in numeric_tasks:
            complexity = planner.evaluate_subtask_complexity(task)
            # We're not asserting a specific complexity here, just making sure it runs
            assert complexity in list(TaskComplexity), f"Task '{task}' should be classified with a valid complexity"

    def test_evaluate_subtask_complexity_with_special_characters(self) -> None:
        """Test complexity analysis with special characters in the description."""
        planner = PlannerAgent()

        # Test with descriptions containing special characters
        special_char_tasks = [
            "Fix bug #123: User can't login with special characters in password",
            "Implement feature: 'Remember me' checkbox on login form",
            "Update README.md with installation instructions",
            "Create CI/CD pipeline for automatic deployment",
        ]

        for task in special_char_tasks:
            complexity = planner.evaluate_subtask_complexity(task)
            # We're not asserting a specific complexity here, just making sure it runs
            assert complexity in list(TaskComplexity), f"Task '{task}' should be classified with a valid complexity"

    def test_evaluate_subtask_complexity_with_code_snippets(self) -> None:
        """Test complexity analysis with code snippets in the description."""
        planner = PlannerAgent()

        # Test with descriptions containing code snippets
        code_snippet_tasks = [
            "Fix the bug in `authenticate()` function that causes login failures",
            "Update the SQL query: `SELECT * FROM users WHERE id = ?`",
            "Implement the following function: `def calculate_total(items): return sum(item.price for item in items)`",
        ]

        for task in code_snippet_tasks:
            complexity = planner.evaluate_subtask_complexity(task)
            # We're not asserting a specific complexity here, just making sure it runs
            assert complexity in list(TaskComplexity), f"Task '{task}' should be classified with a valid complexity"

    def test_evaluate_subtask_complexity_with_multiple_sentences(self) -> None:
        """Test complexity analysis with multiple sentences in the description."""
        planner = PlannerAgent()

        # Test with descriptions containing multiple sentences
        multi_sentence_tasks = [
            "Implement user authentication. Add login and registration forms. Integrate with the backend API.",
            "Fix the payment processing bug. Users are charged twice for the same order. This is a critical issue.",
            "Update the documentation. Add examples for all API endpoints. Include error handling scenarios.",
        ]

        for task in multi_sentence_tasks:
            complexity = planner.evaluate_subtask_complexity(task)
            # We're not asserting a specific complexity here, just making sure it runs
            assert complexity in list(TaskComplexity), f"Task '{task}' should be classified with a valid complexity"

    def test_evaluate_subtask_complexity_with_extreme_length(self) -> None:
        """Test complexity analysis with extremely long descriptions."""
        planner = PlannerAgent()

        # Test with an extremely long description
        long_description = "Implement a feature " + "that does something. " * 100
        complexity = planner.evaluate_subtask_complexity(long_description)
        # We're not asserting a specific complexity here, just making sure it runs
        assert complexity in list(TaskComplexity), "Long description should be classified with a valid complexity"
