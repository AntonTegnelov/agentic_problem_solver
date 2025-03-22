"""Script to fix complexity issues (C901) in planner.py by breaking up complex functions.

This script targets specifically the analyze_requirements function which has too many
branches/returns and splits it into smaller helper functions.
"""

from pathlib import Path


def fix_complexity() -> None:
    """Fix C901 complexity issues in planner.py by refactoring complex functions
    into smaller, more manageable helper functions.
    """
    planner_path = Path("src/agent/agent_types/planner.py")
    content = planner_path.read_text(encoding="utf-8")

    # Replace the complex analyze_requirements function with a refactored version
    # that uses helper functions to reduce complexity

    old_function = '''    def analyze_requirements(self, task_description: str) -> dict:
        """Extract requirements from a task description."""
        requirements = {}

        # Extract specific output format requirements
        if "JSON" in task_description or "json" in task_description:
            requirements["output_format"] = "JSON"
        elif "markdown" in task_description or "Markdown" in task_description:
            requirements["output_format"] = "markdown"
        elif "HTML" in task_description or "html" in task_description:
            requirements["output_format"] = "HTML"
        elif "CSV" in task_description or "csv" in task_description:
            requirements["output_format"] = "CSV"
        elif "XML" in task_description or "xml" in task_description:
            requirements["output_format"] = "XML"
        elif "YAML" in task_description or "yaml" in task_description:
            requirements["output_format"] = "YAML"
        else:
            requirements["output_format"] = "text"

        # Extract specific tool or technology requirements
        if "Python" in task_description or "python" in task_description:
            requirements["technologies"] = requirements.get("technologies", []) + ["Python"]
        if "JavaScript" in task_description or "javascript" in task_description or "JS" in task_description:
            requirements["technologies"] = requirements.get("technologies", []) + ["JavaScript"]
        if "TypeScript" in task_description or "typescript" in task_description or "TS" in task_description:
            requirements["technologies"] = requirements.get("technologies", []) + ["TypeScript"]
        if "React" in task_description or "react" in task_description:
            requirements["technologies"] = requirements.get("technologies", []) + ["React"]
        if "Node" in task_description or "node.js" in task_description or "NodeJS" in task_description:
            requirements["technologies"] = requirements.get("technologies", []) + ["Node.js"]
        if "SQL" in task_description or "sql" in task_description:
            requirements["technologies"] = requirements.get("technologies", []) + ["SQL"]
        if "API" in task_description or "api" in task_description or "REST" in task_description:
            requirements["technologies"] = requirements.get("technologies", []) + ["API"]

        # Extract specific output requirements
        if "step by step" in task_description.lower():
            requirements["output_style"] = "step_by_step"
        elif "concise" in task_description.lower() or "brief" in task_description.lower():
            requirements["output_style"] = "concise"
        elif "detailed" in task_description.lower() or "comprehensive" in task_description.lower():
            requirements["output_style"] = "detailed"
        else:
            requirements["output_style"] = "standard"

        return requirements'''

    new_function = '''    def _extract_output_format(self, task_description: str) -> str:
        """Extract specific output format requirements from task description."""
        if "JSON" in task_description or "json" in task_description:
            return "JSON"
        elif "markdown" in task_description or "Markdown" in task_description:
            return "markdown"
        elif "HTML" in task_description or "html" in task_description:
            return "HTML"
        elif "CSV" in task_description or "csv" in task_description:
            return "CSV"
        elif "XML" in task_description or "xml" in task_description:
            return "XML"
        elif "YAML" in task_description or "yaml" in task_description:
            return "YAML"
        else:
            return "text"

    def _extract_technologies(self, task_description: str) -> list:
        """Extract specific technology requirements from task description."""
        technologies = []
        tech_patterns = [
            ("Python", ["Python", "python"]),
            ("JavaScript", ["JavaScript", "javascript", "JS"]),
            ("TypeScript", ["TypeScript", "typescript", "TS"]),
            ("React", ["React", "react"]),
            ("Node.js", ["Node", "node.js", "NodeJS"]),
            ("SQL", ["SQL", "sql"]),
            ("API", ["API", "api", "REST"])
        ]

        for tech, patterns in tech_patterns:
            if any(pattern in task_description for pattern in patterns):
                technologies.append(tech)

        return technologies

    def _extract_output_style(self, task_description: str) -> str:
        """Extract specific output style requirements from task description."""
        task_lower = task_description.lower()

        if "step by step" in task_lower:
            return "step_by_step"
        elif "concise" in task_lower or "brief" in task_lower:
            return "concise"
        elif "detailed" in task_lower or "comprehensive" in task_lower:
            return "detailed"
        else:
            return "standard"

    def analyze_requirements(self, task_description: str) -> dict:
        """Extract requirements from a task description."""
        requirements = {}

        # Extract specific requirements using helper functions
        requirements["output_format"] = self._extract_output_format(task_description)
        technologies = self._extract_technologies(task_description)
        if technologies:
            requirements["technologies"] = technologies
        requirements["output_style"] = self._extract_output_style(task_description)

        return requirements'''

    updated_content = content.replace(old_function, new_function)

    # Write the updated content back to the file
    planner_path.write_text(updated_content, encoding="utf-8")


if __name__ == "__main__":
    fix_complexity()
