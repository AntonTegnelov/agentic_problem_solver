#!/usr/bin/env python
"""Script to fix E501 line length issues in planner.py file by breaking long lines into multiple lines."""

import re
from pathlib import Path


def fix_e501() -> None:
    """Fix E501 linting issues (line length) in planner.py by breaking long lines into multiple lines."""
    # File path
    planner_file = Path("src/agent/agent_types/planner.py")

    # Read the content of the file
    with planner_file.open(encoding="utf-8") as f:
        content = f.read()

    # Find and fix long lines with f-string expressions
    pattern1 = (
        r'message=\(\s*\n\s*f"Failed to process task: "\s*\n\s*'
        r'f"(.*?)"\s*\n\s*\)'
    )
    replacement1 = (
        r"message=(\n"
        r'                            f"Failed to process task: "\n'
        r'                            f"\1"\n'
        r"                        )"
    )

    pattern2 = (
        r'message=\(\s*\n\s*f"Successfully processed task: "\s*\n\s*'
        r'f"(.*?)"\s*\n\s*\)'
    )
    replacement2 = (
        r"message=(\n"
        r'                            f"Successfully processed task: "\n'
        r'                            f"\1"\n'
        r"                        )"
    )

    # Make the replacements using regex
    content = re.sub(pattern1, replacement1, content)
    content = re.sub(pattern2, replacement2, content)

    # Write the updated content back to the file
    with planner_file.open("w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    fix_e501()
