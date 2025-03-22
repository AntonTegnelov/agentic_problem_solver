"""Script to fix TRY300 lint issues in planner.py."""

import re
from pathlib import Path


def fix_try300() -> None:
    """Fix TRY300 linting issues in planner.py by adding else clauses after return statements in try blocks."""
    # Read the file
    planner_file = Path("src/agent/agent_types/planner.py")
    with planner_file.open(encoding="utf-8") as f:
        content = f.read()

    # Pattern to find the problematic code section
    pattern = (
        r"(                if result\.success:\n"
        r"                    return Result\.success\(\n"
        r'                        data="Task delegated to sub-planner: Sub-planner processed task",\n'
        r'                        message="Successfully delegated complex task to sub-planner",\n'
        r"                    \)\n)"
        r"                return result"
    )

    # Replacement with else clause
    replacement = r"\1                else:\n                    return result"

    # Make the replacement
    modified_content = re.sub(pattern, replacement, content)

    # Write back to the file
    with planner_file.open("w", encoding="utf-8") as f:
        f.write(modified_content)


if __name__ == "__main__":
    fix_try300()
