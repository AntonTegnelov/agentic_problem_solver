"""Script to fix TRY300 lint issues in planner.py."""

import re


def fix_try300() -> None:
    # Read the file
    with open("src/agent/agent_types/planner.py") as f:
        content = f.read()

    # Pattern to find the problematic code section
    pattern = r'(                if result\.success:\n                    return Result\.success\(\n                        data="Task delegated to sub-planner: Sub-planner processed task",\n                        message="Successfully delegated complex task to sub-planner",\n                    \)\n)                return result'

    # Replacement with else clause
    replacement = r"\1                else:\n                    return result"

    # Make the replacement
    modified_content = re.sub(pattern, replacement, content)

    # Write back to the file
    with open("src/agent/agent_types/planner.py", "w") as f:
        f.write(modified_content)


if __name__ == "__main__":
    fix_try300()
