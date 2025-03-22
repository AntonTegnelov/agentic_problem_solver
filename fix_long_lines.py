#!/usr/bin/env python
"""Script to fix long lines in planner.py file by breaking them into multiple lines."""

from pathlib import Path

# File paths
planner_file = Path("src/agent/agent_types/planner.py")

# Read the content of the file
with planner_file.open(encoding="utf-8") as f:
    content = f.read()

# Create the search and replace patterns for success message
success_search = (
    'message=f"Successfully processed task: '
    "{task_obj.description if hasattr(task_obj, 'description') else str(task_obj)}\","
)
success_replace = (
    "message=(\n"
    '                                    f"Successfully processed task: "\n'
    '                                    f"{task_obj.description if hasattr(task_obj, '
    "'description') else str(task_obj)}\"\n"
    "                                ),"
)

# Create the search and replace patterns for failure message
failure_search = (
    'message=f"Failed to process task: '
    "{task_obj.description if hasattr(task_obj, 'description') else str(task_obj)}\","
)
failure_replace = (
    "message=(\n"
    '                                    f"Failed to process task: "\n'
    '                                    f"{task_obj.description if hasattr(task_obj, '
    "'description') else str(task_obj)}\"\n"
    "                                ),"
)

# Replace long lines with multi-line versions
content = content.replace(success_search, success_replace)
content = content.replace(failure_search, failure_replace)

# Write the updated content back to the file
with planner_file.open("w", encoding="utf-8") as f:
    f.write(content)
