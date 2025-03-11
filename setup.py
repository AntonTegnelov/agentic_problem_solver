"""Setup script for the Agentic Problem Solver package."""

from pathlib import Path

import toml
from setuptools import find_packages, setup

# Read the pyproject.toml file
pyproject_data = toml.load("pyproject.toml")
dependencies = pyproject_data.get("project", {}).get("dependencies", [])

with Path("README.md").open(encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agentic_problem_solver",
    version=pyproject_data.get("project", {}).get("version", "0.1.0"),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "python-dotenv",
        "langchain",
        "langchain-core",
        "google-generativeai",
        "pytest",
        "ruff",
    ],
    entry_points={
        "console_scripts": [
            "solve=src.cli.main:cli",
        ],
    },
    python_requires=pyproject_data.get("project", {}).get("requires-python", ">=3.8"),
    author=next((author.get("name") for author in pyproject_data.get("project", {}).get("authors", [])), ""),
    author_email=next((author.get("email") for author in pyproject_data.get("project", {}).get("authors", [])), ""),
    description=pyproject_data.get("project", {}).get("description", ""),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Agentic_problem_solver",
    classifiers=pyproject_data.get("project", {}).get("classifiers", []),
)
