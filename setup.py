"""Setup script for the Agentic Problem Solver package."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agentic_problem_solver",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click>=8.0.0",
        "python-dotenv",
        "langchain>=0.1.0",
        "langchain-core",
        "google-generativeai>=0.3.0",
        "pytest",
        "ruff",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "solve=src.cli.main:cli",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered problem-solving system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Agentic_problem_solver",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
