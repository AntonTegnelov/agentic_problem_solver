"""Tests for package configuration consistency."""

import re
from pathlib import Path

import tomli


def normalize_dependency(dep: str) -> str:
    """Normalize a dependency string to make comparison easier."""
    # Remove version specifiers and whitespace
    return re.sub(r"[<>=~!;].*$", "", dep).strip().lower()


def extract_pyproject_dependencies() -> set[str]:
    """Extract dependencies from pyproject.toml."""
    root_dir = Path(__file__).parent.parent.parent
    pyproject_path = root_dir / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml not found"

    with pyproject_path.open("rb") as f:
        pyproject_data = tomli.load(f)

    dependencies = set()

    # Get dependencies from project.dependencies
    if "project" in pyproject_data and "dependencies" in pyproject_data["project"]:
        dependencies.update(pyproject_data["project"]["dependencies"])

    # Get dependencies from tool.poetry.dependencies
    if (
        "tool" in pyproject_data
        and "poetry" in pyproject_data["tool"]
        and "dependencies" in pyproject_data["tool"]["poetry"]
    ):
        poetry_deps = pyproject_data["tool"]["poetry"]["dependencies"]
        # Poetry dependencies can be strings or dicts
        for name in poetry_deps:
            if name != "python":  # Skip python version requirement
                dependencies.add(name)

    return {normalize_dependency(dep) for dep in dependencies}


def extract_requirements_dependencies() -> set[str]:
    """Extract dependencies from requirements.txt."""
    root_dir = Path(__file__).parent.parent.parent
    requirements_path = root_dir / "requirements.txt"

    assert requirements_path.exists(), "requirements.txt not found"

    dependencies = set()
    with requirements_path.open(encoding="utf-8") as f:
        for line_raw in f:
            line = line_raw.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Skip options like --index-url
            if line.startswith("-"):
                continue
            # Remove any trailing comments
            line = line.split("#")[0].strip()
            dependencies.add(line)

    return {normalize_dependency(dep) for dep in dependencies}


def extract_setup_dependencies() -> set[str]:
    """Extract dependencies from setup.py."""
    root_dir = Path(__file__).parent.parent.parent
    setup_path = root_dir / "setup.py"

    assert setup_path.exists(), "setup.py not found"

    # This is a simple regex-based approach and might not catch all cases
    with setup_path.open(encoding="utf-8") as f:
        content = f.read()

    # Look for install_requires list
    match = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
    assert match, "Could not find install_requires in setup.py"

    deps_str = match.group(1)
    # Extract quoted strings
    deps = re.findall(r"['\"]([^'\"]+)['\"]", deps_str)

    return {normalize_dependency(dep) for dep in deps}


def test_package_config_files_exist() -> None:
    """Test that all package configuration files exist."""
    root_dir = Path(__file__).parent.parent.parent

    assert (root_dir / "pyproject.toml").exists(), "pyproject.toml not found"
    assert (root_dir / "setup.py").exists(), "setup.py not found"
    assert (root_dir / "requirements.txt").exists(), "requirements.txt not found"


def test_package_dependencies_consistency() -> None:
    """Test that dependencies are consistent across all package configuration files."""
    pyproject_deps = extract_pyproject_dependencies()
    requirements_deps = extract_requirements_dependencies()
    setup_deps = extract_setup_dependencies()

    # Check that all dependencies in pyproject.toml are in requirements.txt
    missing_in_requirements = pyproject_deps - requirements_deps
    assert not missing_in_requirements, (
        f"Dependencies in pyproject.toml missing from requirements.txt: {missing_in_requirements}"
    )

    # Check that all dependencies in pyproject.toml are in setup.py
    missing_in_setup = pyproject_deps - setup_deps
    assert not missing_in_setup, f"Dependencies in pyproject.toml missing from setup.py: {missing_in_setup}"

    # Check that all dependencies in setup.py are in pyproject.toml
    missing_in_pyproject = setup_deps - pyproject_deps
    assert not missing_in_pyproject, f"Dependencies in setup.py missing from pyproject.toml: {missing_in_pyproject}"

    # Check that all dependencies in setup.py are in requirements.txt
    missing_in_requirements_from_setup = setup_deps - requirements_deps
    assert not missing_in_requirements_from_setup, (
        f"Dependencies in setup.py missing from requirements.txt: {missing_in_requirements_from_setup}"
    )
