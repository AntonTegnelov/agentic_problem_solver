"""Tests for environment file consistency."""

from pathlib import Path

from dotenv import dotenv_values


def test_env_files_exist() -> None:
    """Test that both .env and .env.example files exist."""
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / ".env"
    env_example_path = root_dir / ".env.example"

    assert env_path.exists(), ".env file not found"
    assert env_example_path.exists(), ".env.example file not found"


def test_env_files_have_matching_keys() -> None:
    """Test that .env and .env.example have matching keys."""
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / ".env"
    env_example_path = root_dir / ".env.example"

    env_vars = dotenv_values(env_path)
    env_example_vars = dotenv_values(env_example_path)

    # Check for keys in .env that are not in .env.example
    missing_in_example = set(env_vars.keys()) - set(env_example_vars.keys())
    assert not missing_in_example, f"Keys in .env missing from .env.example: {missing_in_example}"

    # Check for keys in .env.example that are not in .env
    missing_in_env = set(env_example_vars.keys()) - set(env_vars.keys())
    assert not missing_in_env, f"Keys in .env.example missing from .env: {missing_in_env}"


def test_env_example_has_no_empty_values() -> None:
    """Test that .env.example has no empty values."""
    root_dir = Path(__file__).parent.parent.parent
    env_example_path = root_dir / ".env.example"

    env_example_vars = dotenv_values(env_example_path)

    empty_values = {k for k, v in env_example_vars.items() if not v}
    assert not empty_values, f"Empty values in .env.example for keys: {empty_values}"
