"""Configuration utilities."""

from pathlib import Path
from typing import Any

from src.exceptions import ConfigError


def load_env_var(key: str, env_file: str | Path = ".env") -> str:
    """Load environment variable from .env file.

    Args:
        key: Environment variable key.
        env_file: Path to .env file.

    Returns:
        Environment variable value.

    Raises:
        ConfigError: If .env file is not found or key is not found.

    """
    env_path = Path(env_file)
    if not env_path.exists():
        msg = f"No {env_file} file found"
        raise ConfigError(msg)

    with env_path.open() as f:
        for line in f:
            if line.startswith(f"{key}="):
                return line.split("=")[1].strip()

    msg = f"{key} not found in {env_file}"
    raise ConfigError(msg)


def load_config_from_env(
    keys: list[str],
    env_file: str | Path = ".env",
) -> dict[str, Any]:
    """Load multiple environment variables from .env file.

    Args:
        keys: List of environment variable keys.
        env_file: Path to .env file.

    Returns:
        Dictionary of environment variables.

    Raises:
        ConfigError: If .env file is not found or any key is not found.

    """
    return {key: load_env_var(key, env_file) for key in keys}
