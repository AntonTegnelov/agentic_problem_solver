"""Configuration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common_types import ConfigError


def load_env_var(
    key: str,
    env_file: str | Path = ".env",
    default: str | None = None,
    *,
    required: bool = False,
) -> str:
    """Load environment variable from .env file.

    Args:
        key: Environment variable key.
        env_file: Path to .env file.
        default: Default value if key not found.
        required: Whether the key is required.

    Returns:
        Environment variable value.

    Raises:
        ConfigError: If .env file is not found or key is not found.

    """
    env_path = Path(env_file)
    if not env_path.exists():
        if default is not None:
            return default
        msg = f"No {env_file} file found"
        raise ConfigError(msg)

    with env_path.open() as f:
        for line in f:
            if line.startswith(f"{key}="):
                return line.split("=")[1].strip()

    if default is not None:
        return default

    if required:
        msg = f"{key} not found in {env_file}"
        raise ConfigError(msg)

    return ""


def load_config_from_env(prefix: str, env_file: str | Path = ".env") -> dict[str, Any]:
    """Load configuration from environment variables with prefix.

    Args:
        prefix: Environment variable prefix.
        env_file: Path to .env file.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigError: If prefix is invalid.

    """
    if not prefix:
        msg = "Prefix cannot be empty"
        raise ConfigError(msg)

    if not prefix.endswith("_"):
        msg = "Prefix must end with underscore"
        raise ConfigError(msg)

    config: dict[str, Any] = {}
    env_path = Path(env_file)
    if not env_path.exists():
        return config

    with env_path.open() as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            key, value = line.strip().split("=", 1)
            key = key[len(prefix) :].lower()  # Remove prefix and convert to lowercase
            if "__" in key:
                # Handle nested keys
                parts = key.split("__")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value

    return config
