"""Base configuration classes and utilities."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseConfig:
    """Base configuration class."""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            k: v.to_dict() if isinstance(v, BaseConfig) else v
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            Configuration instance.
        """
        return cls(**data)

    def update(self, other: dict[str, Any]) -> None:
        """Update configuration with dictionary.

        Args:
            other: Dictionary with updates.
        """
        for k, v in other.items():
            if hasattr(self, k):
                if isinstance(getattr(self, k), BaseConfig) and isinstance(v, dict):
                    getattr(self, k).update(v)
                else:
                    setattr(self, k, v)
