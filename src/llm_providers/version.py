"""Provider version management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.common_types import InvalidModelError


@dataclass
class Version:
    """Provider version information."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        """Get string representation.

        Returns:
            Version string in format major.minor.patch.

        """
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> Version:
        """Create version from string.

        Args:
            version_str: Version string in format major.minor.patch.

        Returns:
            Version instance.

        Raises:
            InvalidModelError: If version string is invalid.

        """
        try:
            major, minor, patch = map(int, version_str.split("."))
            return cls(major=major, minor=minor, patch=patch)
        except (ValueError, AttributeError) as e:
            msg = f"Invalid version format: {version_str}. Error: {e!s}"
            raise InvalidModelError(msg) from e

    def __lt__(self, other: Version) -> bool:
        """Compare versions.

        Args:
            other: Version to compare with.

        Returns:
            True if this version is less than other.

        """
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )


@dataclass
class ModelVersion:
    """Model version information."""

    name: str
    version: Version
    capabilities: list[str]
    min_provider_version: Version

    def __str__(self) -> str:
        """Get string representation.

        Returns:
            Model version string.

        """
        return f"{self.name}@{self.version}"


@dataclass
class ProviderVersion:
    """Provider version information."""

    name: str
    version: Version
    supported_models: dict[str, ModelVersion]
    default_model: str

    # Known provider versions
    GEMINI_V1: ClassVar[ProviderVersion]

    def get_model(self, model_name: str | None = None) -> ModelVersion:
        """Get model version.

        Args:
            model_name: Optional model name. If not provided, returns default model.

        Returns:
            Model version.

        Raises:
            InvalidModelError: If model is not supported.

        """
        if model_name is None:
            model_name = self.default_model

        if model_name not in self.supported_models:
            msg = f"Model {model_name} not supported by provider {self.name}@{self.version}"
            raise InvalidModelError(
                msg,
            )

        return self.supported_models[model_name]

    def has_capability(self, model_name: str, capability: str) -> bool:
        """Check if model has capability.

        Args:
            model_name: Model name.
            capability: Capability to check.

        Returns:
            True if model has capability, False otherwise.

        """
        try:
            model = self.get_model(model_name)
        except InvalidModelError:
            return False
        else:
            return capability in model.capabilities


# Define known provider versions
ProviderVersion.GEMINI_V1 = ProviderVersion(
    name="gemini",
    version=Version(1, 0, 0),
    supported_models={
        "gemini-2.0-flash-lite": ModelVersion(
            name="gemini-2.0-flash-lite",
            version=Version(2, 0, 0),
            capabilities=[
                "text-generation",
                "chat",
                "code-generation",
                "code-analysis",
                "multimodal",
            ],
            min_provider_version=Version(1, 0, 0),
        ),
        "gemini-2.0-flash": ModelVersion(
            name="gemini-2.0-flash",
            version=Version(2, 0, 0),
            capabilities=[
                "text-generation",
                "chat",
                "code-generation",
                "code-analysis",
                "multimodal",
            ],
            min_provider_version=Version(1, 0, 0),
        ),
        "gemini-1.5-flash": ModelVersion(
            name="gemini-1.5-flash",
            version=Version(1, 5, 0),
            capabilities=[
                "text-generation",
                "chat",
                "code-generation",
                "code-analysis",
                "multimodal",
            ],
            min_provider_version=Version(1, 0, 0),
        ),
        "gemini-1.5-pro": ModelVersion(
            name="gemini-1.5-pro",
            version=Version(1, 5, 0),
            capabilities=[
                "text-generation",
                "chat",
                "code-generation",
                "code-analysis",
                "multimodal",
            ],
            min_provider_version=Version(1, 0, 0),
        ),
    },
    default_model="gemini-2.0-flash-lite",
)
