"""Provider version management."""

from dataclasses import dataclass
from typing import ClassVar

from src.exceptions import InvalidModelError


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
    def from_string(cls, version_str: str) -> "Version":
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
            raise InvalidModelError(msg)

    def __lt__(self, other: "Version") -> bool:
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
    GEMINI_V1: ClassVar["ProviderVersion"]

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

    def supports_capability(
        self, capability: str, model_name: str | None = None,
    ) -> bool:
        """Check if provider supports capability.

        Args:
            capability: Capability to check.
            model_name: Optional model name. If not provided, checks default model.

        Returns:
            True if capability is supported.

        """
        try:
            model = self.get_model(model_name)
            return capability in model.capabilities
        except InvalidModelError:
            return False


# Define known provider versions
ProviderVersion.GEMINI_V1 = ProviderVersion(
    name="gemini",
    version=Version(1, 0, 0),
    supported_models={
        "gemini-pro": ModelVersion(
            name="gemini-pro",
            version=Version(1, 0, 0),
            capabilities=[
                "text-generation",
                "chat",
                "code-generation",
                "code-analysis",
            ],
            min_provider_version=Version(1, 0, 0),
        ),
        "gemini-pro-vision": ModelVersion(
            name="gemini-pro-vision",
            version=Version(1, 0, 0),
            capabilities=[
                "text-generation",
                "chat",
                "image-analysis",
                "multimodal",
            ],
            min_provider_version=Version(1, 0, 0),
        ),
    },
    default_model="gemini-pro",
)
