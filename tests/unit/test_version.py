"""Tests for the version module."""

from __future__ import annotations

import pytest

from src.exceptions import InvalidModelError
from src.llm_providers.version import ModelVersion, ProviderVersion, Version


class TestVersion:
    """Tests for the Version class."""

    def test_version_initialization(self) -> None:
        """Test version initialization."""
        version = Version(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_version_string_representation(self) -> None:
        """Test version string representation."""
        version = Version(1, 2, 3)
        assert str(version) == "1.2.3"

    def test_version_from_string_valid(self) -> None:
        """Test creating version from valid string."""
        version = Version.from_string("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_version_from_string_invalid(self) -> None:
        """Test creating version from invalid string."""
        with pytest.raises(InvalidModelError):
            Version.from_string("invalid")

        with pytest.raises(InvalidModelError):
            Version.from_string("1.2")

        with pytest.raises(InvalidModelError):
            Version.from_string("1.2.3.4")

    def test_version_comparison(self) -> None:
        """Test version comparison."""
        v1 = Version(1, 0, 0)
        v2 = Version(1, 1, 0)
        v3 = Version(1, 1, 1)
        v4 = Version(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v3 < v4
        assert not (v2 < v1)
        assert not (v4 < v3)


class TestModelVersion:
    """Tests for the ModelVersion class."""

    def test_model_version_initialization(self) -> None:
        """Test model version initialization."""
        version = Version(1, 0, 0)
        min_version = Version(0, 5, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=min_version,
        )

        assert model_version.name == "test-model"
        assert model_version.version == version
        assert model_version.capabilities == ["text-generation", "chat"]
        assert model_version.min_provider_version == min_version

    def test_model_version_string_representation(self) -> None:
        """Test model version string representation."""
        version = Version(1, 0, 0)
        min_version = Version(0, 5, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=min_version,
        )

        assert str(model_version) == "test-model@1.0.0"


class TestProviderVersion:
    """Tests for the ProviderVersion class."""

    def test_provider_version_initialization(self) -> None:
        """Test provider version initialization."""
        version = Version(1, 0, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=version,
        )

        provider_version = ProviderVersion(
            name="test-provider",
            version=version,
            supported_models={"test-model": model_version},
            default_model="test-model",
        )

        assert provider_version.name == "test-provider"
        assert provider_version.version == version
        assert provider_version.supported_models == {"test-model": model_version}
        assert provider_version.default_model == "test-model"

    def test_get_model_with_name(self) -> None:
        """Test get_model method with model name."""
        version = Version(1, 0, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=version,
        )

        provider_version = ProviderVersion(
            name="test-provider",
            version=version,
            supported_models={"test-model": model_version},
            default_model="test-model",
        )

        result = provider_version.get_model("test-model")
        assert result == model_version

    def test_get_model_default(self) -> None:
        """Test get_model method with default model."""
        version = Version(1, 0, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=version,
        )

        provider_version = ProviderVersion(
            name="test-provider",
            version=version,
            supported_models={"test-model": model_version},
            default_model="test-model",
        )

        result = provider_version.get_model()
        assert result == model_version

    def test_get_model_not_supported(self) -> None:
        """Test get_model method with unsupported model."""
        version = Version(1, 0, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=version,
        )

        provider_version = ProviderVersion(
            name="test-provider",
            version=version,
            supported_models={"test-model": model_version},
            default_model="test-model",
        )

        with pytest.raises(InvalidModelError):
            provider_version.get_model("unsupported-model")

    def test_has_capability_true(self) -> None:
        """Test has_capability method returns True."""
        version = Version(1, 0, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=version,
        )

        provider_version = ProviderVersion(
            name="test-provider",
            version=version,
            supported_models={"test-model": model_version},
            default_model="test-model",
        )

        assert provider_version.has_capability("test-model", "text-generation")
        assert provider_version.has_capability("test-model", "chat")

    def test_has_capability_false(self) -> None:
        """Test has_capability method returns False."""
        version = Version(1, 0, 0)
        model_version = ModelVersion(
            name="test-model",
            version=version,
            capabilities=["text-generation", "chat"],
            min_provider_version=version,
        )

        provider_version = ProviderVersion(
            name="test-provider",
            version=version,
            supported_models={"test-model": model_version},
            default_model="test-model",
        )

        assert not provider_version.has_capability("test-model", "code-generation")
        assert not provider_version.has_capability("unsupported-model", "text-generation")

    def test_gemini_v1_predefined_version(self) -> None:
        """Test the predefined GEMINI_V1 provider version."""
        gemini_v1 = ProviderVersion.GEMINI_V1

        assert gemini_v1.name == "gemini"
        assert gemini_v1.version == Version(1, 0, 0)
        assert gemini_v1.default_model == "gemini-2.0-flash-lite"

        # Check that it has the expected models
        assert "gemini-2.0-flash-lite" in gemini_v1.supported_models
        assert "gemini-2.0-flash" in gemini_v1.supported_models
        assert "gemini-1.5-flash" in gemini_v1.supported_models
        assert "gemini-1.5-pro" in gemini_v1.supported_models

        # Check capabilities of a model
        model = gemini_v1.get_model("gemini-2.0-flash-lite")
        assert "text-generation" in model.capabilities
        assert "chat" in model.capabilities
        assert "code-generation" in model.capabilities
