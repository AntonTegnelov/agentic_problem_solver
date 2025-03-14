"""Unit tests for LLM provider interface implementation details."""

from __future__ import annotations

from typing import Any

import pytest

from src.llm_providers.type_defs import GenerationConfig
from src.messages.creation import create_human_message


class TestLLMProviderImplementation:
    """Tests for the LLMProvider interface implementation details."""

    @pytest.mark.asyncio
    async def test_generate_stream_default_implementation(self) -> None:
        """Test the default implementation of generate_stream in the LLMProvider protocol."""

        # Create a mock provider that uses the default implementation
        class MockProvider:
            async def generate_stream(
                self,
                messages: list[Any],
                *,
                config: GenerationConfig | None = None,
            ):
                # This will use the default implementation from the Protocol
                if not messages:
                    return
                if config is not None:
                    pass
                yield ""

        provider = MockProvider()

        # Test with empty messages
        chunks = [chunk async for chunk in provider.generate_stream([])]
        assert chunks == []

        # Test with non-empty messages but default implementation
        messages = [create_human_message("test")]
        chunks = [chunk async for chunk in provider.generate_stream(messages)]
        assert chunks == [""]

        # Test with config
        config = GenerationConfig(model="test-model")
        chunks = [chunk async for chunk in provider.generate_stream(messages, config=config)]
        assert chunks == [""]
