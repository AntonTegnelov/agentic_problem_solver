import pytest
from unittest.mock import Mock, patch
from typing import AsyncGenerator
from src.llm_providers import LLMProvider, GeminiProvider, LLMProviderFactory

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.config = {"temperature": 0.7}
    
    async def generate(self, prompt: str, **kwargs):
        return "Mock response"
    
    async def generate_stream(self, prompt: str, **kwargs):
        async def mock_stream():
            yield "Mock"
            yield " stream"
            yield " response"
        return mock_stream()
    
    def get_token_count(self, text: str):
        return len(text.split())
    
    def get_config(self):
        return self.config.copy()
    
    def update_config(self, config):
        self.config.update(config)

@pytest.fixture
def mock_provider():
    return MockLLMProvider()

@pytest.fixture
def factory():
    return LLMProviderFactory()

def test_provider_registration(factory):
    """Test provider registration in factory."""
    factory.register_provider("mock", MockLLMProvider)
    assert "mock" in factory.available_providers

@pytest.mark.asyncio
async def test_mock_provider_generate(mock_provider):
    """Test basic text generation."""
    response = await mock_provider.generate("Test prompt")
    assert isinstance(response, str)
    assert response == "Mock response"

@pytest.mark.asyncio
async def test_mock_provider_stream(mock_provider):
    """Test streaming text generation."""
    chunks = []
    async for chunk in mock_provider.generate_stream("Test prompt"):
        chunks.append(chunk)
    assert chunks == ["Mock", " stream", " response"]

def test_mock_provider_token_count(mock_provider):
    """Test token counting."""
    text = "This is a test"
    count = mock_provider.get_token_count(text)
    assert count == 4  # number of words

def test_mock_provider_config(mock_provider):
    """Test configuration management."""
    initial_config = mock_provider.get_config()
    assert initial_config["temperature"] == 0.7
    
    mock_provider.update_config({"temperature": 0.9})
    updated_config = mock_provider.get_config()
    assert updated_config["temperature"] == 0.9

@pytest.mark.asyncio
async def test_gemini_provider_initialization():
    """Test Gemini provider initialization."""
    with patch("google.generativeai.configure") as mock_configure, \
         patch("google.generativeai.GenerativeModel") as mock_model:
        provider = GeminiProvider("fake_api_key")
        
        mock_configure.assert_called_once_with(api_key="fake_api_key")
        mock_model.assert_called_once_with("gemini-pro")
        
        assert provider.api_key == "fake_api_key"
        assert provider.model == "gemini-pro"
        assert isinstance(provider.config, dict)

def test_factory_singleton(factory):
    """Test factory singleton pattern."""
    another_factory = LLMProviderFactory()
    assert factory is another_factory

@pytest.mark.asyncio
async def test_factory_provider_switching(factory):
    """Test provider switching in factory."""
    factory.register_provider("mock", MockLLMProvider)
    
    with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
        factory.set_provider("mock")
        provider = factory.get_provider()
        assert isinstance(provider, MockLLMProvider)
        
        response = await provider.generate("Test")
        assert response == "Mock response" 