import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import AsyncGenerator, Dict, Any
from src.llm_providers import LLMProvider, GeminiProvider, LLMProviderFactory
from google.api_core.exceptions import ResourceExhausted
import asyncio
import os

@pytest.fixture(autouse=True)
def reset_factory():
    """Reset factory singleton between tests."""
    LLMProviderFactory._instance = None
    yield

@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for all tests."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
        yield

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self):
        """Initialize mock provider."""
        super().__init__()
        self._config = {"temperature": 0.7, "max_tokens": 100}
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock text generation."""
        return "Mock response"
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Mock streaming text generation."""
        chunks = ["Mock", " stream", " response"]
        for chunk in chunks:
            yield chunk
    
    def get_token_count(self, text: str) -> int:
        """Mock token counting."""
        return len(text.split())
    
    def get_config(self):
        return self._config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self._config.update(config)

@pytest.fixture
def mock_provider():
    return MockLLMProvider()

@pytest.fixture
def factory():
    """Create LLM provider factory."""
    return LLMProviderFactory()

def test_provider_registration(factory):
    """Test provider registration in factory."""
    factory.register_provider("mock", MockLLMProvider)
    assert "mock" in factory.available_providers

def test_factory_initialization_error():
    """Test factory initialization with invalid provider."""
    with patch.dict("os.environ", {"LLM_PROVIDER": "invalid"}):
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            LLMProviderFactory()

def test_factory_missing_api_key():
    """Test factory initialization with missing API key."""
    with patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable not set"):
            LLMProviderFactory()

def test_factory_set_invalid_provider(factory):
    """Test setting an invalid provider."""
    with pytest.raises(ValueError, match="Unsupported provider: invalid"):
        factory.set_provider("invalid")

def test_factory_set_api_key(factory):
    """Test setting API key directly."""
    factory._set_api_key("new_key")
    assert factory._get_api_key() == "new_key"

@pytest.mark.asyncio
async def test_factory_create_provider(factory):
    """Test creating provider with specific model."""
    provider = factory._create_provider("test_key", "test-model")
    assert isinstance(provider, GeminiProvider)
    assert provider.api_key == "test_key"
    assert provider.model == "test-model"

@pytest.mark.asyncio
async def test_factory_get_provider_caching(factory):
    """Test that get_provider caches the provider instance."""
    provider1 = factory.get_provider()
    provider2 = factory.get_provider()
    assert provider1 is provider2  # Same instance
    
    # Different model should create new instance
    provider3 = factory.get_provider(model="different-model")
    assert provider3 is not provider1

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
        mock_model.assert_called_once_with("gemini-2.0-flash-lite")

        assert provider.api_key == "fake_api_key"
        assert provider.model == "gemini-2.0-flash-lite"
        assert isinstance(provider.config, dict)

@pytest.mark.asyncio
async def test_gemini_provider_retry_logic():
    """Test retry logic in Gemini provider."""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model:
        provider = GeminiProvider("fake_api_key")
        
        # Mock the generate method to fail twice then succeed
        mock_response = Mock()
        mock_response.text = "Success"
        mock_generate = AsyncMock(side_effect=[
            ResourceExhausted("Rate limit"),
            ResourceExhausted("Rate limit"),
            mock_response
        ])
        provider._model.generate_content_async = mock_generate
        
        response = await provider.generate("test")
        assert response == "Success"
        assert mock_generate.call_count == 3

@pytest.mark.asyncio
async def test_gemini_provider_retry_exhaustion():
    """Test retry logic when all retries are exhausted."""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model:
        provider = GeminiProvider("fake_api_key")
        
        # Mock the generate method to always fail
        mock_generate = AsyncMock(side_effect=ResourceExhausted("Rate limit"))
        provider._model.generate_content_async = mock_generate
        
        with pytest.raises(ResourceExhausted):
            await provider.generate("test")
        assert mock_generate.call_count == 3  # Default max retries

@pytest.mark.asyncio
async def test_gemini_provider_streaming():
    """Test Gemini provider streaming with retry logic."""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model:
        provider = GeminiProvider("fake_api_key")
        
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.text = "Hello"
        mock_chunk2 = Mock()
        mock_chunk2.text = " World"
        
        async def mock_aiter(self):
            yield mock_chunk1
            yield mock_chunk2
        
        mock_response = AsyncMock()
        mock_response.__aiter__ = mock_aiter
        
        provider._model.generate_content_async = AsyncMock(return_value=mock_response)
        
        chunks = []
        async for chunk in provider.generate_stream("test"):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " World"]

@pytest.mark.asyncio
async def test_gemini_provider_empty_chunks():
    """Test Gemini provider handling of empty chunks in streaming."""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model:
        provider = GeminiProvider("fake_api_key")
        
        # Mock streaming response with empty chunks
        mock_chunk1 = Mock()
        mock_chunk1.text = ""  # Empty chunk
        mock_chunk2 = Mock()
        mock_chunk2.text = "Content"
        mock_chunk3 = Mock()
        mock_chunk3.text = ""  # Empty chunk
        
        async def mock_aiter(self):
            yield mock_chunk1
            yield mock_chunk2
            yield mock_chunk3
        
        mock_response = AsyncMock()
        mock_response.__aiter__ = mock_aiter
        
        provider._model.generate_content_async = AsyncMock(return_value=mock_response)
        
        chunks = []
        async for chunk in provider.generate_stream("test"):
            chunks.append(chunk)
        
        assert chunks == ["Content"]  # Empty chunks should be filtered out

def test_gemini_provider_config_management():
    """Test Gemini provider configuration management."""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel"):
        provider = GeminiProvider("fake_api_key")
        
        # Test initial config
        initial_config = provider.get_config()
        assert initial_config["temperature"] == 0.7
        assert id(initial_config) != id(provider.config)  # Should be a copy
        
        # Test updating config
        new_config = {"temperature": 0.9, "max_output_tokens": 100}
        provider.update_config(new_config)
        updated_config = provider.get_config()
        assert updated_config["temperature"] == 0.9
        assert updated_config["max_output_tokens"] == 100
        
        # Test generation config
        gen_config = provider._get_generation_config(max_tokens=50, temperature=0.5)
        assert gen_config.temperature == 0.5
        assert gen_config.max_output_tokens == 50

def test_gemini_provider_safety_settings():
    """Test Gemini provider safety settings."""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel"):
        provider = GeminiProvider("fake_api_key")
        
        settings = provider._get_safety_settings()
        assert len(settings) == 4  # Should have 4 safety categories
        assert all(s["threshold"] == "BLOCK_NONE" for s in settings)
        assert any(s["category"] == "HARM_CATEGORY_HARASSMENT" for s in settings)

def test_factory_singleton(factory):
    """Test factory singleton pattern."""
    another_factory = LLMProviderFactory()
    assert factory is another_factory

@pytest.mark.asyncio
async def test_factory_provider_switching(factory):
    """Test provider switching in factory."""
    factory.register_provider("mock", MockLLMProvider)
    
    # Reset the factory's state
    factory._provider = None
    factory._model = None
    
    # Test switching to mock provider
    factory.set_provider("mock")
    provider = factory.get_provider()
    assert isinstance(provider, MockLLMProvider), "Factory should return MockLLMProvider"

    response = await provider.generate("Test")
    assert response == "Mock response"

def test_base_provider_abstract_methods():
    """Test that base provider methods raise NotImplementedError."""
    class TestProvider(LLMProvider):
        async def generate(self, prompt: str, **kwargs) -> str:
            raise NotImplementedError()
        
        async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
            async def _gen():
                yield "test"  # Need to yield at least once to test the error
                raise NotImplementedError()
            async for item in _gen():
                yield item
        
        def get_token_count(self, text: str) -> int:
            raise NotImplementedError()
        
        def get_config(self) -> Dict[str, Any]:
            raise NotImplementedError()
        
        def update_config(self, config: Dict[str, Any]) -> None:
            raise NotImplementedError()
    
    provider = TestProvider()
    
    with pytest.raises(NotImplementedError):
        provider.get_config()
    
    with pytest.raises(NotImplementedError):
        provider.update_config({})
    
    with pytest.raises(NotImplementedError):
        provider.get_token_count("test")
    
    with pytest.raises(NotImplementedError):
        asyncio.run(provider.generate("test"))
    
    async def test_stream():
        async for _ in provider.generate_stream("test"):
            pass
    with pytest.raises(NotImplementedError):
        asyncio.run(test_stream()) 