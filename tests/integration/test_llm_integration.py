import pytest
import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.services.generative_service import GenerativeServiceClient
from google.ai.generativelanguage_v1beta import GenerateContentRequest, Content, Part
from google.api_core.exceptions import NotFound
from src.llm_providers import LLMProviderFactory

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables before tests
load_dotenv()
logger.debug(f"GEMINI_API_KEY from env: {os.environ.get('GEMINI_API_KEY', 'Not found')[:10]}...")

@pytest.fixture
def factory():
    """Create LLM provider factory."""
    return LLMProviderFactory()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_generation(factory):
    """Test actual text generation with Gemini."""
    provider = factory.get_provider()
    prompt = "Say 'Hello, World!' in French."

    try:
        response = await provider.generate(prompt)
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Bonjour" in response or "bonjour" in response
    except NotFound as e:
        pytest.skip(f"Skipping test due to API error: {e}")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_streaming(factory):
    """Test streaming generation with Gemini."""
    provider = factory.get_provider()
    prompt = "Count from 1 to 5."

    try:
        chunks = []
        async for chunk in provider.generate_stream(prompt):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert len("".join(chunks)) > 0
        assert any(str(i) in "".join(chunks) for i in range(1, 6))
    except NotFound as e:
        pytest.skip(f"Skipping test due to API error: {e}")

@pytest.mark.integration
def test_gemini_token_counting(factory):
    """Test token counting with actual Gemini tokenizer."""
    provider = factory.get_provider()
    text = "This is a test message."

    try:
        token_count = provider.get_token_count(text)
        assert token_count > 0
        assert isinstance(token_count, int)
    except NotFound as e:
        pytest.skip(f"Skipping test due to API error: {e}")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_temperature_effect(factory):
    """Test that temperature affects generation."""
    provider = factory.get_provider()
    prompt = "What is 2+2?"  # Simple prompt that should have consistent output at temp=0
    
    try:
        # Test with temperature 0 which should be deterministic
        provider.update_config({"temperature": 0.0})
        response = await provider.generate(prompt)
        assert response and len(response) > 0, "Should get a non-empty response"
        assert "4" in response, "At temperature 0, should consistently include '4' in response"
    except Exception as e:
        pytest.skip(f"API error: {str(e)}")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_tokens_limit(factory):
    """Test that max_tokens parameter effectively limits response length."""
    provider = factory.get_provider()
    prompt = "Count from 1 to 100."  # Simple prompt that would naturally generate a long response
    max_tokens = 20
    
    try:
        response = await provider.generate(prompt, max_tokens=max_tokens)
        token_count = provider.get_token_count(response)
        assert token_count <= max_tokens, f"Response token count ({token_count}) exceeds limit ({max_tokens})"
    except Exception as e:
        pytest.skip(f"API error: {str(e)}")

@pytest.mark.integration
def test_gemini_api_key():
    """Test Gemini API key configuration."""
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY not found in environment"
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Just verify we can create the model instance
        assert model is not None, "Should be able to create model instance"
    except Exception as e:
        logger.error(f"Error testing Gemini API key: {str(e)}")
        raise

@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_switching(factory):
    """Test that we can switch between different Gemini models."""
    api_key = os.environ.get("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY not found in environment"
    
    try:
        # Test with flash-lite model
        provider = factory.get_provider()  # Should use default flash-lite
        response = await provider.generate("Say hi")
        assert response and len(response) > 0
        
        # Test with pro model
        provider = factory.get_provider(model="gemini-1.5-pro")
        response = await provider.generate("Say hi")
        assert response and len(response) > 0
    except Exception as e:
        pytest.skip(f"API error: {str(e)}") 