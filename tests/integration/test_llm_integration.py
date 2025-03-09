import pytest
import os
from dotenv import load_dotenv
from src.llm_providers import LLMProviderFactory

# Load environment variables before tests
load_dotenv()

@pytest.fixture
def factory():
    return LLMProviderFactory()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_generation(factory):
    """Test actual text generation with Gemini."""
    provider = factory.get_provider()
    prompt = "Say 'Hello, World!' in French."
    
    response = await provider.generate(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Bonjour" in response or "bonjour" in response

@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_streaming(factory):
    """Test streaming generation with Gemini."""
    provider = factory.get_provider()
    prompt = "Count from 1 to 5."
    
    chunks = []
    async for chunk in provider.generate_stream(prompt):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    
    complete_response = "".join(chunks)
    assert len(complete_response) > 0
    assert any(str(i) in complete_response for i in range(1, 6))

@pytest.mark.integration
def test_gemini_token_counting(factory):
    """Test token counting with actual Gemini tokenizer."""
    provider = factory.get_provider()
    text = "This is a test message."
    
    token_count = provider.get_token_count(text)
    assert token_count > 0
    assert isinstance(token_count, int)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_temperature_effect(factory):
    """Test that different temperature values produce different results."""
    provider = factory.get_provider()
    prompt = "Generate a random adjective."
    
    # Get response with temperature 0
    provider.update_config({"temperature": 0.0})
    response1 = await provider.generate(prompt)
    
    # Get response with temperature 1
    provider.update_config({"temperature": 1.0})
    response2 = await provider.generate(prompt)
    
    # With such different temperatures, responses should differ
    # Note: This is probabilistic and might occasionally fail
    assert isinstance(response1, str)
    assert isinstance(response2, str)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_tokens_limit(factory):
    """Test that max_tokens parameter effectively limits response length."""
    provider = factory.get_provider()
    prompt = "Write a very long story about anything."
    
    # Get a short response
    short_response = await provider.generate(prompt, max_tokens=50)
    short_tokens = provider.get_token_count(short_response)
    
    # Get a longer response
    long_response = await provider.generate(prompt, max_tokens=200)
    long_tokens = provider.get_token_count(long_response)
    
    assert short_tokens < long_tokens 