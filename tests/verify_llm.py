import asyncio
from dotenv import load_dotenv
from src.llm_providers import LLMProviderFactory


async def verify_llm_provider():
    """Verify that the LLM provider works correctly."""
    print("Verifying LLM provider functionality...")

    # Initialize provider
    factory = LLMProviderFactory()
    provider = factory.get_provider()

    # Test basic generation
    prompt = "Write 'Hello, World!' in three different programming languages."
    print("\nTesting basic generation...")
    response = await provider.generate(prompt)
    print(f"Response:\n{response}\n")

    # Test streaming
    print("Testing streaming generation...")
    async for chunk in provider.generate_stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")

    # Test token counting
    text = "This is a test message to count tokens."
    token_count = provider.get_token_count(text)
    print(f"Token count for '{text}': {token_count}")

    # Test configuration
    print("\nTesting configuration...")
    config = provider.get_config()
    print(f"Current config: {config}")

    new_config = {"temperature": 0.9}
    provider.update_config(new_config)
    updated_config = provider.get_config()
    print(f"Updated config: {updated_config}")

    print("\nVerification complete!")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Run verification
    asyncio.run(verify_llm_provider())
