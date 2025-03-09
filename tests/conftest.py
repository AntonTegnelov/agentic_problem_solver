"""Test configuration file."""

import asyncio
import os

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope="session")
async def event_loop_session():
    """Create a session-scoped event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test():
    """Set up any test prerequisites."""
    # Setup code here if needed
    yield
    # Cleanup code here if needed


@pytest.fixture(scope="function")
def setup_env() -> None:
    """Set up test environment."""
    os.environ["GOOGLE_API_KEY"] = "test_key"
    yield
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
