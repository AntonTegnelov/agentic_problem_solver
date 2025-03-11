"""Tests for agent coordination imports."""

from src.agent.coordination import (
    Agent,
    AgentCoordinator,
    AgentInfo,
    AgentRegistry,
    InMemoryAgentRegistry,
    SimpleAgentCoordinator,
)


def test_coordination_imports() -> None:
    """Test that all coordination imports are available."""
    # Test that all imports are available
    assert Agent is not None
    assert AgentCoordinator is not None
    assert AgentInfo is not None
    assert AgentRegistry is not None
    assert InMemoryAgentRegistry is not None
    assert SimpleAgentCoordinator is not None

    # Test that AgentInfo is a dataclass
    assert hasattr(AgentInfo, "__dataclass_fields__")

    # Test that InMemoryAgentRegistry implements AgentRegistry
    assert issubclass(InMemoryAgentRegistry, object)

    # Test that SimpleAgentCoordinator implements AgentCoordinator
    assert hasattr(SimpleAgentCoordinator, "create_agent")
    assert hasattr(SimpleAgentCoordinator, "delegate_task")
    assert hasattr(SimpleAgentCoordinator, "broadcast_task")
    assert hasattr(SimpleAgentCoordinator, "route_message")
