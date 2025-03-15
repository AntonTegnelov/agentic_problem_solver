"""Tests for hierarchical message routing."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.common_types.error_types import AgentNotFoundError, RoutingError
from src.messages.routing import HierarchicalRouter


@pytest.fixture
def mock_router():
    """Create a mock router."""
    router = MagicMock()
    router.route_message = AsyncMock()
    return router


@pytest.fixture
def mock_registry():
    """Create a mock agent registry."""
    registry = MagicMock()
    registry.get_parent_id.return_value = "parent-1"
    registry.get_children.return_value = ["child-1", "child-2"]
    registry.is_child_of.return_value = True
    registry.get_agent.return_value = MagicMock()
    return registry


@pytest.fixture
def hierarchical_router(mock_router, mock_registry):
    """Create a hierarchical router with mock dependencies."""
    return HierarchicalRouter(mock_router, mock_registry)


@pytest.fixture
def test_message():
    """Create a test message."""
    return HumanMessage(content="Test message")


class TestHierarchicalRouter:
    """Tests for the HierarchicalRouter class."""

    async def test_route_to_parent(self, hierarchical_router, test_message, mock_router, mock_registry) -> None:
        """Test routing a message to a parent agent."""
        # Setup
        agent_id = "agent-1"
        parent_id = "parent-1"
        mock_registry.get_parent_id.return_value = parent_id
        mock_router.route_message.return_value = "success"

        # Execute
        result = await hierarchical_router.route_to_parent(test_message, agent_id)

        # Verify
        assert result == "success"
        mock_registry.get_parent_id.assert_called_once_with(agent_id)
        mock_router.route_message.assert_called_once()
        assert test_message.additional_kwargs["metadata"]["sender_id"] == agent_id
        assert test_message.additional_kwargs["metadata"]["receiver_id"] == parent_id
        assert test_message.additional_kwargs["metadata"]["hierarchy_path"] == [agent_id]

    async def test_route_to_parent_no_parent(self, hierarchical_router, test_message, mock_registry) -> None:
        """Test routing a message to a parent when the agent has no parent."""
        # Setup
        agent_id = "agent-1"
        mock_registry.get_parent_id.return_value = None

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Agent {agent_id} has no parent"):
            await hierarchical_router.route_to_parent(test_message, agent_id)

    async def test_route_to_parent_parent_not_found(
        self,
        hierarchical_router,
        test_message,
        mock_router,
        mock_registry,
    ) -> None:
        """Test routing a message to a parent that doesn't exist."""
        # Setup
        agent_id = "agent-1"
        parent_id = "parent-1"
        mock_registry.get_parent_id.return_value = parent_id
        mock_router.route_message.side_effect = AgentNotFoundError(f"Agent not found: {parent_id}")

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Parent agent {parent_id} not found for {agent_id}"):
            await hierarchical_router.route_to_parent(test_message, agent_id)

    async def test_route_to_child(self, hierarchical_router, test_message, mock_router, mock_registry) -> None:
        """Test routing a message from a parent to a child agent."""
        # Setup
        parent_id = "parent-1"
        child_id = "child-1"
        grandparent_id = "grandparent-1"
        mock_registry.get_parent_id.side_effect = lambda agent_id: {
            parent_id: grandparent_id,
            child_id: parent_id,
        }.get(agent_id)
        mock_router.route_message.return_value = "success"

        # Execute
        result = await hierarchical_router.route_to_child(test_message, parent_id, child_id)

        # Verify
        assert result == "success"
        mock_registry.is_child_of.assert_called_once_with(child_id, parent_id)
        mock_router.route_message.assert_called_once()
        assert test_message.additional_kwargs["metadata"]["sender_id"] == parent_id
        assert test_message.additional_kwargs["metadata"]["receiver_id"] == child_id
        assert test_message.additional_kwargs["metadata"]["sender_parent_id"] == grandparent_id
        assert test_message.additional_kwargs["metadata"]["receiver_parent_id"] == parent_id
        assert test_message.additional_kwargs["metadata"]["hierarchy_path"] == [parent_id]

    async def test_route_to_child_not_a_child(self, hierarchical_router, test_message, mock_registry) -> None:
        """Test routing a message to an agent that is not a child of the parent."""
        # Setup
        parent_id = "parent-1"
        child_id = "child-1"
        mock_registry.is_child_of.return_value = False

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Agent {child_id} is not a child of {parent_id}"):
            await hierarchical_router.route_to_child(test_message, parent_id, child_id)

    async def test_route_to_child_not_found(
        self,
        hierarchical_router,
        test_message,
        mock_router,
        mock_registry,
    ) -> None:
        """Test routing a message to a child that doesn't exist."""
        # Setup
        parent_id = "parent-1"
        child_id = "child-1"
        mock_router.route_message.side_effect = AgentNotFoundError(f"Agent not found: {child_id}")

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Child agent {child_id} not found for {parent_id}"):
            await hierarchical_router.route_to_child(test_message, parent_id, child_id)

    async def test_route_to_children(self, hierarchical_router, test_message, mock_router, mock_registry) -> None:
        """Test broadcasting a message to all children of a parent."""
        # Setup
        parent_id = "parent-1"
        children = ["child-1", "child-2"]
        mock_registry.get_children.return_value = children
        mock_router.route_message.return_value = "success"

        # Execute
        results = await hierarchical_router.route_to_children(test_message, parent_id)

        # Verify
        assert len(results) == 2
        assert all(result == "success" for result in results)
        mock_registry.get_children.assert_called_once_with(parent_id)
        assert mock_router.route_message.call_count == 2

    async def test_route_to_children_no_children(self, hierarchical_router, test_message, mock_registry) -> None:
        """Test broadcasting a message when the parent has no children."""
        # Setup
        parent_id = "parent-1"
        mock_registry.get_children.return_value = []

        # Execute
        results = await hierarchical_router.route_to_children(test_message, parent_id)

        # Verify
        assert results == []
        mock_registry.get_children.assert_called_once_with(parent_id)

    async def test_route_to_children_with_errors(
        self,
        hierarchical_router,
        test_message,
        mock_router,
        mock_registry,
    ) -> None:
        """Test broadcasting a message when some children can't be reached."""
        # Setup
        parent_id = "parent-1"
        children = ["child-1", "child-2"]
        mock_registry.get_children.return_value = children

        # First child succeeds, second child fails
        mock_router.route_message.side_effect = [
            "success",
            AgentNotFoundError("Agent not found: child-2"),
        ]

        # Patch the route_to_child method to handle the error for the second child
        original_route_to_child = hierarchical_router.route_to_child

        async def patched_route_to_child(message, p_id, c_id, chain=None) -> str:
            if c_id == "child-1":
                return "success"
            msg = f"Child agent {c_id} not found for {p_id}"
            raise RoutingError(msg)

        hierarchical_router.route_to_child = patched_route_to_child

        # Execute
        results = await hierarchical_router.route_to_children(test_message, parent_id)

        # Verify
        assert len(results) == 1
        assert results[0] == "success"
        mock_registry.get_children.assert_called_once_with(parent_id)

        # Restore the original method
        hierarchical_router.route_to_child = original_route_to_child

    async def test_route_to_sibling(self, hierarchical_router, test_message, mock_router, mock_registry) -> None:
        """Test routing a message between sibling agents."""
        # Setup
        sender_id = "agent-1"
        sibling_id = "agent-2"
        parent_id = "parent-1"
        mock_registry.get_parent_id.return_value = parent_id
        mock_router.route_message.return_value = "success"

        # Execute
        result = await hierarchical_router.route_to_sibling(test_message, sender_id, sibling_id)

        # Verify
        assert result == "success"
        assert mock_registry.get_parent_id.call_count == 2
        mock_router.route_message.assert_called_once()
        assert test_message.additional_kwargs["metadata"]["sender_id"] == sender_id
        assert test_message.additional_kwargs["metadata"]["receiver_id"] == sibling_id
        assert test_message.additional_kwargs["metadata"]["sender_parent_id"] == parent_id
        assert test_message.additional_kwargs["metadata"]["receiver_parent_id"] == parent_id
        assert test_message.additional_kwargs["metadata"]["hierarchy_path"] == [sender_id]

    async def test_route_to_sibling_not_siblings(self, hierarchical_router, test_message, mock_registry) -> None:
        """Test routing a message between agents that are not siblings."""
        # Setup
        sender_id = "agent-1"
        sibling_id = "agent-2"
        mock_registry.get_parent_id.side_effect = lambda agent_id: {
            sender_id: "parent-1",
            sibling_id: "parent-2",
        }.get(agent_id)

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Agents {sender_id} and {sibling_id} are not siblings"):
            await hierarchical_router.route_to_sibling(test_message, sender_id, sibling_id)

    async def test_route_to_sibling_no_parent(self, hierarchical_router, test_message, mock_registry) -> None:
        """Test routing a message when one of the agents has no parent."""
        # Setup
        sender_id = "agent-1"
        sibling_id = "agent-2"
        mock_registry.get_parent_id.side_effect = lambda agent_id: {
            sender_id: "parent-1",
            sibling_id: None,
        }.get(agent_id)

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Agents {sender_id} and {sibling_id} are not siblings"):
            await hierarchical_router.route_to_sibling(test_message, sender_id, sibling_id)

    async def test_route_to_sibling_not_found(
        self,
        hierarchical_router,
        test_message,
        mock_router,
        mock_registry,
    ) -> None:
        """Test routing a message to a sibling that doesn't exist."""
        # Setup
        sender_id = "agent-1"
        sibling_id = "agent-2"
        mock_router.route_message.side_effect = AgentNotFoundError(f"Agent not found: {sibling_id}")

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Sibling agent {sibling_id} not found"):
            await hierarchical_router.route_to_sibling(test_message, sender_id, sibling_id)

    async def test_route_by_path(self, hierarchical_router, test_message, mock_router, mock_registry) -> None:
        """Test routing a message along a specific path in the hierarchy."""
        # Setup
        path = ["parent-1", "child-1", "grandchild-1"]
        mock_router.route_message.return_value = "success"

        # Execute
        result = await hierarchical_router.route_by_path(test_message, path)

        # Verify
        assert result == "success"
        assert mock_registry.is_child_of.call_count == 2
        mock_router.route_message.assert_called_once()
        assert test_message.additional_kwargs["metadata"]["sender_id"] == path[0]
        assert test_message.additional_kwargs["metadata"]["receiver_id"] == path[-1]
        assert test_message.additional_kwargs["metadata"]["hierarchy_path"] == path

    async def test_route_by_path_too_short(self, hierarchical_router, test_message) -> None:
        """Test routing a message with a path that's too short."""
        # Setup
        path = ["agent-1"]

        # Execute and verify
        with pytest.raises(RoutingError, match="Path must contain at least a source and destination agent"):
            await hierarchical_router.route_by_path(test_message, path)

    async def test_route_by_path_invalid_path(self, hierarchical_router, test_message, mock_registry) -> None:
        """Test routing a message with an invalid path."""
        # Setup
        path = ["parent-1", "child-1", "grandchild-1"]
        mock_registry.is_child_of.side_effect = [True, False]

        # Execute and verify
        with pytest.raises(RoutingError, match=f"Invalid path: {path[2]} is not a child of {path[1]}"):
            await hierarchical_router.route_by_path(test_message, path)

    async def test_route_by_path_agent_not_found(
        self,
        hierarchical_router,
        test_message,
        mock_router,
        mock_registry,
    ) -> None:
        """Test routing a message to an agent in the path that doesn't exist."""
        # Setup
        path = ["parent-1", "child-1", "grandchild-1"]
        mock_router.route_message.side_effect = AgentNotFoundError(f"Agent not found: {path[-1]}")

        # Execute and verify
        with pytest.raises(RoutingError, match=r"Agent grandchild-1 not found in path"):
            await hierarchical_router.route_by_path(test_message, path)

    def test_get_agent(self, hierarchical_router, mock_registry) -> None:
        """Test getting an agent by ID."""
        # Setup
        agent_id = "agent-1"
        mock_agent = MagicMock()
        mock_registry.get_agent.return_value = mock_agent

        # Execute
        agent = hierarchical_router.get_agent(agent_id)

        # Verify
        assert agent == mock_agent
        mock_registry.get_agent.assert_called_once_with(agent_id)

    def test_get_agent_not_found(self, hierarchical_router, mock_registry) -> None:
        """Test getting an agent that doesn't exist."""
        # Setup
        agent_id = "agent-1"
        mock_registry.get_agent.return_value = None

        # Execute and verify
        with pytest.raises(AgentNotFoundError, match=f"Agent not found: {agent_id}"):
            hierarchical_router.get_agent(agent_id)
