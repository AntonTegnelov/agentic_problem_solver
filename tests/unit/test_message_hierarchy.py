"""Tests for message hierarchy functions."""

from src.messages.creation import create_ai_message, create_human_message
from src.messages.utils import (
    add_to_hierarchy_path,
    get_hierarchy_path,
    get_receiver_id,
    get_receiver_parent_id,
    get_sender_id,
    get_sender_parent_id,
    is_hierarchical_message,
    set_hierarchy_path,
    set_receiver_id,
    set_receiver_parent_id,
    set_sender_id,
    set_sender_parent_id,
)


class TestMessageHierarchy:
    """Tests for message hierarchy functions."""

    def test_sender_id(self) -> None:
        """Test sender ID functions."""
        message = create_human_message("Test message")
        assert get_sender_id(message) is None

        set_sender_id(message, "agent1")
        assert get_sender_id(message) == "agent1"

    def test_receiver_id(self) -> None:
        """Test receiver ID functions."""
        message = create_ai_message("Test response")
        assert get_receiver_id(message) is None

        set_receiver_id(message, "agent2")
        assert get_receiver_id(message) == "agent2"

    def test_sender_parent_id(self) -> None:
        """Test sender parent ID functions."""
        message = create_human_message("Test message")
        assert get_sender_parent_id(message) is None

        set_sender_parent_id(message, "parent1")
        assert get_sender_parent_id(message) == "parent1"

    def test_receiver_parent_id(self) -> None:
        """Test receiver parent ID functions."""
        message = create_ai_message("Test response")
        assert get_receiver_parent_id(message) is None

        set_receiver_parent_id(message, "parent2")
        assert get_receiver_parent_id(message) == "parent2"

    def test_hierarchy_path(self) -> None:
        """Test hierarchy path functions."""
        message = create_human_message("Test message")
        assert get_hierarchy_path(message) is None

        path = ["agent1", "agent2", "agent3"]
        set_hierarchy_path(message, path)
        assert get_hierarchy_path(message) == path

        # Test that the original list is not modified
        path.append("agent4")
        assert get_hierarchy_path(message) == ["agent1", "agent2", "agent3"]

    def test_add_to_hierarchy_path(self) -> None:
        """Test adding to hierarchy path."""
        message = create_human_message("Test message")
        assert get_hierarchy_path(message) is None

        add_to_hierarchy_path(message, "agent1")
        assert get_hierarchy_path(message) == ["agent1"]

        add_to_hierarchy_path(message, "agent2")
        assert get_hierarchy_path(message) == ["agent1", "agent2"]

        # Test that duplicates are not added
        add_to_hierarchy_path(message, "agent1")
        assert get_hierarchy_path(message) == ["agent1", "agent2"]

    def test_is_hierarchical_message(self) -> None:
        """Test is_hierarchical_message function."""
        message = create_human_message("Test message")
        assert not is_hierarchical_message(message)

        set_sender_id(message, "agent1")
        assert not is_hierarchical_message(message)

        set_receiver_id(message, "agent2")
        assert is_hierarchical_message(message)

    def test_complete_hierarchy_info(self) -> None:
        """Test setting all hierarchy information."""
        message = create_human_message("Test message")

        # Set all hierarchy information
        set_sender_id(message, "child1")
        set_sender_parent_id(message, "parent1")
        set_receiver_id(message, "child2")
        set_receiver_parent_id(message, "parent2")
        set_hierarchy_path(message, ["parent1", "child1"])

        # Verify all information is correctly set
        assert get_sender_id(message) == "child1"
        assert get_sender_parent_id(message) == "parent1"
        assert get_receiver_id(message) == "child2"
        assert get_receiver_parent_id(message) == "parent2"
        assert get_hierarchy_path(message) == ["parent1", "child1"]
        assert is_hierarchical_message(message)

        # Add to the hierarchy path
        add_to_hierarchy_path(message, "child2")
        assert get_hierarchy_path(message) == ["parent1", "child1", "child2"]
