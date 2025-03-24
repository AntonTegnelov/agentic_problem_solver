import json
import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.utils.log_utils import (
    DelegationInfo,
    HierarchicalDelegationInfo,
    log_delegation_decision,
    log_hierarchical_delegation,
    render_task_tree,
    setup_logging,
)


def test_setup_logging_with_file() -> None:
    """Test setup_logging with a log file."""
    temp_dir = tempfile.mkdtemp()
    try:
        log_file = os.path.join(temp_dir, "test.log")
        setup_logging(log_file=log_file, verbose=True)
        logger = logging.getLogger()

        # Verify log file was created
        assert os.path.exists(log_file)

        # Verify handlers were added
        assert len(logger.handlers) >= 2
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

        # Verify debug level was set
        assert logger.level == logging.DEBUG

        # Clean up handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    finally:
        # Clean up temp directory
        try:
            for handler in logging.getLogger().handlers[:]:
                handler.close()
            os.remove(log_file)
            os.rmdir(temp_dir)
        except OSError:
            pass


def test_setup_logging_file_creation() -> None:
    """Test setup_logging creates parent directories for log file."""
    temp_dir = tempfile.mkdtemp()
    try:
        nested_path = os.path.join(temp_dir, "logs", "nested", "test.log")
        setup_logging(log_file=nested_path)

        # Verify directory structure was created
        assert os.path.exists(os.path.dirname(nested_path))
        assert os.path.exists(nested_path)

        # Clean up handlers
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
    finally:
        # Clean up temp directory
        try:
            for handler in logging.getLogger().handlers[:]:
                handler.close()
            os.remove(nested_path)
            os.rmdir(os.path.dirname(nested_path))
            os.rmdir(os.path.dirname(os.path.dirname(nested_path)))
            os.rmdir(temp_dir)
        except OSError:
            pass


def test_log_delegation_decision_with_additional_info() -> None:
    """Test log_delegation_decision with additional info."""
    mock_logger = MagicMock()
    info = DelegationInfo(
        source_agent_id="source",
        target_agent_id="target",
        task="test task",
        reason="test reason",
        additional_info={"key": "value"},
    )

    log_delegation_decision(mock_logger, info)

    # Verify logger was called with correct data
    mock_logger.info.assert_called_once()
    log_data = json.loads(mock_logger.info.call_args[0][1])
    assert log_data["additional_info"] == {"key": "value"}


def test_log_hierarchical_delegation_success() -> None:
    """Test log_hierarchical_delegation for successful case."""
    info = HierarchicalDelegationInfo(
        source_agent_id="source",
        parent_task_id="parent_id",
        parent_task="parent task",
        total_subtasks=5,
        successful_delegations=3,
        failed_delegations=2,
    )

    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_hierarchical_delegation(info)

        # Verify success message was logged with formatted string
        mock_logger.info.assert_called_once_with(
            "Hierarchical delegation from %s: %d subtasks delegated (%d successful, %d failed)",
            "source",
            5,
            3,
            2,
        )


def test_log_hierarchical_delegation_error() -> None:
    """Test log_hierarchical_delegation for error case."""
    info = HierarchicalDelegationInfo(
        source_agent_id="source",
        error="test error",
    )

    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_hierarchical_delegation(info)

        # Verify error message was logged with formatted string
        mock_logger.error.assert_called_once_with(
            "Hierarchical delegation from %s failed: %s",
            "source",
            "test error",
        )


def test_log_hierarchical_delegation_serialization_error() -> None:
    """Test log_hierarchical_delegation handles serialization errors."""

    # Create an object that can't be JSON serialized
    class UnserializableObject:
        pass

    info = HierarchicalDelegationInfo(
        source_agent_id="source",
        additional_info={"unserializable": UnserializableObject()},
    )

    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_hierarchical_delegation(info)

        # Verify partial details were logged with error
        mock_logger.debug.assert_called_once()
        assert "serialization error" in mock_logger.debug.call_args[0][0]


def test_render_task_tree_empty() -> None:
    """Test render_task_tree with empty subtasks."""
    result = render_task_tree("Parent Task", [])
    assert "Parent Task" in result
    assert "(No subtasks)" in result


def test_render_task_tree_with_task_objects() -> None:
    """Test render_task_tree with Task objects."""

    class MockTask:
        def __init__(self, description, task_id, complexity) -> None:
            self.description = description
            self.task_id = task_id
            self.complexity = MagicMock(name=complexity)
            self.subtasks = []

    subtasks = [
        MockTask("Task 1", "id1", "HIGH"),
        MockTask("Task 2", "id2", "LOW"),
    ]

    result = render_task_tree("Parent", subtasks)

    assert "Task 1" in result
    assert "Task 2" in result
    assert "HIGH" in result
    assert "LOW" in result
    assert "id1" in result
    assert "id2" in result


def test_render_task_tree_with_dictionaries() -> None:
    """Test render_task_tree with dictionary subtasks."""
    subtasks = [
        {
            "description": "Dict Task 1",
            "id": "dict1",
            "complexity": "MEDIUM",
            "subtasks": [],
        },
        {
            "description": "Dict Task 2",
            "id": "dict2",
            "complexity": "HIGH",
            "subtasks": [],
        },
    ]

    result = render_task_tree("Parent", subtasks)

    assert "Dict Task 1" in result
    assert "Dict Task 2" in result
    assert "MEDIUM" in result
    assert "HIGH" in result
    assert "dict1" in result
    assert "dict2" in result


def test_render_task_tree_nested() -> None:
    """Test render_task_tree with nested subtasks."""
    nested_task = {
        "description": "Nested Task",
        "id": "nested1",
        "complexity": "LOW",
        "subtasks": [],
    }

    parent_task = {
        "description": "Parent Task",
        "id": "parent1",
        "complexity": "HIGH",
        "subtasks": [nested_task],
    }

    result = render_task_tree("Root", [parent_task])

    assert "Root" in result
    assert "Parent Task" in result
    assert "Nested Task" in result
    assert "HIGH" in result
    assert "LOW" in result


def test_render_task_tree_max_depth() -> None:
    """Test render_task_tree respects max_depth parameter."""
    deeply_nested = {
        "description": "Deep Task",
        "id": "deep1",
        "subtasks": [],
    }

    nested = {
        "description": "Nested Task",
        "id": "nested1",
        "subtasks": [deeply_nested],
    }

    parent = {
        "description": "Parent Task",
        "id": "parent1",
        "subtasks": [nested],
    }

    # Set max_depth to 1 to prevent rendering nested tasks
    result = render_task_tree("Root", [parent], max_depth=1)

    assert "Root" in result
    assert "Parent Task" in result
    assert "Nested Task" not in result
    assert "Deep Task" not in result


def test_render_task_tree_long_description() -> None:
    """Test render_task_tree handles long descriptions."""
    long_description = "A" * 100
    subtasks = [{"description": long_description}]

    result = render_task_tree("Parent", subtasks)

    # Verify description was truncated
    assert "..." in result
    assert len(result.split("\n")[1]) < len(long_description)
