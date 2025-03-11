"""Tests for common message types."""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.common_types.message_types import (
    AIMessage as ImportedAIMessage,
)
from src.common_types.message_types import (
    CriteriaDict,
    CriteriaValue,
    MessageValue,
)
from src.common_types.message_types import (
    HumanMessage as ImportedHumanMessage,
)
from src.common_types.message_types import (
    SystemMessage as ImportedSystemMessage,
)
from src.common_types.message_types import (
    ToolMessage as ImportedToolMessage,
)


def test_message_imports() -> None:
    """Test that message imports are correct."""
    # Test that imported message types match the original types
    assert ImportedAIMessage is AIMessage
    assert ImportedHumanMessage is HumanMessage
    assert ImportedSystemMessage is SystemMessage
    assert ImportedToolMessage is ToolMessage


def test_message_value_type() -> None:
    """Test MessageValue type annotation."""
    # Test string value
    value: MessageValue = "test string"
    assert isinstance(value, str)

    # Test numeric values
    value = 42
    assert isinstance(value, int)
    value = 3.14
    assert isinstance(value, float)

    # Test boolean value
    value = True
    assert isinstance(value, bool)

    # Test dictionary value
    value = {"key": "value", "number": 42}
    assert isinstance(value, dict)

    # Test list value
    value = [1, 2, 3, "test"]
    assert isinstance(value, list)

    # Test None value
    value = None
    assert value is None


def test_criteria_value_type() -> None:
    """Test CriteriaValue type annotation."""
    # Test string value
    value: CriteriaValue = "test string"
    assert isinstance(value, str)

    # Test numeric values
    value = 42
    assert isinstance(value, int)
    value = 3.14
    assert isinstance(value, float)

    # Test boolean value
    value = True
    assert isinstance(value, bool)

    # Test None value
    value = None
    assert value is None


def test_criteria_dict_type() -> None:
    """Test CriteriaDict type annotation."""
    # Test empty dictionary
    criteria: CriteriaDict = {}
    assert isinstance(criteria, dict)

    # Test dictionary with string values
    criteria = {"name": "test", "type": "message"}
    assert isinstance(criteria, dict)
    assert all(isinstance(k, str) for k in criteria)
    assert all(isinstance(v, (str, int, float, bool, type(None))) for v in criteria.values())

    # Test dictionary with mixed values
    criteria = {"name": "test", "priority": 1, "active": True, "value": None}
    assert isinstance(criteria, dict)
    assert all(isinstance(k, str) for k in criteria)
    assert all(isinstance(v, (str, int, float, bool, type(None))) for v in criteria.values())
