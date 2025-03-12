"""Tests for the Result class."""

import pytest

from src.common_types.error_types import AgentError
from src.common_types.result_types import Result


def test_result_initialization() -> None:
    """Test Result initialization."""
    # Test successful result
    result = Result(success=True, data="test_data", message="test_message")
    assert result.success is True
    assert result.data == "test_data"
    assert result.message == "test_message"
    assert result.error is None

    # Test error result
    error = ValueError("test_error")
    result = Result(success=False, error=error)
    assert result.success is False
    assert result.error == error
    assert result.data is None
    assert result.message is None


def test_result_bool_conversion() -> None:
    """Test Result bool conversion."""
    # Test successful result
    result = Result(success=True)
    assert bool(result) is True

    # Test error result
    result = Result(success=False)
    assert bool(result) is False


def test_result_string_conversion() -> None:
    """Test Result string conversion."""
    # Test successful result with message
    result = Result(success=True, message="test_message")
    assert str(result) == "Success: test_message"

    # Test successful result without message
    result = Result(success=True)
    assert str(result) == "Success: No message"

    # Test error result
    error = ValueError("test_error")
    result = Result(success=False, error=error)
    assert str(result) == f"Error: {error}"


def test_result_ok_factory() -> None:
    """Test Result.ok factory method."""
    # Test with data and message
    result = Result.ok(data="test_data", message="test_message")
    assert result.success is True
    assert result.data == "test_data"
    assert result.message == "test_message"
    assert result.error is None

    # Test with data only
    result = Result.ok(data="test_data")
    assert result.success is True
    assert result.data == "test_data"
    assert result.message is None
    assert result.error is None

    # Test with message only
    result = Result.ok(message="test_message")
    assert result.success is True
    assert result.data is None
    assert result.message == "test_message"
    assert result.error is None

    # Test with no arguments
    result = Result.ok()
    assert result.success is True
    assert result.data is None
    assert result.message is None
    assert result.error is None


def test_result_create_error_factory() -> None:
    """Test Result.create_error factory method."""
    # Test with Exception
    error = ValueError("test_error")
    result = Result.create_error(error)
    assert result.success is False
    assert result.error == error
    assert result.data is None
    assert result.message is None

    # Test with string message
    error = ValueError("test_error")
    result = Result.create_error(error, message="test_message")
    assert result.success is False
    assert result.error == error
    assert result.data is None
    assert result.message == "test_message"


def test_result_unwrap() -> None:
    """Test Result.unwrap method."""
    # Test successful result with data
    result = Result(success=True, data="test_data")
    assert result.unwrap() == "test_data"

    # Test successful result without data
    result = Result(success=True)
    with pytest.raises(AgentError):
        result.unwrap()

    # Test error result
    error = ValueError("test_error")
    result = Result(success=False, error=error)
    with pytest.raises(ValueError, match="Cannot unwrap unsuccessful result"):
        result.unwrap()


def test_result_map() -> None:
    """Test Result.map method."""
    # Test successful result with data
    result = Result(success=True, data=5, message="test_message")
    mapped_result = result.map(lambda x: x * 2)
    assert mapped_result.success is True
    assert mapped_result.data == 10
    assert mapped_result.message == "test_message"
    assert mapped_result.error is None

    # Test successful result without data
    result = Result(success=True, message="test_message")
    mapped_result = result.map(lambda x: x * 2)
    # Our updated implementation returns a new Result with None data
    assert mapped_result.success is True
    assert mapped_result.data is None
    assert mapped_result.message == "test_message"
    assert mapped_result.error is None

    # Test error result
    error = ValueError("test_error")
    result = Result(success=False, error=error, message="test_message")
    mapped_result = result.map(lambda x: x * 2)
    assert mapped_result.success is False
    assert mapped_result.data is None
    assert mapped_result.message == "test_message"
    assert mapped_result.error == error

    # Test map with exception - wrap in try/except since our implementation doesn't catch exceptions
    result = Result(success=True, data="not_a_number", message="test_message")
    try:
        mapped_result = result.map(lambda x: int(x))
        msg = "Expected ValueError was not raised"
        raise AssertionError(msg)
    except ValueError:
        # This is expected
        pass


def test_result_success_factory() -> None:
    """Test Result.success factory method."""
    # Test with data and message
    result = Result.success(data="test_data", message="test_message")
    assert result.success is True
    assert result.data == "test_data"
    assert result.message == "test_message"
    assert result.error is None

    # Test with data only
    result = Result.success(data="test_data")
    assert result.success is True
    assert result.data == "test_data"
    assert result.message is None
    assert result.error is None

    # Test with message only
    result = Result.success(message="test_message")
    assert result.success is True
    assert result.data is None
    assert result.message == "test_message"
    assert result.error is None

    # Test with no arguments
    result = Result.success()
    assert result.success is True
    assert result.data is None
    assert result.message is None
    assert result.error is None


def test_result_failure_factory() -> None:
    """Test Result.failure factory method."""
    # Test with error
    error = ValueError("test_error")
    result = Result.failure(error)
    assert result.success is False
    assert result.error == error
    assert result.data is None
    assert result.message is None

    # Test with error and message
    error = ValueError("test_error")
    result = Result.failure(error, message="test_message")
    assert result.success is False
    assert result.error == error
    assert result.data is None
    assert result.message == "test_message"


def test_result_with_complex_data() -> None:
    """Test Result with complex data types."""
    # Test with dictionary
    data = {"key": "value", "nested": {"key": "value"}}
    result = Result.ok(data=data)
    assert result.success is True
    assert result.data == data
    assert result.unwrap() == data

    # Test with list
    data = [1, 2, 3, [4, 5, 6]]
    result = Result.ok(data=data)
    assert result.success is True
    assert result.data == data
    assert result.unwrap() == data

    # Test with custom object
    class TestObject:
        def __init__(self, value: str) -> None:
            self.value = value

    obj = TestObject("test")
    result = Result.ok(data=obj)
    assert result.success is True
    assert result.data == obj
    assert result.unwrap() == obj
    assert result.unwrap().value == "test"


def test_result_chaining() -> None:
    """Test chaining Result operations."""
    # Test successful chain
    result = Result.ok(data=5)
    chained_result = result.map(lambda x: x * 2).map(lambda x: x + 1)
    assert chained_result.success is True
    assert chained_result.data == 11

    # Test chain breaking on error
    result = Result.ok(data=5)
    # Use a try-except block to handle the TypeError
    try:
        chained_result = result.map(lambda x: x * 2).map(lambda x: "a" + x)
        # This should fail with TypeError
        msg = "Expected TypeError was not raised"
        raise AssertionError(msg)
    except TypeError:
        # This is expected
        pass

    # Test chain starting with error
    result = Result.create_error(ValueError("test_error"))
    chained_result = result.map(lambda x: x * 2).map(lambda x: x + 1)
    assert chained_result.success is False
    assert isinstance(chained_result.error, ValueError)
