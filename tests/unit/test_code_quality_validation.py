"""Tests for code quality validation functions."""

from src.common_types.enums import VerificationStatus
from src.common_types.task_types import Task
from src.utils.validation import (
    validate_code_quality,
    validate_requirements_compliance,
    validate_task_execution_success,
)


def test_validate_code_quality_valid() -> None:
    """Test that valid code passes validation."""
    code = """
def hello_world():
    print("Hello, world!")
    return True
"""
    is_valid, error = validate_code_quality(code)
    assert is_valid
    assert error is None


def test_validate_code_quality_empty() -> None:
    """Test that empty code fails validation."""
    code = ""
    is_valid, error = validate_code_quality(code)
    assert not is_valid
    assert "Code cannot be empty" in error


def test_validate_code_quality_syntax_error() -> None:
    """Test that code with syntax errors fails validation."""
    code = """
def hello_world():
    print("Hello, world!"
    return True
"""
    is_valid, error = validate_code_quality(code)
    assert not is_valid
    assert "Syntax error in code" in error


def test_validate_code_quality_long_lines() -> None:
    """Test validation of code with lines that exceed the maximum length limit."""
    code = """
def example_function():
    # This is a very long line that exceeds the maximum line length limit
    return "This is a very long string that exceeds the maximum line length limit and triggers validation error"
"""
    is_valid, error = validate_code_quality(code)
    assert not is_valid
    assert "too long" in error


def test_validate_code_quality_todo_comments() -> None:
    """Test that code with TODO comments fails validation."""
    code = """
def hello_world():
    # TODO: Implement this function properly
    print("Hello, world!")
    return True
"""
    is_valid, error = validate_code_quality(code)
    assert not is_valid
    assert "TODO comment" in error


def test_validate_code_quality_excessive_nesting() -> None:
    """Test that code with excessive nesting fails validation."""
    code = """
def nested_function():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        print("Too much nesting!")
    return True
"""
    is_valid, error = validate_code_quality(code)
    assert not is_valid
    assert "excessive nesting" in error


def test_validate_code_quality_multiple_issues() -> None:
    """Test validation of code with multiple issues."""
    code = """
def example_function():
    # TODO: Fix this function
    if condition1:
        if condition2:
            if condition3:
                if condition4:
                    # This is a very long line that exceeds the limit
                    return "This is a very very very very very very very very very very very very very very very very very long string that exceeds the maximum line length limit"
"""  # noqa: E501
    is_valid, error = validate_code_quality(code)
    assert not is_valid
    assert "TODO comment" in error
    assert "excessive nesting" in error
    assert "too long" in error


def test_validate_requirements_compliance_valid() -> None:
    """Test that implementation meeting requirements passes validation."""
    task = Task(description="Create a function that calculates the sum of two numbers")
    implementation = """
def add_numbers(a, b):
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a + b
"""
    is_valid, error = validate_requirements_compliance(task, implementation)
    assert is_valid
    assert error is None


def test_validate_requirements_compliance_explicit_requirements() -> None:
    """Test validation with explicitly defined requirements in metadata."""
    task = Task(description="Create a math utility")
    task.metadata["requirements"] = [
        "Must implement addition function",
        "Should handle negative numbers",
        "Needs to return the result as a number",
    ]

    # Implementation meeting requirements - simplified to clearly match keywords
    implementation = """
def add_numbers(a, b):
    # This function implements addition
    # It handles negative numbers
    # It returns the result as a number
    return a + b
"""
    is_valid, error = validate_requirements_compliance(task, implementation)
    assert is_valid
    assert error is None

    # Implementation missing requirements
    bad_implementation = """
def multiply_numbers(a, b):
    \"\"\"Multiply two numbers together.\"\"\"
    return a * b
"""
    is_valid, error = validate_requirements_compliance(task, bad_implementation)
    assert not is_valid
    assert "requirements" in error


def test_validate_requirements_compliance_extracted_requirements() -> None:
    """Test validation with requirements extracted from description."""
    task = Task(description="Create a function that must calculate the sum and should handle negative numbers")

    # Implementation meeting requirements
    implementation = """
def add_numbers(a, b):
    \"\"\"Add two numbers together, handling negative values.\"\"\"
    return a + b
"""
    is_valid, error = validate_requirements_compliance(task, implementation)
    assert is_valid
    assert error is None

    # Implementation missing requirements
    bad_implementation = """
def process_data(data):
    \"\"\"Process data in some way.\"\"\"
    return data
"""
    is_valid, error = validate_requirements_compliance(task, bad_implementation)
    assert not is_valid
    assert "requirements" in error


def test_validate_task_execution_success_passed() -> None:
    """Test that a task with passed verification status passes validation."""
    task = Task(description="Test task")
    task.verification_status = VerificationStatus.PASSED

    is_valid, error = validate_task_execution_success(task)
    assert is_valid
    assert error is None


def test_validate_task_execution_success_failed() -> None:
    """Test that a task with failed verification status fails validation."""
    task = Task(description="Test task")
    task.verification_status = VerificationStatus.FAILED

    is_valid, error = validate_task_execution_success(task)
    assert not is_valid
    assert "verification failed" in error


def test_validate_task_execution_success_partial() -> None:
    """Test that a task with partial verification status fails validation."""
    task = Task(description="Test task")
    task.verification_status = VerificationStatus.PARTIAL

    is_valid, error = validate_task_execution_success(task)
    assert not is_valid
    assert "partially failed" in error


def test_validate_task_execution_success_pending() -> None:
    """Test that a task with pending verification status fails validation."""
    task = Task(description="Test task")
    task.verification_status = VerificationStatus.PENDING

    is_valid, error = validate_task_execution_success(task)
    assert not is_valid
    assert "not passed" in error


def test_validate_task_execution_success_with_error() -> None:
    """Test that a task with an error fails validation."""
    task = Task(description="Test task")
    task.verification_status = VerificationStatus.PASSED
    task.error = "Something went wrong"

    is_valid, error = validate_task_execution_success(task)
    assert not is_valid
    assert "reported an error" in error


def test_validate_task_execution_success_with_failed_criteria() -> None:
    """Test that a task with failed success criteria fails validation."""
    task = Task(description="Test task")
    task.verification_status = VerificationStatus.PASSED
    task.execution_metadata = {
        "success_criteria": {
            "code_quality": False,
            "test_coverage": True,
        },
    }

    is_valid, error = validate_task_execution_success(task)
    assert not is_valid
    assert "failed success criterion" in error


def test_validate_task_execution_success_with_failed_tests() -> None:
    """Test that a task with failed tests fails validation."""
    task = Task(description="Test task")
    task.verification_status = VerificationStatus.PASSED
    task.verification_details = {
        "test_results": {
            "test_function_1": True,
            "test_function_2": False,
        },
    }

    is_valid, error = validate_task_execution_success(task)
    assert not is_valid
    assert "failed tests" in error
