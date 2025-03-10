"""Test agent step processing functionality."""

from unittest.mock import MagicMock

import pytest

from src.agent.agent_types.agent_types import StepResult
from src.agent.state.base import AgentState
from src.common_types.enums import AgentStep
from src.exceptions import ConfigError, RetryError
from src.prompts import (
    execute_step_with_retry,
    get_next_step,
    get_retry_prompt,
    get_step_description,
    get_step_prompt,
    validate_step_result,
)


def test_get_step_prompt() -> None:
    """Test getting step prompts."""
    state = AgentState(agent_id="test-id")

    # Test understanding step
    state.current_step = AgentStep.UNDERSTAND
    prompt = get_step_prompt(state)
    assert isinstance(prompt, str)
    assert len(prompt) > 0

    # Test planning step
    state.current_step = AgentStep.PLAN
    prompt = get_step_prompt(state)
    assert isinstance(prompt, str)
    assert len(prompt) > 0

    # Test verification step
    state.current_step = AgentStep.VERIFY
    prompt = get_step_prompt(state)
    assert isinstance(prompt, str)
    assert len(prompt) > 0

    # Test invalid step
    state.current_step = "invalid_step"  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="Invalid step: invalid_step"):
        get_step_prompt(state)


def test_get_retry_prompt() -> None:
    """Test getting retry prompts."""
    state = AgentState(agent_id="test-id")
    error = "Test error message"

    # Test understanding step
    state.current_step = AgentStep.UNDERSTAND
    prompt = get_retry_prompt(state, error)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert error in prompt

    # Test planning step
    state.current_step = AgentStep.PLAN
    prompt = get_retry_prompt(state, error)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert error in prompt

    # Test verification step
    state.current_step = AgentStep.VERIFY
    prompt = get_retry_prompt(state, error)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert error in prompt

    # Test invalid step
    state.current_step = "invalid_step"  # type: ignore[assignment]
    with pytest.raises(ConfigError, match="Invalid step: invalid_step"):
        get_retry_prompt(state, error)


def test_validate_step_result() -> None:
    """Test step result validation."""
    state = AgentState(agent_id="test-id")

    # Test understanding step
    state.current_step = AgentStep.UNDERSTAND
    result = StepResult(
        success=True,
        data="This is a long enough response to meet the length requirement. "
        "This is a comprehensive understanding of the problem with key insights and analysis.",
        error="",
    )
    assert validate_step_result(state, AgentStep.UNDERSTAND, result) is None

    # Test planning step
    state.current_step = AgentStep.PLAN
    result = StepResult(
        success=True,
        data="This is a long enough response to meet the validation requirement. "
        "This includes key insights and comprehensive analysis of the requirements.",
        error="",
    )
    assert validate_step_result(state, AgentStep.PLAN, result) is None

    # Test verification step
    state.current_step = AgentStep.VERIFY
    result = StepResult(
        success=True,
        data=(
            "This is a long enough response that includes a thorough "
            "analysis of the problem domain and identification of key "
            "requirements and constraints."
        ),
        error="",
    )
    assert validate_step_result(state, AgentStep.VERIFY, result) is None

    # Test failed result
    result = StepResult(success=False, data=None, error="Test error")
    with pytest.raises(ConfigError, match="Step failed: Test error"):
        validate_step_result(state, AgentStep.VERIFY, result)

    # Test short response
    result = StepResult(success=True, data="Too short", error="")
    with pytest.raises(ConfigError, match="Verification is too brief"):
        validate_step_result(state, AgentStep.VERIFY, result)


def test_execute_step_with_retry() -> None:
    """Test step execution with retry mechanism."""
    state = AgentState()
    state.current_step = AgentStep.UNDERSTAND

    # Test successful execution
    success_result = StepResult(
        success=True,
        data=(
            "Detailed understanding of the problem with sufficient content to pass validation. "
            "This includes key insights and comprehensive analysis of the requirements."
        ),
        error="",
    )
    mock_execute = MagicMock(return_value=success_result)

    result = execute_step_with_retry(state, mock_execute)
    assert result.success
    assert mock_execute.call_count == 1

    # Test execution with validation failure then success
    brief_result = StepResult(success=True, data="Too brief", error="")
    good_result = StepResult(
        success=True,
        data=(
            "Detailed understanding with sufficient content to pass validation. "
            "This includes a thorough analysis of the problem domain and identification "
            "of key requirements and constraints."
        ),
        error="",
    )
    mock_execute_retry = MagicMock(side_effect=[brief_result, good_result])

    result = execute_step_with_retry(state, mock_execute_retry, max_retries=1)
    assert result.success
    assert mock_execute_retry.call_count == 2

    # Test execution with persistent failure
    always_brief = StepResult(success=True, data="Always too brief", error="")
    mock_execute_fail = MagicMock(return_value=always_brief)

    with pytest.raises(RetryError):
        execute_step_with_retry(state, mock_execute_fail, max_retries=2)
    assert mock_execute_fail.call_count == 3  # Initial + 2 retries

    # Test execution with exception
    mock_execute_exception = MagicMock(side_effect=Exception("Test exception"))

    with pytest.raises(RetryError):
        execute_step_with_retry(state, mock_execute_exception, max_retries=1)
    assert mock_execute_exception.call_count == 2  # Initial + 1 retry


def test_get_next_step() -> None:
    """Test getting next step in sequence."""
    # Test progression
    assert get_next_step(AgentStep.UNDERSTAND) == AgentStep.PLAN
    assert get_next_step(AgentStep.PLAN) == AgentStep.VERIFY
    assert get_next_step(AgentStep.VERIFY) is None

    # Test invalid step
    with pytest.raises(ConfigError, match="Invalid step: invalid_step"):
        get_next_step("invalid_step")  # type: ignore[arg-type]


def test_get_step_description() -> None:
    """Test getting step descriptions."""
    # Test valid steps
    assert isinstance(get_step_description(AgentStep.UNDERSTAND), str)
    assert isinstance(get_step_description(AgentStep.PLAN), str)
    assert isinstance(get_step_description(AgentStep.VERIFY), str)

    # Test invalid step
    with pytest.raises(ConfigError, match="Invalid step: invalid_step"):
        get_step_description("invalid_step")  # type: ignore[arg-type]

    # Test understanding step
    create_understanding_result(
        "This is a long enough response to meet the length requirement. "
        "This is a comprehensive understanding of the problem with key "
        "insights and analysis.",
    )

    # Test plan step
    create_plan_result(
        "This is a long enough response to meet the validation "
        "requirement. This includes key insights and comprehensive "
        "analysis of the requirements.",
    )

    # Test verify step
    create_verify_result(
        "This is a long enough response that includes a thorough "
        "analysis of the problem domain and identification of key "
        "requirements and constraints.",
    )


def create_understanding_result(content: str) -> StepResult:
    """Create a test result for the understanding step."""
    result = StepResult(success=True, data=content, error="")
    validate_step_result(AgentStep.UNDERSTAND, result)
    return result


def create_plan_result(content: str) -> StepResult:
    """Create a test result for the plan step."""
    result = StepResult(success=True, data=content, error="")
    validate_step_result(AgentStep.PLAN, result)
    return result


def create_verify_result(content: str) -> StepResult:
    """Create a test result for the verify step."""
    result = StepResult(success=True, data=content, error="")
    validate_step_result(AgentStep.VERIFY, result)
    return result
