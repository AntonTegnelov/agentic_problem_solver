"""Test agent steps."""

from unittest.mock import MagicMock

import pytest

from src.agent.agent_types.agent_types import MockAgent
from src.agent.result import Result
from src.agent.state.base import AgentState
from src.common_types.enums import AgentStep
from src.exceptions import ConfigError
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
    result = Result(
        success=True,
        data="This is a long enough response to meet the length requirement. "
        "This is a comprehensive understanding of the problem with key insights and analysis.",
        error=None,
    )
    assert validate_step_result(AgentStep.UNDERSTAND, result, state) is None

    # Test planning step
    state.current_step = AgentStep.PLAN
    result = Result(
        success=True,
        data="This is a long enough response to meet the validation requirement. "
        "This includes key insights and comprehensive analysis of the requirements.",
        error=None,
    )
    assert validate_step_result(AgentStep.PLAN, result, state) is None

    # Test verification step
    state.current_step = AgentStep.VERIFY
    result = Result(
        success=True,
        data=True,  # Verification result must be boolean
        error=None,
    )
    assert validate_step_result(AgentStep.VERIFY, result, state) is None

    # Test failed result
    result = Result(success=False, data=None, error="Test error")
    with pytest.raises(ConfigError, match="Step failed: Test error"):
        validate_step_result(AgentStep.VERIFY, result, state)

    # Test short response
    result = Result(success=True, data="Too short", error=None)
    with pytest.raises(ConfigError, match="Understanding is too brief"):
        validate_step_result(AgentStep.UNDERSTAND, result, state)


def test_execute_step_with_retry() -> None:
    """Test step execution with retry mechanism."""
    state = AgentState()
    state.current_step = AgentStep.UNDERSTAND

    # Create a mock agent
    mock_agent = MockAgent("test_agent", ["test"])
    state.register_agent("test_agent", mock_agent)

    # Test successful execution
    success_result = Result(
        success=True,
        data=(
            "Detailed understanding of the problem with sufficient content to pass validation. "
            "This includes key insights and comprehensive analysis of the requirements."
        ),
        error=None,
    )
    mock_agent.process = MagicMock(return_value=success_result)

    result = execute_step_with_retry(state, state.current_step)
    assert result.success
    assert result.data == success_result.data

    # Test failure with retry
    error_result = Result(success=False, data=None, error="Test error")
    side_effects = [error_result, error_result, success_result]
    mock_agent.process = MagicMock(side_effect=side_effects)

    result = execute_step_with_retry(state, state.current_step)
    assert result.success, f"Expected success=True, got {result}"
    assert result.data == success_result.data

    # Test failure with max retries exceeded
    mock_agent.process = MagicMock(return_value=error_result)

    result = execute_step_with_retry(state, state.current_step)
    assert not result.success
    assert result.error == "Test error"


def test_get_next_step() -> None:
    """Test getting next step in sequence."""
    # Test progression
    assert get_next_step(AgentStep.UNDERSTAND) == AgentStep.PLAN
    assert get_next_step(AgentStep.PLAN) == AgentStep.EXECUTE
    assert get_next_step(AgentStep.EXECUTE) == AgentStep.VERIFY
    assert get_next_step(AgentStep.VERIFY) == AgentStep.UNDERSTAND

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


def create_understanding_result(content: str) -> Result:
    """Create a test result for the understanding step."""
    result = Result(success=True, data=content, error=None)
    validate_step_result(AgentStep.UNDERSTAND, result)
    return result


def create_plan_result(content: str) -> Result:
    """Create a test result for the plan step."""
    result = Result(success=True, data=content, error=None)
    validate_step_result(AgentStep.PLAN, result)
    return result


def create_verify_result(_content: str) -> Result:
    """Create verify step result.

    Args:
        _content: Content to verify (unused).

    Returns:
        Verification result.

    """
    return Result(success=True, data="Verified", error="")
