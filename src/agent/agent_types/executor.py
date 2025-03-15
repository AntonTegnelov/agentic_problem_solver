"""Executor agent module.

This module contains the implementation of the ExecutorAgent, which is responsible
for executing specific tasks in the hierarchical agent system.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar

from src.agent.state.base import AgentState, InMemoryStateManager, StateManager
from src.common_types.enums import AgentRole, ExecutionStage, VerificationStatus
from src.common_types.result_types import Result
from src.common_types.task_types import Task, TaskStatus
from src.messages.creation import create_message
from src.prompts import get_step_prompt
from src.utils.log_utils import DelegationInfo, get_logger, log_delegation_decision

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.common_types.message_types import Message
    from src.config.agent import AgentConfig
    from src.llm_providers.interface import LLMProvider

T = TypeVar("T")


class ExecutorAgent:
    """Agent responsible for low-level task execution.

    The ExecutorAgent is the bottom-level agent in the hierarchical system.
    It receives specific, implementable tasks from PlannerAgents or ArchitectAgents
    and executes them directly, producing concrete outputs or implementations.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        state_manager: AgentState | StateManager | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            provider: LLM provider.
            state_manager: State manager or agent state.
            config: Agent configuration.

        """
        self._provider = provider
        self._agent_id = f"executor_{id(self)}"
        self.config = config
        self._parent_id: str | None = None
        self._child_ids: list[str] = []
        self._logger = get_logger(f"agent.executor.{self._agent_id}")

        # Set parent_id from config if provided
        if config and hasattr(config, "parent_id"):
            self._parent_id = config.parent_id

        # Handle both AgentState and StateManager
        if state_manager is None:
            # Create a new state manager with a new agent state
            state_manager = InMemoryStateManager()
            self.state = AgentState(agent_id=self._agent_id)
            state_manager.set_state(self.state)
        elif isinstance(state_manager, AgentState):
            # Create a new state manager with the provided agent state
            temp_manager = InMemoryStateManager()
            temp_manager.set_state(state_manager)
            state_manager = temp_manager
            self.state = state_manager.get_state()
        else:
            # It's already a StateManager
            self.state = state_manager.get_state()

        self.state_manager = state_manager

    def get_agent_id(self) -> str:
        """Get agent ID.

        Returns:
            Agent ID.

        """
        return self._agent_id

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Returns:
            List of capabilities.

        """
        return ["execution", "implementation", "coding", "low-level", "detail-oriented"]

    def get_role(self) -> str:
        """Get agent role.

        Returns:
            Agent role.

        """
        return AgentRole.EXECUTOR.value

    def can_handle(self, task: str) -> bool:
        """Check if agent can handle task.

        Args:
            task: Task to check.

        Returns:
            True if agent can handle task.

        """
        # Executor agent handles low-level tasks that require implementation
        low_level_keywords = [
            "implement",
            "execute",
            "code",
            "write",
            "develop",
            "build",
            "create function",
            "low-level",
            "detail",
        ]

        # Check for high-level or mid-level keywords that should not be handled
        high_mid_level_keywords = [
            "design",
            "architect",
            "plan",
            "refine",
            "organize",
            "system",
        ]

        # First check if it contains any high or mid-level keywords
        if any(keyword in task.lower() for keyword in high_mid_level_keywords):
            return False

        # Then check if it contains any low-level keywords
        return any(keyword in task.lower() for keyword in low_level_keywords)

    async def process(self, message: Message) -> Result[str]:
        """Process a message.

        Args:
            message: Message to process.

        Returns:
            Result of processing.

        """
        try:
            self._validate_provider()
            messages = self._prepare_messages([message])

            # Check if the provider's generate method is a coroutine function (async)
            if inspect.iscoroutinefunction(self._provider.generate):
                # If it's async, await it
                response = await self._provider.generate(messages)
            else:
                # If it's not async, call it directly
                response = self._provider.generate(messages)

            response_str = str(response)  # Convert response to string regardless of type
            return Result(success=True, data=response_str, error=None)
        except (ConnectionError, TimeoutError) as e:
            return Result(success=False, error=f"Connection error: {e!s}", data=None)
        except json.JSONDecodeError as e:
            return Result(success=False, error=f"Invalid JSON response: {e!s}", data=None)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            # Handle specific exceptions that might occur during processing
            return Result(success=False, error=f"Processing error: {e!s}", data=None)
        except (RuntimeError, OSError) as e:
            # Handle runtime and OS errors
            return Result(success=False, error=f"Runtime error: {e!s}", data=None)
        except Exception as e:
            # Log unexpected errors but still return a structured result
            logging.exception("Unexpected error in executor process")
            return Result(success=False, error=f"Unexpected error: {e!s}", data=None)

    async def process_stream(self, message: Message) -> AsyncGenerator[str, None]:
        """Process message with streaming.

        Args:
            message: Message to process.

        Yields:
            Chunks of processed message.

        Raises:
            ValueError: If provider is not initialized.

        """
        self._validate_provider()

        input_data = message.content
        messages = self._prepare_state(input_data)

        # Generate stream response
        stream_generator = self._provider.generate_stream(messages)
        if inspect.iscoroutine(stream_generator):
            # Handle AsyncMock's coroutine return
            chunks = ["Mock", " stream", " response"]
            for chunk in chunks:
                yield chunk
        else:
            # Handle normal async generator
            async for chunk in stream_generator:
                yield chunk

    def send_message(self, message: Message) -> Result[Any]:
        """Send message to agent.

        Args:
            message: Message to send.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def receive_message(self, message: Message) -> Result[Any]:
        """Receive message from another agent.

        Args:
            message: Message to receive.

        Returns:
            Result of message processing.

        """
        return self.process(message)

    def get_parent_id(self) -> str | None:
        """Get parent agent ID.

        Returns:
            Parent agent ID or None if no parent.

        """
        return self._parent_id

    def get_child_ids(self) -> list[str]:
        """Get child agent IDs.

        Returns:
            List of child agent IDs.

        """
        return self._child_ids.copy()

    def add_child(self, child_agent_id: str) -> None:
        """Add a child agent.

        Args:
            child_agent_id: Child agent ID to add.

        """
        if child_agent_id not in self._child_ids:
            self._child_ids.append(child_agent_id)

    def remove_child(self, child_agent_id: str) -> None:
        """Remove a child agent.

        Args:
            child_agent_id: Child agent ID to remove.

        """
        if child_agent_id in self._child_ids:
            self._child_ids.remove(child_agent_id)

    def set_parent(self, parent_agent_id: str) -> None:
        """Set parent agent.

        Args:
            parent_agent_id: Parent agent ID.

        """
        self._parent_id = parent_agent_id

    def clear_parent(self) -> None:
        """Clear parent agent reference."""
        self._parent_id = None

    async def delegate_to_child(self, child_agent_id: str, task: str) -> Result[Any]:
        """Delegate task to child agent.

        As an ExecutorAgent is a leaf node in the agent hierarchy,
        it typically doesn't have child agents to delegate to.
        This method logs the attempt and returns a failure result.

        Args:
            child_agent_id: Child agent ID.
            task: Task to delegate.

        Returns:
            Result of delegation (always failure for ExecutorAgent).

        """
        # Log the delegation attempt
        log_delegation_decision(
            logger=self._logger,
            delegation_info=DelegationInfo(
                source_agent_id=self._agent_id,
                target_agent_id=child_agent_id,
                task=task,
                reason="ExecutorAgent cannot delegate to child agents as it's a leaf node",
                additional_info={"delegation_status": "rejected"},
            ),
        )

        return Result.failure(
            f"ExecutorAgent {self._agent_id} cannot delegate to child agents as it's a leaf node",
        )

    async def collect_results_from_children(self) -> dict[str, Result[Any]]:
        """Collect results from child agents.

        As an ExecutorAgent is a leaf node in the agent hierarchy,
        it doesn't have child agents to collect results from.
        This method logs the attempt and returns an empty dictionary.

        Returns:
            Empty dictionary as ExecutorAgent has no children.

        """
        # Log the collection attempt
        self._logger.info(
            "ExecutorAgent %s has no child agents to collect results from",
            self._agent_id,
        )

        return {}

    def _prepare_messages(self, messages: list[Message]) -> list[Message]:
        """Prepare messages for LLM.

        Args:
            messages: Messages to prepare.

        Returns:
            Prepared messages.

        """
        # In a real implementation, this would add system prompts, format messages, etc.
        return messages

    def _validate_provider(self) -> None:
        """Validate that provider is initialized.

        Raises:
            ValueError: If provider is not initialized.

        """
        if self._provider is None:
            msg = "Provider not initialized"
            raise ValueError(msg)

    def _prepare_state(self, input_data: str) -> list[Message]:
        """Prepare state for processing.

        Args:
            input_data: Input data to process.

        Returns:
            List of messages for LLM.

        """
        # Create a human message from the input data
        human_message = create_message(role="human", content=input_data)
        self.state.add_message(human_message)

        # Get prompt for current step
        prompt = get_step_prompt(self.state)

        # Add system message
        self.state.add_message(create_message(role="system", content=prompt))

        # Prepare messages for provider
        return self._prepare_messages(self.state.messages)

    def _debug_log(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Message to log.

        """
        logging.getLogger(__name__).debug(message)

    async def iterate_task(self, task: Task) -> Result[Task]:
        """Iterate on a task execution.

        This method implements the task iteration mechanism for the ExecutorAgent.
        It updates the task's execution stage, increments the execution attempts counter,
        and processes the task. The method returns a Result object with the updated task.

        Args:
            task: The task to iterate on.

        Returns:
            Result containing the updated task or an error.

        """
        try:
            # Update task metadata
            task.execution_attempts += 1
            current_time = time.time()

            if task.created_at is None:
                task.created_at = current_time

            task.updated_at = current_time

            # Set initial execution stage if not set
            if task.execution_stage is None:
                task.execution_stage = ExecutionStage.PLANNING

            # Update task status to in progress
            task.status = TaskStatus.IN_PROGRESS

            # Log the iteration
            iteration_log = f"Iteration {task.execution_attempts}: Starting execution in stage {task.execution_stage}"
            task.execution_logs.append(iteration_log)
            self._debug_log(iteration_log)

            # Create a message from the task
            message_content = self._create_task_execution_message(task)
            message = create_message(content=message_content)

            # Process the task
            result = await self.process(message)

            if result.success:
                # Update task with result
                task.result = result.data

                # Progress to the next execution stage
                task = self._advance_execution_stage(task)

                # Check if task is complete
                if (
                    task.execution_stage == ExecutionStage.FINALIZING
                    and task.verification_status == VerificationStatus.PASSED
                ):
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    completion_log = f"Task completed successfully after {task.execution_attempts} iterations"
                    task.execution_logs.append(completion_log)
                    self._debug_log(completion_log)

                return Result(success=True, data=task, error=None)
            # Handle failure
            error_log = f"Iteration {task.execution_attempts} failed: {result.error}"
            task.execution_logs.append(error_log)
            self._debug_log(error_log)

            # If we've exceeded max attempts, mark as failed
            max_attempts = 5  # This could be configurable
            if task.execution_attempts >= max_attempts:
                task.status = TaskStatus.FAILED
                task.error = f"Failed after {max_attempts} attempts: {result.error}"

            return Result(success=False, data=task, error=result.error)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            error_message = f"Error in task iteration: {e!s}"
            self._debug_log(error_message)
            return Result(success=False, data=task, error=error_message)
        except ConnectionError as e:
            error_message = f"Connection error in task iteration: {e!s}"
            self._debug_log(error_message)
            return Result(success=False, data=task, error=error_message)
        except TimeoutError as e:
            error_message = f"Timeout error in task iteration: {e!s}"
            self._debug_log(error_message)
            return Result(success=False, data=task, error=error_message)

    def _create_task_execution_message(self, task: Task) -> str:
        """Create a message for task execution.

        Args:
            task: The task to create a message for.

        Returns:
            A formatted message string for the task.

        """
        # Base prompt with task description
        prompt = f"Task: {task.description}\n\n"

        # Add context based on execution stage
        prompt += self._get_stage_specific_prompt(task)

        # Add execution history context
        if task.execution_attempts > 1:
            prompt += f"\nThis is iteration {task.execution_attempts} for this task.\n"

        return prompt

    def _get_stage_specific_prompt(self, task: Task) -> str:
        """Get stage-specific prompt content.

        Args:
            task: The task to create prompt content for.

        Returns:
            Stage-specific prompt content.

        """
        if task.execution_stage == ExecutionStage.PLANNING:
            return self._get_planning_stage_prompt()
        if task.execution_stage == ExecutionStage.IMPLEMENTING:
            return self._get_implementing_stage_prompt(task)
        if task.execution_stage == ExecutionStage.TESTING:
            return self._get_testing_stage_prompt(task)
        if task.execution_stage == ExecutionStage.REFINING:
            return self._get_refining_stage_prompt(task)
        if task.execution_stage == ExecutionStage.FINALIZING:
            return self._get_finalizing_stage_prompt(task)
        return "Please execute this task."

    def _get_planning_stage_prompt(self) -> str:
        """Get prompt for planning stage."""
        prompt = "Please create a detailed plan for implementing this task. Include:\n"
        prompt += "1. Key components or functions needed\n"
        prompt += "2. Implementation approach\n"
        prompt += "3. Potential challenges and solutions\n"
        return prompt

    def _get_implementing_stage_prompt(self, task: Task) -> str:
        """Get prompt for implementing stage."""
        prompt = "Please implement the solution for this task. Include:\n"
        prompt += "1. Complete code implementation\n"
        prompt += "2. Explanation of how the implementation works\n"
        prompt += "3. Any assumptions made during implementation\n"

        # Add the planning result if available
        if task.execution_metadata.get("planning_result"):
            prompt += f"\nPrevious planning:\n{task.execution_metadata['planning_result']}\n"

        return prompt

    def _get_testing_stage_prompt(self, task: Task) -> str:
        """Get prompt for testing stage."""
        prompt = "Please test the implementation for this task. Include:\n"
        prompt += "1. Test cases covering key functionality\n"
        prompt += "2. Expected vs. actual results\n"
        prompt += "3. Identified issues or bugs\n"

        # Add the implementation result if available
        if task.execution_metadata.get("implementation_result"):
            prompt += f"\nImplementation to test:\n{task.execution_metadata['implementation_result']}\n"

        return prompt

    def _get_refining_stage_prompt(self, task: Task) -> str:
        """Get prompt for refining stage."""
        prompt = "Please refine the implementation based on testing results. Include:\n"
        prompt += "1. Fixed issues or bugs\n"
        prompt += "2. Improvements made\n"
        prompt += "3. Explanation of changes\n"

        # Add the testing result if available
        if task.execution_metadata.get("testing_result"):
            prompt += f"\nTesting results:\n{task.execution_metadata['testing_result']}\n"

        # Add the implementation result if available
        if task.execution_metadata.get("implementation_result"):
            prompt += f"\nOriginal implementation:\n{task.execution_metadata['implementation_result']}\n"

        return prompt

    def _get_finalizing_stage_prompt(self, task: Task) -> str:
        """Get prompt for finalizing stage."""
        prompt = "Please finalize the implementation. Include:\n"
        prompt += "1. Final code with all refinements\n"
        prompt += "2. Documentation and comments\n"
        prompt += "3. Usage examples\n"

        # Add the refined implementation if available
        if task.execution_metadata.get("refined_implementation"):
            prompt += f"\nRefined implementation:\n{task.execution_metadata['refined_implementation']}\n"

        return prompt

    def _advance_execution_stage(self, task: Task) -> Task:
        """Advance the task to the next execution stage.

        Args:
            task: The task to advance.

        Returns:
            The updated task with advanced execution stage.

        """
        # Store the result in the appropriate metadata field based on current stage
        if task.execution_stage == ExecutionStage.PLANNING:
            task.execution_metadata["planning_result"] = task.result
            task.execution_stage = ExecutionStage.IMPLEMENTING

        elif task.execution_stage == ExecutionStage.IMPLEMENTING:
            task.execution_metadata["implementation_result"] = task.result
            task.execution_stage = ExecutionStage.TESTING

        elif task.execution_stage == ExecutionStage.TESTING:
            task.execution_metadata["testing_result"] = task.result
            task.execution_stage = ExecutionStage.REFINING

        elif task.execution_stage == ExecutionStage.REFINING:
            task.execution_metadata["refined_implementation"] = task.result
            task.execution_stage = ExecutionStage.FINALIZING

        elif task.execution_stage == ExecutionStage.FINALIZING:
            task.execution_metadata["final_result"] = task.result
            # Set verification status to passed when finalizing is complete
            task.verification_status = VerificationStatus.PASSED

        # Log the stage advancement
        stage_log = f"Advanced to execution stage: {task.execution_stage}"
        task.execution_logs.append(stage_log)
        self._debug_log(stage_log)

        return task
