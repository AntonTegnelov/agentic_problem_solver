"""Main problem-solving agent implementation."""

from enum import Enum
from typing import Any

from langchain.schema import AIMessage, HumanMessage

from src.agents.base import AgentState, BaseAgent
from src.llm_providers.factory import LLMProviderFactory
from src.prompts import get_step_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Error messages
INVALID_STEP_ERROR = "Invalid step: {}"
INVALID_STATE_ERROR = "Invalid step state: {}"
EMPTY_MESSAGE_ERROR = "Message content cannot be empty"

# Constants
TEST_SYSTEM_MESSAGE = "Test system message"


class ProcessingStep(str, Enum):
    """Agent processing steps."""

    UNDERSTAND = "understand"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    END = "end"


class SolverAgent(BaseAgent):
    """Main problem-solving agent using langgraph."""

    def __init__(
        self,
        name: str = "solver",
        config: dict[str, Any] | None = None,
        llm_factory: LLMProviderFactory | None = None,
    ) -> None:
        """Initialize the solver agent.

        Args:
            name: Name of the agent (default: solver)
            config: Optional configuration dictionary
            llm_factory: Optional LLM provider factory (default: None)
        """
        super().__init__(name, config)
        self.llm_factory = llm_factory or LLMProviderFactory()
        self.llm = self.llm_factory.get_provider()
        logger.info("Initialized %s agent with %s", name, self.llm.__class__.__name__)
        self.state = AgentState()
        self.current_step = ProcessingStep.UNDERSTAND
        self.execution_result = ""
        self.state.add_system_message(
            TEST_SYSTEM_MESSAGE,
            metadata={"initialization": True},
        )

    async def process_message(self, message: HumanMessage) -> AIMessage:
        """Process a message.

        Args:
            message: The message to process.

        Returns:
            Agent response.

        Raises:
            ValueError: If message is empty or step is invalid.
        """
        if not message.content:
            raise ValueError(EMPTY_MESSAGE_ERROR)

        try:
            return await self._process_step(message)
        except Exception as e:
            if self.state.error:
                logger.exception("Processing failed: %s", self.state.error)
            else:
                logger.exception("Unexpected error during processing")
                self.state.error = str(e)
            raise

    async def _process_step(self, message: HumanMessage) -> AIMessage:
        """Process the current step.

        Args:
            message: The message to process.

        Returns:
            Agent response.

        Raises:
            ValueError: If step is invalid.
        """
        if self.current_step == ProcessingStep.UNDERSTAND:
            return await self._understand_task(message)
        if self.current_step == ProcessingStep.PLAN:
            return await self._plan_implementation(message)
        if self.current_step == ProcessingStep.IMPLEMENT:
            return await self._implement_solution(message)
        if self.current_step == ProcessingStep.VERIFY:
            return await self._verify_result()
        if self.current_step == ProcessingStep.END:
            return await self._end_processing()

        error_msg = INVALID_STEP_ERROR.format(self.current_step)
        logger.error(error_msg)
        self.state.error = error_msg
        raise ValueError(error_msg)

    async def _understand_task(self, message: HumanMessage) -> AIMessage:
        """Understand the task and extract key information."""
        logger.info("Understanding task...")
        prompt = get_step_prompt(ProcessingStep.UNDERSTAND, {"task": message.content})
        logger.debug("Understanding prompt: %s", prompt)
        response = await self.llm.generate(prompt)
        if isinstance(response, str):
            response = AIMessage(content=response)
        logger.debug("Understanding response: %s", response.content)

        self.current_step = ProcessingStep.PLAN
        return response

    async def _plan_implementation(self, message: HumanMessage) -> AIMessage:
        """Plan the implementation approach."""
        logger.info("Planning implementation...")
        prompt = get_step_prompt(ProcessingStep.PLAN, {"task": message.content})
        logger.debug("Planning prompt: %s", prompt)
        response = await self.llm.generate(prompt)
        if isinstance(response, str):
            response = AIMessage(content=response)
        logger.debug("Planning response: %s", response.content)

        self.current_step = ProcessingStep.IMPLEMENT
        return response

    async def _implement_solution(self, message: HumanMessage) -> AIMessage:
        """Implement the solution."""
        logger.info("Implementing solution...")
        prompt = get_step_prompt(ProcessingStep.IMPLEMENT, {"task": message.content})
        logger.debug("Implementation prompt: %s", prompt)
        response = await self.llm.generate(prompt)
        if isinstance(response, str):
            response = AIMessage(content=response)
        logger.debug("Implementation response: %s", response.content)

        self.execution_result = response.content
        self.current_step = ProcessingStep.VERIFY
        return response

    async def _verify_result(self) -> AIMessage:
        """Verify the execution results."""
        logger.info("Verifying results...")
        prompt = get_step_prompt(
            ProcessingStep.VERIFY, {"result": self.execution_result}
        )
        logger.debug("Verification prompt: %s", prompt)
        response = await self.llm.generate(prompt)
        if isinstance(response, str):
            response = AIMessage(content=response)
        logger.debug("Verification response: %s", response.content)

        self.current_step = ProcessingStep.END
        return response

    async def _end_processing(self) -> AIMessage:
        """End the processing of the current task."""
        logger.info("Ending processing...")
        code_start = self.execution_result.find("[CODE]")
        code_end = self.execution_result.find("[/CODE]")
        logger.debug("Code block markers: start=%d, end=%d", code_start, code_end)

        if code_start != -1 and code_end != -1:
            solution = self.execution_result[code_start + 6 : code_end].strip()
            logger.info("Extracted solution: %.100s...", solution)
            return AIMessage(
                content=f"Here's your solution:\n\n```python\n{solution}\n```"
            )

        # If no code tags found, return the full result
        logger.warning("No code block found, returning full result")
        return AIMessage(content=self.execution_result)

    def _should_continue(self) -> bool:
        """Check if processing should continue.

        Returns:
            True if processing should continue, False otherwise.
        """
        try:
            ProcessingStep(self.current_step)
        except ValueError:
            error_msg = INVALID_STATE_ERROR.format(self.current_step)
            logger.exception(error_msg)
            self.state.error = error_msg
            return False
        return True

    def clear_state(self) -> None:
        """Clear the agent's state."""
        super().clear_state()
        self.state = AgentState()
        self.current_step = ProcessingStep.UNDERSTAND
        self.execution_result = ""
        self.state.add_system_message(
            TEST_SYSTEM_MESSAGE,
            metadata={"initialization": True},
        )
