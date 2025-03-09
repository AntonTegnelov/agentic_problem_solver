"""Main problem-solving agent implementation."""

from typing import Any, Dict, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.llm_providers import LLMProviderFactory
from src.prompts import get_step_prompt
from src.utils.logging import get_logger
from src.validation import AgentStep, TaskInput

logger = get_logger(__name__)


class AgentGraphState(TypedDict):
    """State for the agent graph."""

    messages: list[Union[HumanMessage, AIMessage, SystemMessage]]
    next_step: str
    context: Dict[str, Any]


class SolverAgent(BaseAgent):
    """Main problem-solving agent using langgraph."""

    def __init__(self, name: str = "solver", config: Optional[Dict[str, Any]] = None):
        """Initialize the solver agent.

        Args:
            name: Name of the agent (default: solver)
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.llm_factory = LLMProviderFactory()
        self.llm = self.llm_factory.get_provider()
        logger.info(f"Initialized {name} agent with {self.llm.__class__.__name__}")
        self.graph_state = AgentGraphState()
        self.current_step = AgentStep.UNDERSTAND

    async def _process_message_impl(self, message: HumanMessage) -> AIMessage:
        """Process a message using the agent's workflow.

        Args:
            message: Input message to process

        Returns:
            Agent's response
        """
        # Initialize task input
        task_input = TaskInput(content=message.content)
        self.state.messages.append(
            SystemMessage(
                content="Starting task processing",
                additional_kwargs={"metadata": {"task_input": task_input.model_dump()}},
            )
        )

        # Process task through steps
        while await self._should_continue(message):
            if self.current_step == AgentStep.UNDERSTAND:
                await self._understand_task(message)
            elif self.current_step == AgentStep.PLAN:
                await self._create_plan(message)
            elif self.current_step == AgentStep.EXECUTE:
                await self._execute_plan(message)
            elif self.current_step == AgentStep.VERIFY:
                await self._verify_result(message)
            elif self.current_step == AgentStep.END:
                await self._end_processing(message)
                break

        # Return final result or error
        result = (
            self.state.result or "Task completed successfully"
            if self.current_step == AgentStep.END and not self.state.error
            else self.state.error or "Task processing failed"
        )
        return AIMessage(content=result)

    async def _understand_task(self, message: HumanMessage) -> AIMessage:
        """Understand the task and extract key information."""
        prompt = get_step_prompt(AgentStep.UNDERSTAND, {"task": message.content})
        response = await self.llm.generate(prompt)
        self.graph_state["messages"].append(response)
        self.graph_state["next_step"] = AgentStep.PLAN.value
        self.state.current_step = AgentStep.PLAN
        return response

    async def _create_plan(self, message: HumanMessage) -> AIMessage:
        """Create a plan based on task understanding."""
        current_node = self.graph_state.get("messages")[-1]
        if not current_node:
            return AIMessage(content="Error: No task analysis available")

        prompt = get_step_prompt(
            AgentStep.PLAN, {"task_analysis": current_node["content"]}
        )
        response = await self.llm.generate(prompt)
        self.graph_state["messages"].append(response)
        self.graph_state["next_step"] = AgentStep.EXECUTE.value
        self.state.current_step = AgentStep.EXECUTE
        return response

    async def _execute_plan(self, message: HumanMessage) -> AIMessage:
        """Execute the created plan."""
        current_node = self.graph_state.get("messages")[-1]
        if not current_node:
            return AIMessage(content="Error: No plan available")

        prompt = get_step_prompt(AgentStep.EXECUTE, {"plan": current_node["content"]})
        response = await self.llm.generate(prompt)
        self.graph_state["messages"].append(response)
        self.graph_state["next_step"] = AgentStep.VERIFY.value
        self.state.current_step = AgentStep.VERIFY
        return response

    async def _verify_result(self, message: HumanMessage) -> AIMessage:
        """Verify the execution results."""
        current_node = self.graph_state.get("messages")[-1]
        if not current_node:
            return AIMessage(content="Error: No execution result available")

        prompt = get_step_prompt(AgentStep.VERIFY, {"result": current_node["content"]})
        response = await self.llm.generate(prompt)
        self.graph_state["messages"].append(response)
        self.graph_state["next_step"] = AgentStep.END.value
        self.state.current_step = AgentStep.END
        return response

    async def _end_processing(self, message: HumanMessage) -> AIMessage:
        """End the processing and prepare final response."""
        self.state.result = "Task completed successfully"
        current_node = self.graph_state.get("messages")[-1]
        if not current_node:
            return AIMessage(content="Error: No verification result available")

        prompt = get_step_prompt(AgentStep.END, {"task": message.content})
        response = await self.llm.generate(prompt)
        self.graph_state["messages"].append(response)
        self.state.current_step = AgentStep.END
        return response

    async def _should_continue(self, message: HumanMessage) -> bool:
        """Determine if processing should continue."""
        return not self.state.error and self.state.current_step != AgentStep.END
