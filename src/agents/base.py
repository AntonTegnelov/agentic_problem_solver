"""Base agent module."""

from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.messages import (
    create_ai_message,
    create_human_message,
    create_system_message,
    create_tool_message,
    get_message_metadata,
    set_message_metadata,
)


class AgentState:
    """State class for agents."""

    def __init__(self):
        """Initialize agent state."""
        self.messages: List[
            Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]
        ] = []
        self.current_step = 0
        self.error = None
        self.result = None

    def get_message_metadata(self, message: Any, key: str, default: Any = None) -> Any:
        """Get metadata from a message.

        Args:
            message: Message object
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return get_message_metadata(message, key, default)

    def set_message_metadata(self, message: Any, key: str, value: Any) -> None:
        """Set metadata for a message.

        Args:
            message: Message object
            key: Metadata key
            value: Metadata value
        """
        set_message_metadata(message, key, value)

    def add_system_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a system message to the state.

        Args:
            content: Message content
            metadata: Optional metadata dictionary
        """
        self.messages.append(create_system_message(content, metadata))

    def add_human_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a human message to the state.

        Args:
            content: Message content
            metadata: Optional metadata dictionary
        """
        self.messages.append(create_human_message(content, metadata))

    def add_ai_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an AI message to the state.

        Args:
            content: Message content
            metadata: Optional metadata dictionary
        """
        self.messages.append(create_ai_message(content, metadata))

    def add_tool_message(
        self, content: str, tool_call_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a tool message to the state.

        Args:
            content: Message content
            tool_call_id: ID of the tool call
            metadata: Optional metadata dictionary
        """
        self.messages.append(create_tool_message(content, tool_call_id, metadata))

    def clear(self) -> None:
        """Clear the state."""
        self.messages = []
        self.current_step = 0
        self.error = None
        self.result = None


class BaseAgent:
    """Base class for all agents in the system."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent.

        Args:
            name: Name of the agent
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.state = AgentState()

    async def process_message(self, message: str) -> str:
        """Process a message and return a response.

        Args:
            message: Input message to process

        Returns:
            Agent's response
        """
        try:
            # Add user message to state
            human_message = HumanMessage(
                content=message,
                additional_kwargs={
                    "metadata": {},
                    "tool_calls": None,
                    "function_call": None,
                },
            )
            self.state.messages.append(human_message)

            # Process the message
            response = await self._process_message_impl(human_message)

            # Add agent response to state if it's not already added
            if response not in self.state.messages:
                self.state.messages.append(response)

            return response.content

        except Exception as e:
            # Add error message to state
            error_message = SystemMessage(
                content=f"Error: {str(e)}",
                additional_kwargs={
                    "metadata": {"error": str(e), "type": type(e).__name__},
                    "tool_calls": None,
                    "function_call": None,
                },
            )
            self.state.messages.append(error_message)
            self.state.error = str(e)
            raise

    async def _process_message_impl(self, message: HumanMessage) -> AIMessage:
        """Process a message and return a response.

        This method should be implemented by subclasses.

        Args:
            message: Input message to process

        Returns:
            Agent's response
        """
        raise NotImplementedError

    def get_state(self) -> AgentState:
        """Get current agent state.

        Returns:
            Current state of the agent
        """
        return self.state

    def update_state(self, **kwargs) -> None:
        """Update agent state.

        Args:
            **kwargs: State fields to update
        """
        for key, value in kwargs.items():
            setattr(self.state, key, value)

    def clear_state(self):
        """Clear the agent's state."""
        self.state.clear()

    def update_config(self, **kwargs) -> None:
        """Update agent configuration.

        Args:
            **kwargs: Configuration fields to update
        """
        self.config.update(kwargs)

    def get_message_metadata(self, index: int, key: str, default: Any = None) -> Any:
        """Get metadata from a message at the specified index.

        Args:
            index: Index of the message
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self.state.get_message_metadata(self.state.messages[index], key, default)

    def set_message_metadata(self, index: int, key: str, value: Any) -> None:
        """Set metadata for a message at the specified index.

        Args:
            index: Index of the message
            key: Metadata key
            value: Metadata value
        """
        self.state.set_message_metadata(self.state.messages[index], key, value)
