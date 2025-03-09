from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

class AgentState(BaseModel):
    """State of the agent during execution."""
    messages: List[HumanMessage | AIMessage] = Field(default_factory=list)
    current_task: Optional[str] = None
    task_stack: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str):
        """Initialize the agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
        self.state = AgentState()
    
    async def process_message(self, message: str) -> str:
        """Process a message and return a response.
        
        Args:
            message: Input message to process
            
        Returns:
            Agent's response
        """
        # Add message to state
        self.state.messages.append(HumanMessage(content=message))
        
        # Process message (to be implemented by subclasses)
        response = await self._process_message_impl(message)
        
        # Add response to state
        self.state.messages.append(AIMessage(content=response))
        
        return response
    
    async def _process_message_impl(self, message: str) -> str:
        """Implementation of message processing.
        
        To be implemented by subclasses.
        
        Args:
            message: Input message to process
            
        Returns:
            Processed response
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
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def clear_state(self) -> None:
        """Reset agent state to initial values."""
        self.state = AgentState() 