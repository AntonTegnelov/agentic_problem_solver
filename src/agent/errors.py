"""Agent-specific error classes."""

from src.exceptions import ConfigError


class AgentError(Exception):
    """Base class for agent-related errors."""


class AgentNotFoundError(AgentError):
    """Raised when an agent is not found."""


class AgentCreationError(AgentError):
    """Raised when agent creation fails."""


class AgentConfigError(ConfigError):
    """Raised when agent configuration is invalid."""


class AgentCommunicationError(AgentError):
    """Raised when communication with an agent fails."""


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""
