"""Agent-specific error classes.

.. deprecated:: 0.1.0
   These error classes have been moved to src.common_types.error_types.
   This module will be removed in a future release.

Migration steps:
1. Update imports to use src.common_types.error_types
2. Known usages:
   - tests/unit/test_utils.py
   - tests/unit/test_errors.py
   - tests/integration/test_enhanced_messages.py
   - src/messages/router.py
   - src/agent/result.py
"""

import warnings

from src.common_types.error_types import (
    AgentCommunicationError as CommonAgentCommunicationError,
)
from src.common_types.error_types import (
    AgentConfigError as CommonAgentConfigError,
)
from src.common_types.error_types import (
    AgentCreationError as CommonAgentCreationError,
)
from src.common_types.error_types import (
    AgentError as CommonAgentError,
)
from src.common_types.error_types import (
    AgentExecutionError as CommonAgentExecutionError,
)
from src.common_types.error_types import (
    AgentNotFoundError as CommonAgentNotFoundError,
)


class AgentError(CommonAgentError):
    """Base class for agent-related errors.

    .. deprecated:: 0.1.0
       This class has been moved to src.common_types.error_types.AgentError.
       This version will be removed in a future release.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the error."""
        super().__init__(*args, **kwargs)
        warnings.warn(
            "AgentError has been moved to src.common_types.error_types.AgentError. "
            "This version will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )


class AgentNotFoundError(CommonAgentNotFoundError):
    """Raised when an agent is not found.

    .. deprecated:: 0.1.0
       This class has been moved to src.common_types.error_types.AgentNotFoundError.
       This version will be removed in a future release.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the error."""
        super().__init__(*args, **kwargs)
        warnings.warn(
            "AgentNotFoundError has been moved to src.common_types.error_types.AgentNotFoundError. "
            "This version will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )


class AgentCreationError(CommonAgentCreationError):
    """Raised when agent creation fails.

    .. deprecated:: 0.1.0
       This class has been moved to src.common_types.error_types.AgentCreationError.
       This version will be removed in a future release.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the error."""
        super().__init__(*args, **kwargs)
        warnings.warn(
            "AgentCreationError has been moved to src.common_types.error_types.AgentCreationError. "
            "This version will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )


class AgentConfigError(CommonAgentConfigError):
    """Raised when agent configuration is invalid.

    .. deprecated:: 0.1.0
       This class has been moved to src.common_types.error_types.AgentConfigError.
       This version will be removed in a future release.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the error."""
        super().__init__(*args, **kwargs)
        warnings.warn(
            "AgentConfigError has been moved to src.common_types.error_types.AgentConfigError. "
            "This version will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )


class AgentCommunicationError(CommonAgentCommunicationError):
    """Raised when communication with an agent fails.

    .. deprecated:: 0.1.0
       This class has been moved to src.common_types.error_types.AgentCommunicationError.
       This version will be removed in a future release.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the error."""
        super().__init__(*args, **kwargs)
        warnings.warn(
            "AgentCommunicationError has been moved to src.common_types.error_types.AgentCommunicationError. "
            "This version will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )


class AgentExecutionError(CommonAgentExecutionError):
    """Raised when agent execution fails.

    .. deprecated:: 0.1.0
       This class has been moved to src.common_types.error_types.AgentExecutionError.
       This version will be removed in a future release.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the error."""
        super().__init__(*args, **kwargs)
        warnings.warn(
            "AgentExecutionError has been moved to src.common_types.error_types.AgentExecutionError. "
            "This version will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
