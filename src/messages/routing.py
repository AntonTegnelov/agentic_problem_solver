"""Hierarchical message routing for agent communication.

This module provides specialized routing capabilities for hierarchical agent structures,
enabling messages to be routed up and down the agent hierarchy. It extends the base
routing functionality to support parent-child relationships between agents.

Key features:
- Hierarchical message routing (up/down the agent tree)
- Ancestor/descendant message delivery
- Sibling message routing
- Path-based message delivery
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from src.agent.agent_types.agent_types import Agent
from src.common_types import AgentNotFoundError
from src.common_types.error_types import RoutingError
from src.messages.utils import (
    add_to_hierarchy_path,
    is_hierarchical_message,
    set_hierarchy_path,
    set_receiver_id,
    set_receiver_parent_id,
    set_sender_id,
    set_sender_parent_id,
)

if TYPE_CHECKING:
    from src.agent.agent_types.agent_types import AgentRegistry
    from src.common_types.message_types import Message
    from src.common_types.result_types import Result as StepResult
    from src.messages.chain import MessageChain
    from src.messages.router import MessageRouter

logger = logging.getLogger(__name__)

# Constants
MIN_PATH_LENGTH = 2  # Minimum number of agents in a path (source and destination)


class HierarchicalRouter:
    """Router for hierarchical message delivery between agents.

    This class extends the base message routing capabilities to support
    hierarchical agent structures, allowing messages to be routed up and
    down the agent tree.

    Attributes:
        router: Base message router for direct agent-to-agent communication.
        registry: Agent registry for looking up agents by ID and relationships.

    """

    def __init__(
        self,
        router: MessageRouter,
        registry: AgentRegistry,
    ) -> None:
        """Initialize hierarchical router.

        Args:
            router: Base message router for direct agent-to-agent communication.
            registry: Agent registry for looking up agents by ID and relationships.

        """
        self.router = router
        self.registry = registry

    async def route_to_parent(
        self,
        message: Message,
        agent_id: str,
        chain: MessageChain | None = None,
    ) -> StepResult:
        """Route message from an agent to its parent.

        Args:
            message: Message to route.
            agent_id: ID of the agent sending the message.
            chain: Optional message chain to record the message.

        Returns:
            Step result from the parent agent's processing.

        Raises:
            RoutingError: If the agent has no parent or parent not found.

        """
        # Get the parent ID for the agent
        parent_id = self.registry.get_parent_id(agent_id)
        if not parent_id:
            msg = f"Agent {agent_id} has no parent"
            raise RoutingError(msg)

        # Set message metadata for hierarchical routing
        set_sender_id(message, agent_id)
        set_receiver_id(message, parent_id)

        # Update hierarchy path if it exists, or create a new one
        if is_hierarchical_message(message):
            add_to_hierarchy_path(message, agent_id)
        else:
            set_hierarchy_path(message, [agent_id])

        try:
            return await self.router.route_message(message, parent_id, chain)
        except AgentNotFoundError as e:
            msg = f"Parent agent {parent_id} not found for {agent_id}"
            raise RoutingError(msg) from e

    async def route_to_child(
        self,
        message: Message,
        parent_id: str,
        child_id: str,
        chain: MessageChain | None = None,
    ) -> StepResult:
        """Route message from a parent agent to a specific child.

        Args:
            message: Message to route.
            parent_id: ID of the parent agent sending the message.
            child_id: ID of the child agent to receive the message.
            chain: Optional message chain to record the message.

        Returns:
            Step result from the child agent's processing.

        Raises:
            RoutingError: If the child is not a child of the parent or child not found.

        """
        # Verify parent-child relationship
        if not self.registry.is_child_of(child_id, parent_id):
            msg = f"Agent {child_id} is not a child of {parent_id}"
            raise RoutingError(msg)

        # Set message metadata for hierarchical routing
        set_sender_id(message, parent_id)
        set_receiver_id(message, child_id)
        set_sender_parent_id(message, self.registry.get_parent_id(parent_id))
        set_receiver_parent_id(message, parent_id)

        # Update hierarchy path if it exists, or create a new one
        if is_hierarchical_message(message):
            add_to_hierarchy_path(message, parent_id)
        else:
            set_hierarchy_path(message, [parent_id])

        try:
            return await self.router.route_message(message, child_id, chain)
        except AgentNotFoundError as e:
            msg = f"Child agent {child_id} not found for {parent_id}"
            raise RoutingError(msg) from e

    async def route_to_children(
        self,
        message: Message,
        parent_id: str,
    ) -> list[StepResult]:
        """Broadcast message from a parent to all its children.

        Args:
            message: Message to broadcast.
            parent_id: ID of the parent agent sending the message.

        Returns:
            List of results from child agents.

        """
        children = self.registry.get_children(parent_id)
        if not children:
            return []

        results = []
        for child_id in children:
            try:
                result = await self.route_to_child(message, parent_id, child_id)
                results.append(result)
            except RoutingError as e:
                logger.warning("Error routing to child %s: %s", child_id, e)

        return results

    async def route_to_sibling(
        self,
        message: Message,
        sender_id: str,
        sibling_id: str,
        chain: MessageChain | None = None,
    ) -> StepResult:
        """Route message between sibling agents (agents with the same parent).

        Args:
            message: Message to route.
            sender_id: ID of the sending agent.
            sibling_id: ID of the sibling agent to receive the message.
            chain: Optional message chain to record the message.

        Returns:
            Step result from the sibling agent's processing.

        Raises:
            RoutingError: If agents are not siblings or sibling not found.

        """
        # Get parent IDs for both agents
        sender_parent_id = self.registry.get_parent_id(sender_id)
        sibling_parent_id = self.registry.get_parent_id(sibling_id)

        # Check if sender has no parent
        if not sender_parent_id:
            msg = f"Agent {sender_id} has no parent"
            raise RoutingError(msg)

        # Check if sibling has no parent
        if not sibling_parent_id:
            msg = f"Agent {sibling_id} has no parent"
            raise RoutingError(msg)

        # Verify they have the same parent
        if sender_parent_id != sibling_parent_id:
            msg = f"Agents {sender_id} and {sibling_id} are not siblings"
            raise RoutingError(msg)

        # Set message metadata for hierarchical routing
        set_sender_id(message, sender_id)
        set_receiver_id(message, sibling_id)
        set_sender_parent_id(message, sender_parent_id)
        set_receiver_parent_id(message, sibling_parent_id)

        # Update hierarchy path if it exists, or create a new one
        if is_hierarchical_message(message):
            add_to_hierarchy_path(message, sender_id)
        else:
            set_hierarchy_path(message, [sender_id])

        try:
            return await self.router.route_message(message, sibling_id, chain)
        except AgentNotFoundError as e:
            msg = f"Sibling agent {sibling_id} not found for {sender_id}"
            raise RoutingError(msg) from e

    async def route_by_path(
        self,
        message: Message,
        path: list[str],
        chain: MessageChain | None = None,
    ) -> StepResult:
        """Route message along a specific path in the agent hierarchy.

        Args:
            message: Message to route.
            path: List of agent IDs defining the routing path.
            chain: Optional message chain to record the message.

        Returns:
            Step result from the final agent's processing.

        Raises:
            RoutingError: If the path is invalid or an agent in the path is not found.

        """
        if len(path) < MIN_PATH_LENGTH:
            msg = "Path must contain at least one agent ID"
            raise RoutingError(msg)

        # Set the hierarchy path in the message
        set_hierarchy_path(message, path)

        # Set sender and receiver IDs
        set_sender_id(message, path[0])
        set_receiver_id(message, path[-1])

        # Verify the path is valid (each agent is a child of the previous one)
        for i in range(1, len(path)):
            parent_id = path[i - 1]
            child_id = path[i]
            if not self.registry.is_child_of(child_id, parent_id):
                msg = f"Invalid path: {child_id} is not a child of {parent_id}"
                raise RoutingError(msg)

        # Route to the final destination
        try:
            return await self.router.route_message(message, path[-1], chain)
        except AgentNotFoundError as e:
            msg = f"Agent {path[-1]} not found in path {path}"
            raise RoutingError(msg) from e

    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent instance.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        agent = self.registry.get_agent(agent_id)
        if not agent:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)
        return cast(Agent, agent)
