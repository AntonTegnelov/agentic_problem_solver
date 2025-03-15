"""Agent coordination module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agent.agent_types.agent_types import Agent, AgentRegistry
from src.common_types import AgentInfo, AgentNotFoundError
from src.common_types.enums import AgentRole
from src.common_types.message_types import Message
from src.common_types.result_types import Result
from src.common_types.task_types import TaskComplexity, TaskStatus
from src.messages.creation import create_human_message
from src.utils.log_utils import DelegationInfo, get_logger, log_delegation_decision

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.common_types.result_types import Result as StepResult

# Constants
DESCRIPTION_TRUNCATION_LENGTH = 100
MIN_CAPABILITY_MATCH_THRESHOLD = 0.3


class InMemoryAgentRegistry(AgentRegistry):
    """Agent registry for managing agents."""

    def __init__(self) -> None:
        """Initialize agent registry."""
        self._agents: dict[str, Agent] = {}
        self._agent_info: dict[str, AgentInfo] = {}
        self._hierarchy_cache: dict[str, dict[str, list[str]]] = {}

    def register_agent(self, agent: Agent, info: AgentInfo | None = None) -> None:
        """Register agent.

        Args:
            agent: Agent instance.
            info: Optional agent info.

        """
        agent_id = agent.get_agent_id()
        self._agents[agent_id] = agent
        if info:
            self._agent_info[agent_id] = info
        else:
            # Create default agent info if not provided
            agent_type = agent_id.split("_")[0] if "_" in agent_id else "unknown"
            self._agent_info[agent_id] = AgentInfo(
                agent_id=agent_id,
                name=f"{agent_type.capitalize()} Agent",
                description=f"A {agent_type} agent with ID {agent_id}",
                capabilities=agent.get_capabilities(),
                status="idle",
                parent_id=agent.get_parent_id(),
                child_ids=agent.get_child_ids(),
            )

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent.

        Args:
            agent_id: Agent ID.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        # Remove parent-child relationships
        agent = self._agents[agent_id]
        parent_id = agent.get_parent_id()
        if parent_id and parent_id in self._agents:
            parent_agent = self._agents[parent_id]
            parent_agent.remove_child(agent_id)

        # Remove this agent as parent from all its children
        for child_id in agent.get_child_ids():
            if child_id in self._agents:
                child_agent = self._agents[child_id]
                child_agent.clear_parent()

        self._agents.pop(agent_id)
        if agent_id in self._agent_info:
            self._agent_info.pop(agent_id)

        # Invalidate hierarchy cache
        self._hierarchy_cache.clear()

    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent instance.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)
        return self._agents[agent_id]

    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """Get agent info.

        Args:
            agent_id: Agent ID.

        Returns:
            Agent info.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agent_info:
            msg = f"Agent info not found: {agent_id}"
            raise AgentNotFoundError(msg)
        return self._agent_info[agent_id]

    def list_agents(self) -> list[AgentInfo]:
        """List registered agents.

        Returns:
            List of agent information.

        """
        return list(self._agent_info.values())

    def get_agents(self) -> dict[str, Agent]:
        """Get all agents.

        Returns:
            Dictionary of agent IDs to agents.

        """
        return self._agents.copy()

    def find_agents_by_capability(self, capability: str) -> list[AgentInfo]:
        """Find agents by capability.

        Args:
            capability: Capability to search for.

        Returns:
            List of agent information.

        """
        result = []
        for agent_id, agent in self._agents.items():
            if capability in agent.get_capabilities():
                result.append(self._agent_info[agent_id])
        return result

    def find_agents_by_role(self, role: str) -> list[AgentInfo]:
        """Find agents by role.

        Args:
            role: Role to search for.

        Returns:
            List of agent information.

        """
        return [info for info in self._agent_info.values() if hasattr(info, "role") and info.role == role]

    def find_agents_by_parent(self, parent_id: str) -> list[AgentInfo]:
        """Find agents by parent ID.

        Args:
            parent_id: Parent ID to search for.

        Returns:
            List of agent information with the specified parent.

        """
        return [info for info in self._agent_info.values() if info.parent_id == parent_id]

    def get_parent_agent(self, agent_id: str) -> Agent | None:
        """Get parent agent of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Parent agent instance or None if no parent.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        agent = self._agents[agent_id]
        parent_id = agent.get_parent_id()

        if parent_id and parent_id in self._agents:
            return self._agents[parent_id]

        return None

    def get_child_agents(self, agent_id: str) -> list[Agent]:
        """Get child agents of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of child agent instances.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        agent = self._agents[agent_id]
        child_ids = agent.get_child_ids()

        return [self._agents[child_id] for child_id in child_ids if child_id in self._agents]

    def get_sibling_agents(self, agent_id: str) -> list[Agent]:
        """Get sibling agents of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of sibling agent instances.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        agent = self._agents[agent_id]
        parent_id = agent.get_parent_id()

        if not parent_id or parent_id not in self._agents:
            return []

        parent_agent = self._agents[parent_id]
        sibling_ids = [
            child_id for child_id in parent_agent.get_child_ids() if child_id != agent_id and child_id in self._agents
        ]

        return [self._agents[sibling_id] for sibling_id in sibling_ids]

    def register_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Register parent-child relationship between agents.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        Raises:
            AgentNotFoundError: If parent or child agent not found.

        """
        if parent_id not in self._agents:
            msg = f"Parent agent not found: {parent_id}"
            raise AgentNotFoundError(msg)

        if child_id not in self._agents:
            msg = f"Child agent not found: {child_id}"
            raise AgentNotFoundError(msg)

        parent_agent = self._agents[parent_id]
        child_agent = self._agents[child_id]

        # Check for circular relationships
        if self._would_create_cycle(parent_id, child_id):
            msg = f"Cannot create circular relationship between {parent_id} and {child_id}"
            raise ValueError(msg)

        # Remove existing parent-child relationship if any
        existing_parent_id = child_agent.get_parent_id()
        if existing_parent_id and existing_parent_id in self._agents:
            existing_parent = self._agents[existing_parent_id]
            existing_parent.remove_child(child_id)

        # Set new parent-child relationship
        parent_agent.add_child(child_id)
        child_agent.set_parent(parent_id)

        # Update agent info
        if parent_id in self._agent_info:
            self._agent_info[parent_id].child_ids = parent_agent.get_child_ids()

        if child_id in self._agent_info:
            self._agent_info[child_id].parent_id = parent_id

        # Invalidate hierarchy cache
        self._hierarchy_cache.clear()

    def remove_parent_child_relationship(self, parent_id: str, child_id: str) -> None:
        """Remove parent-child relationship between agents.

        Args:
            parent_id: Parent agent ID.
            child_id: Child agent ID.

        Raises:
            AgentNotFoundError: If parent or child agent not found.

        """
        if parent_id not in self._agents:
            msg = f"Parent agent not found: {parent_id}"
            raise AgentNotFoundError(msg)

        if child_id not in self._agents:
            msg = f"Child agent not found: {child_id}"
            raise AgentNotFoundError(msg)

        parent_agent = self._agents[parent_id]
        child_agent = self._agents[child_id]

        # Only remove if the relationship exists
        if child_agent.get_parent_id() == parent_id and child_id in parent_agent.get_child_ids():
            parent_agent.remove_child(child_id)
            child_agent.clear_parent()

            # Update agent info
            if parent_id in self._agent_info:
                self._agent_info[parent_id].child_ids = parent_agent.get_child_ids()

            if child_id in self._agent_info:
                self._agent_info[child_id].parent_id = None

            # Invalidate hierarchy cache
            self._hierarchy_cache.clear()

    def get_agent_hierarchy(self, root_agent_id: str) -> dict[str, list[str]]:
        """Get the agent hierarchy starting from the specified root agent.

        Args:
            root_agent_id: Root agent ID.

        Returns:
            Dictionary mapping agent IDs to lists of child agent IDs.

        Raises:
            AgentNotFoundError: If root agent not found.

        """
        if root_agent_id not in self._agents:
            msg = f"Root agent not found: {root_agent_id}"
            raise AgentNotFoundError(msg)

        # Check if hierarchy is cached
        if root_agent_id in self._hierarchy_cache:
            return self._hierarchy_cache[root_agent_id].copy()

        hierarchy = {}
        self._build_hierarchy(root_agent_id, hierarchy)

        # Cache the hierarchy
        self._hierarchy_cache[root_agent_id] = hierarchy.copy()

        return hierarchy

    def _build_hierarchy(self, agent_id: str, hierarchy: dict[str, list[str]]) -> None:
        """Recursively build the agent hierarchy.

        Args:
            agent_id: Current agent ID.
            hierarchy: Dictionary to populate with the hierarchy.

        """
        if agent_id not in self._agents:
            return

        agent = self._agents[agent_id]
        child_ids = agent.get_child_ids()
        valid_child_ids = [child_id for child_id in child_ids if child_id in self._agents]

        hierarchy[agent_id] = valid_child_ids

        for child_id in valid_child_ids:
            self._build_hierarchy(child_id, hierarchy)

    def _would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        """Check if adding a parent-child relationship would create a cycle.

        Args:
            parent_id: Potential parent agent ID.
            child_id: Potential child agent ID.

        Returns:
            True if a cycle would be created, False otherwise.

        """
        # If child is the same as parent, it's a cycle
        if parent_id == child_id:
            return True

        # Check if parent is a descendant of child
        current_id = parent_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            if current_id not in self._agents:
                break

            current_agent = self._agents[current_id]
            current_id = current_agent.get_parent_id()

            if current_id == child_id:
                return True

        return False

    def get_root_agents(self) -> list[Agent]:
        """Get all root agents (agents without parents).

        Returns:
            List of root agent instances.

        """
        return [agent for agent_id, agent in self._agents.items() if agent.get_parent_id() is None]

    def get_leaf_agents(self) -> list[Agent]:
        """Get all leaf agents (agents without children).

        Returns:
            List of leaf agent instances.

        """
        return [agent for agent_id, agent in self._agents.items() if not agent.get_child_ids()]

    def get_ancestors(self, agent_id: str) -> list[Agent]:
        """Get all ancestors of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of ancestor agent instances, ordered from parent to root.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        ancestors = []
        current_id = self._agents[agent_id].get_parent_id()

        while current_id and current_id in self._agents:
            ancestor = self._agents[current_id]
            ancestors.append(ancestor)
            current_id = ancestor.get_parent_id()

        return ancestors

    def get_descendants(self, agent_id: str) -> list[Agent]:
        """Get all descendants of the specified agent.

        Args:
            agent_id: Agent ID.

        Returns:
            List of descendant agent instances.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        if agent_id not in self._agents:
            msg = f"Agent not found: {agent_id}"
            raise AgentNotFoundError(msg)

        descendants = []
        hierarchy = self.get_agent_hierarchy(agent_id)

        # Skip the root agent itself
        if agent_id in hierarchy:
            self._collect_descendants(agent_id, hierarchy, descendants)

        return descendants

    def _collect_descendants(self, agent_id: str, hierarchy: dict[str, list[str]], descendants: list[Agent]) -> None:
        """Recursively collect descendants from hierarchy.

        Args:
            agent_id: Current agent ID.
            hierarchy: Agent hierarchy dictionary.
            descendants: List to populate with descendant agents.

        """
        if agent_id not in hierarchy:
            return

        for child_id in hierarchy[agent_id]:
            if child_id in self._agents:
                descendants.append(self._agents[child_id])
                self._collect_descendants(child_id, hierarchy, descendants)

    def validate_hierarchy(self) -> list[str]:
        """Validate the entire agent hierarchy for consistency.

        Returns:
            List of inconsistency messages, empty if hierarchy is valid.

        """
        inconsistencies = []

        # Check parent-child relationship consistency
        for agent_id, agent in self._agents.items():
            # Check child references
            for child_id in agent.get_child_ids():
                if child_id not in self._agents:
                    inconsistencies.append(f"Agent {agent_id} references non-existent child {child_id}")
                    continue

                child = self._agents[child_id]
                if child.get_parent_id() != agent_id:
                    inconsistencies.append(
                        f"Inconsistent parent-child relationship: {agent_id} lists {child_id} as child, "
                        f"but {child_id} has parent {child.get_parent_id()}",
                    )

            # Check parent references
            parent_id = agent.get_parent_id()
            if parent_id:
                if parent_id not in self._agents:
                    inconsistencies.append(f"Agent {agent_id} references non-existent parent {parent_id}")
                    continue

                parent = self._agents[parent_id]
                if agent_id not in parent.get_child_ids():
                    inconsistencies.append(
                        f"Inconsistent parent-child relationship: {agent_id} has parent {parent_id}, "
                        f"but {parent_id} doesn't list {agent_id} as child",
                    )

        # Check for cycles
        cycle_messages = [
            f"Cycle detected in hierarchy starting from {agent_id}"
            for agent_id in self._agents
            if self._has_cycle(agent_id)
        ]
        inconsistencies.extend(cycle_messages)

        return inconsistencies

    def _has_cycle(self, start_agent_id: str) -> bool:
        """Check if there's a cycle in the hierarchy starting from the given agent.

        Args:
            start_agent_id: Starting agent ID.

        Returns:
            True if a cycle is detected, False otherwise.

        """
        visited = set()
        current_id = start_agent_id

        while current_id and current_id in self._agents:
            if current_id in visited:
                return True

            visited.add(current_id)
            current_id = self._agents[current_id].get_parent_id()

        return False

    def repair_hierarchy(self) -> int:
        """Repair inconsistencies in the agent hierarchy.

        Returns:
            Number of inconsistencies repaired.

        """
        repairs_count = 0

        repairs_count += self._repair_missing_child_references()
        repairs_count += self._repair_missing_parent_references()
        repairs_count += self._repair_invalid_references()
        repairs_count += self._repair_agent_info()

        # Invalidate hierarchy cache after repairs
        if repairs_count > 0:
            self._hierarchy_cache.clear()

        return repairs_count

    def _repair_missing_child_references(self) -> int:
        """Fix missing child references in the hierarchy.

        Returns:
            Number of repairs made.

        """
        repairs_count = 0

        # Fix missing child references
        for agent_id, agent in self._agents.items():
            parent_id = agent.get_parent_id()
            if parent_id and parent_id in self._agents:
                parent = self._agents[parent_id]
                if agent_id not in parent.get_child_ids():
                    parent.add_child(agent_id)
                    repairs_count += 1

        return repairs_count

    def _repair_missing_parent_references(self) -> int:
        """Fix missing parent references in the hierarchy.

        Returns:
            Number of repairs made.

        """
        repairs_count = 0

        # Fix missing parent references
        for agent_id, agent in self._agents.items():
            for child_id in agent.get_child_ids():
                if child_id in self._agents:
                    child = self._agents[child_id]
                    if child.get_parent_id() != agent_id:
                        child.set_parent(agent_id)
                        repairs_count += 1

        return repairs_count

    def _repair_invalid_references(self) -> int:
        """Remove references to non-existent agents.

        Returns:
            Number of repairs made.

        """
        repairs_count = 0

        # Remove references to non-existent agents
        for agent in self._agents.values():
            # Clean up child references
            invalid_children = [child_id for child_id in agent.get_child_ids() if child_id not in self._agents]

            for invalid_child in invalid_children:
                agent.remove_child(invalid_child)
                repairs_count += 1

            # Clean up parent reference
            parent_id = agent.get_parent_id()
            if parent_id and parent_id not in self._agents:
                agent.clear_parent()
                repairs_count += 1

        return repairs_count

    def _repair_agent_info(self) -> int:
        """Update agent info to match agent state.

        Returns:
            Number of repairs made.

        """
        repairs_count = 0

        # Update agent info to match agent state
        for agent_id, agent in self._agents.items():
            if agent_id in self._agent_info:
                info = self._agent_info[agent_id]
                if info.parent_id != agent.get_parent_id():
                    info.parent_id = agent.get_parent_id()
                    repairs_count += 1

                if set(info.child_ids) != set(agent.get_child_ids()):
                    info.child_ids = agent.get_child_ids()
                    repairs_count += 1

        return repairs_count


class AgentCoordinator:
    """Agent coordinator for delegating tasks to agents."""

    def __init__(self, registry: InMemoryAgentRegistry) -> None:
        """Initialize agent coordinator.

        Args:
            registry: Agent registry.

        """
        self._registry = registry
        self._factories: dict[str, Callable[..., Agent]] = {}
        self._logger = get_logger("agent.coordinator")
        # Resource management configuration
        self._resource_limits = {
            "max_agents": 50,  # Maximum number of agents allowed
            "max_agents_per_role": {  # Maximum number of agents per role
                AgentRole.ARCHITECT.value: 5,
                AgentRole.PLANNER.value: 15,
                AgentRole.EXECUTOR.value: 30,
            },
            "max_children_per_agent": 10,  # Maximum number of child agents per parent
            "max_hierarchy_depth": 5,  # Maximum depth of agent hierarchy
        }
        # Capability categories for organization
        self._capability_categories = {
            "design": ["architecture", "system design", "design", "modeling"],
            "planning": ["planning", "organization", "scheduling", "coordination"],
            "development": ["coding", "implementation", "development", "programming"],
            "testing": ["testing", "quality assurance", "verification", "validation"],
            "analysis": ["analysis", "research", "investigation", "evaluation"],
            "documentation": ["documentation", "writing", "reporting"],
            "maintenance": ["maintenance", "support", "operations"],
            "specialized": [],  # Will be populated with capabilities that don't fit other categories
        }
        # Constants
        self._TASK_SUMMARY_MAX_LENGTH = 50  # Maximum length for task summaries in logs

    def register_agent_factory(self, agent_type: str, factory: Callable[..., Agent]) -> None:
        """Register agent factory.

        Args:
            agent_type: Agent type.
            factory: Agent factory function.

        """
        self._factories[agent_type] = factory

    async def delegate_task(self, agent_id: str, task: str) -> Result:
        """Delegate task to agent.

        Args:
            agent_id: Agent ID.
            task: Task to delegate.

        Returns:
            Task result.

        Raises:
            AgentNotFoundError: If agent not found.

        """
        agent = self._registry.get_agent(agent_id)
        message = Message(content=task, role="human")
        return await agent.process(message)

    async def broadcast_task(self, task: str) -> dict[str, Result]:
        """Broadcast task to all agents.

        Args:
            task: Task to broadcast.

        Returns:
            Dictionary of agent IDs to results.

        """
        results = {}
        message = Message(content=task, role="human")

        for agent_id, agent in self._registry.get_agents().items():
            results[agent_id] = await agent.process(message)

        return results

    def find_agent_for_task(self, task: str) -> str | None:
        """Find agent for task.

        Args:
            task: Task to find agent for.

        Returns:
            Agent ID or None if no suitable agent found.

        """
        for agent_id in self._registry.get_agents():
            info = self._registry.get_agent_info(agent_id)
            if info and hasattr(info, "can_handle_task") and info.can_handle_task(task):
                return agent_id
        return None

    def _check_resource_limits(self, agent_type: str, parent_id: str | None = None) -> tuple[bool, str]:
        """Check if creating a new agent would exceed resource limits.

        Args:
            agent_type: Type of agent to create.
            parent_id: Optional parent agent ID.

        Returns:
            Tuple of (is_allowed, reason). If is_allowed is False, reason contains
            the explanation for why the agent creation is not allowed.

        """
        # Check total agent count
        total_agents = len(self._registry.get_agents())
        if total_agents >= self._resource_limits["max_agents"]:
            return False, f"Maximum number of agents ({self._resource_limits['max_agents']}) reached"

        # Check role-specific limits if agent_type is a role
        try:
            role = AgentRole(agent_type.lower())
            role_value = role.value
            role_agents = len(self._registry.find_agents_by_role(role_value))
            if role_value in self._resource_limits["max_agents_per_role"]:
                max_for_role = self._resource_limits["max_agents_per_role"][role_value]
                if role_agents >= max_for_role:
                    return False, f"Maximum number of {role_value} agents ({max_for_role}) reached"
        except ValueError:
            # Not a role, continue with other checks
            pass

        # Check parent's child count limit
        if parent_id:
            try:
                parent_agent = self._registry.get_agent(parent_id)
                child_count = len(parent_agent.get_child_ids())
                if child_count >= self._resource_limits["max_children_per_agent"]:
                    return (
                        False,
                        f"Maximum number of children "
                        f"({self._resource_limits['max_children_per_agent']}) "
                        f"for parent {parent_id} reached",
                    )

                # Check hierarchy depth limit
                if self._resource_limits["max_hierarchy_depth"] > 0:
                    # Calculate current depth of parent
                    ancestors = self._registry.get_ancestors(parent_id)
                    current_depth = len(ancestors) + 1  # +1 for the parent itself

                    # New agent would be at current_depth + 1
                    if current_depth + 1 > self._resource_limits["max_hierarchy_depth"]:
                        return (
                            False,
                            f"Maximum hierarchy depth "
                            f"({self._resource_limits['max_hierarchy_depth']}) "
                            f"would be exceeded",
                        )
            except AgentNotFoundError:
                # Parent not found, can't check child count
                pass

        return True, ""

    def create_agent(self, agent_type: str, config: dict, **kwargs: dict[str, Any]) -> Agent:
        """Create agent.

        Args:
            agent_type: Agent type.
            config: Agent configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Created agent.

        Raises:
            ValueError: If agent type not registered or resource limits would be exceeded.

        """
        # Check if agent_type is a role name
        try:
            role = AgentRole(agent_type.lower())
            return self.create_agent_by_role(role, config, **kwargs)
        except ValueError:
            # Not a role, continue with regular agent creation
            pass

        if agent_type not in self._factories:
            msg = f"Invalid agent type: {agent_type}"
            raise ValueError(msg)

        # Extract parent_id from kwargs or config
        parent_id = kwargs.get("parent_id")
        if parent_id is None and "parent_id" in config:
            parent_id = config["parent_id"]

        # Check resource limits
        is_allowed, reason = self._check_resource_limits(agent_type, parent_id)
        if not is_allowed:
            msg = f"Cannot create agent: {reason}"
            self._logger.warning(
                "Agent creation denied due to resource limits",
                extra={
                    "agent_type": agent_type,
                    "parent_id": parent_id,
                    "reason": reason,
                },
            )
            raise ValueError(msg)

        agent = self._factories[agent_type](config=config, **kwargs)
        config.get("agent_id", agent.get_agent_id())
        self._registry.register_agent(agent, None)
        return agent

    def create_agent_by_role(self, role: AgentRole, config: dict, **kwargs: dict[str, Any]) -> Agent:
        """Create an agent by role.

        This method creates an agent based on its role in the hierarchical system.
        It uses the specialized agent creation functions from agent_types module.

        Args:
            role: Agent role.
            config: Agent configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Created agent.

        Raises:
            ValueError: If the role is not supported or resource limits would be exceeded.

        """
        from src.agent.agent_types import create_agent as create_agent_by_role
        from src.config.agent import AgentConfig

        # Convert dict config to AgentConfig
        agent_config = AgentConfig()
        for key, value in config.items():
            if hasattr(agent_config, key):
                setattr(agent_config, key, value)

        # Extract parent_id from kwargs or config
        parent_id = kwargs.get("parent_id")
        if parent_id is None and "parent_id" in config:
            parent_id = config["parent_id"]

        # Extract custom agent_id if provided
        agent_id = kwargs.get("agent_id")
        if agent_id is None and "agent_id" in config:
            agent_id = config["agent_id"]
            # Make sure it's set in the agent_config
            agent_config.agent_id = agent_id

        # Check resource limits
        is_allowed, reason = self._check_resource_limits(role.value, parent_id)
        if not is_allowed:
            msg = f"Cannot create agent: {reason}"
            self._logger.warning(
                "Agent creation denied due to resource limits",
                extra={
                    "role": role.value,
                    "parent_id": parent_id,
                    "reason": reason,
                },
            )
            raise ValueError(msg)

        # Create the agent using the specialized factory function
        agent = create_agent_by_role(
            role=role,
            config=agent_config,
            parent_id=parent_id,
            **kwargs,
        )

        # Register the agent
        self._registry.register_agent(agent, None)

        # If parent_id is provided, establish the parent-child relationship
        if parent_id:
            self._registry.register_parent_child_relationship(parent_id, agent.get_agent_id())

        # Log the creation
        self._logger.info(
            "Created agent by role",
            extra={
                "agent_id": agent.get_agent_id(),
                "role": role.value,
                "parent_id": parent_id,
            },
        )

        return agent

    async def route_message(self, message: Message, target_agent_id: str) -> StepResult:
        """Route message to target agent.

        Args:
            message: Message to route.
            target_agent_id: Target agent ID.

        Returns:
            Result of message processing.

        Raises:
            AgentNotFoundError: If target agent not found.

        """
        agent = self._registry.get_agent(target_agent_id)
        return await agent.process(message)

    async def delegate_task_flexible(
        self,
        source_agent_id: str,
        task: str,
        target_role: str | None = None,
        complexity: str | None = None,
    ) -> Result:
        """Delegate a task using flexible delegation paths.

        This method supports delegation based on role, complexity, or capabilities.
        It determines the most appropriate agent to handle the task based on the
        provided criteria and the source agent's position in the hierarchy.

        Args:
            source_agent_id: ID of the agent delegating the task.
            task: Task to delegate.
            target_role: Optional target role for delegation.
            complexity: Optional task complexity for delegation decisions.

        Returns:
            Result of delegation.

        """
        try:
            source_agent = self._registry.get_agent(source_agent_id)
            source_role = None

            # Try to get the role from the agent info
            try:
                source_info = self._registry.get_agent_info(source_agent_id)
                if hasattr(source_info, "role"):
                    source_role = source_info.role
            except (AttributeError, KeyError, TypeError) as e:
                # If we can't get the role, log the error and continue without it
                self._logger.debug("Could not get role from agent info: %s", str(e))

            # Determine target agent based on provided criteria
            target_agent_id = None
            delegation_reason = ""

            # If target role is specified, find an agent with that role
            if target_role:
                target_agent_id = self._find_agent_by_role(source_agent, target_role)
                delegation_reason = f"Role-based delegation to {target_role}"

            # If complexity is specified, find an agent based on complexity
            elif complexity:
                target_agent_id = self._find_agent_by_complexity(
                    source_agent_id,
                    source_role,
                    complexity,
                )
                delegation_reason = f"Complexity-based delegation ({complexity})"

            # Otherwise, find an agent based on task capabilities
            else:
                target_agent_id = self._find_agent_by_capability(source_agent, task)
                delegation_reason = "Capability-based delegation"

            if not target_agent_id:
                log_delegation_decision(
                    logger=self._logger,
                    delegation_info=DelegationInfo(
                        source_agent_id=source_agent_id,
                        target_agent_id="none",
                        task=task,
                        reason="No suitable agent found for delegation",
                        additional_info={
                            "target_role": target_role,
                            "complexity": complexity,
                        },
                    ),
                )
                return Result.failure("No suitable agent found for delegation")

            # Log the delegation decision
            log_delegation_decision(
                logger=self._logger,
                delegation_info=DelegationInfo(
                    source_agent_id=source_agent_id,
                    target_agent_id=target_agent_id,
                    task=task,
                    reason=delegation_reason,
                    additional_info={
                        "target_role": target_role,
                        "complexity": complexity,
                    },
                ),
            )

            # Create a message for the target agent
            message = create_human_message(content=task)

            # Get the target agent and delegate the task
            target_agent = self._registry.get_agent(target_agent_id)
            return await target_agent.process(message)

        except AgentNotFoundError as e:
            log_delegation_decision(
                logger=self._logger,
                delegation_info=DelegationInfo(
                    source_agent_id=source_agent_id,
                    target_agent_id="error",
                    task=task,
                    reason=f"Agent not found: {e!s}",
                    additional_info={
                        "error_type": "AgentNotFoundError",
                    },
                ),
            )
            return Result.failure(f"Agent not found: {e!s}")

        except (ValueError, TypeError, RuntimeError) as e:
            log_delegation_decision(
                logger=self._logger,
                delegation_info=DelegationInfo(
                    source_agent_id=source_agent_id,
                    target_agent_id="error",
                    task=task,
                    reason=f"Delegation error: {e!s}",
                    additional_info={
                        "error_type": type(e).__name__,
                    },
                ),
            )
            return Result.failure(f"Delegation error: {e!s}")

    async def delegate_hierarchical_tasks(
        self,
        source_agent_id: str,
        task: str,
        parent_task_id: str | None = None,
    ) -> Result:
        """Delegate a task hierarchically by breaking it down and assigning subtasks.

        This method:
        1. Uses the source agent to break down the task into subtasks
        2. Delegates each subtask to an appropriate agent based on complexity and role
        3. Tracks the delegated tasks and their relationships

        Args:
            source_agent_id: ID of the agent delegating the task.
            task: High-level task to break down and delegate.
            parent_task_id: Optional ID of a parent task.

        Returns:
            Result containing information about the delegated tasks.

        """
        try:
            # Get the source agent and prepare for task breakdown
            preparation_result = self._prepare_for_hierarchical_delegation(source_agent_id)
            if not preparation_result.success:
                return preparation_result

            source_agent, agent_role = preparation_result.data

            # Break down the task into subtasks
            breakdown_result = await self._break_down_task(source_agent, agent_role, task, parent_task_id)
            if not breakdown_result.success:
                return breakdown_result

            subtasks = breakdown_result.data

            # Delegate the subtasks
            return await self._delegate_subtasks(source_agent_id, subtasks)

        except AgentNotFoundError as e:
            error_msg = f"Agent not found: {e!s}"
            self._logger.exception(error_msg)
            return Result.failure(error_msg)
        except (ValueError, TypeError, RuntimeError) as e:
            error_msg = f"Delegation error: {e!s}"
            self._logger.exception(error_msg)
            return Result.failure(error_msg)

    def _prepare_for_hierarchical_delegation(self, source_agent_id: str) -> Result:
        """Prepare for hierarchical delegation by getting the source agent and determining its role.

        Args:
            source_agent_id: ID of the agent delegating the task.

        Returns:
            Result containing the source agent and its role.

        """
        # Get the source agent
        source_agent = self._registry.get_agent(source_agent_id)
        source_role = None

        # Try to get the role from the agent info
        try:
            source_info = self._registry.get_agent_info(source_agent_id)
            if hasattr(source_info, "role"):
                source_role = source_info.role
        except (AttributeError, KeyError, TypeError) as e:
            # If we can't get the role, log the error and continue without it
            self._logger.debug("Could not get role from agent info: %s", str(e))

        # Get the agent's state
        state = source_agent.get_state()
        if not state:
            return Result.failure("Agent has no state")

        # Convert string role to AgentRole enum if needed
        from src.common_types.enums import AgentRole

        agent_role = None
        if source_role:
            try:
                agent_role = AgentRole(source_role.upper())
            except (ValueError, AttributeError):
                self._logger.warning(
                    "Invalid role %s for agent %s, using default",
                    source_role,
                    source_agent_id,
                )

        # If we couldn't determine the role, default to ARCHITECT
        if not agent_role:
            agent_role = AgentRole.ARCHITECT

        return Result(success=True, data=(source_agent, agent_role))

    async def _break_down_task(
        self,
        source_agent: Agent,
        agent_role: AgentRole,
        task: str,
        parent_task_id: str | None = None,
    ) -> Result:
        """Break down a task into subtasks.

        Args:
            source_agent: The agent breaking down the task.
            agent_role: The role of the agent.
            task: The task to break down.
            parent_task_id: Optional ID of a parent task.

        Returns:
            Result containing the subtasks.

        """
        # Create a TaskBreakdownStep for the source agent's role
        from src.agent.steps import TaskBreakdownStep

        # Create the task breakdown step
        breakdown_step = TaskBreakdownStep(agent_role=agent_role)
        breakdown_step.set_agent(source_agent)

        # Break down the task
        self._logger.info(
            "Breaking down task for agent %s with role %s",
            source_agent.get_agent_id(),
            agent_role,
        )

        breakdown_result = await breakdown_step(
            state=source_agent.get_state(),
            task_description=task,
            parent_task_id=parent_task_id,
        )

        if not breakdown_result.success:
            error_msg = f"Task breakdown failed: {breakdown_result.error}"
            self._logger.error(error_msg)
            return Result.failure(error_msg)

        # Get the subtasks from the result
        subtasks = breakdown_result.data
        if not subtasks:
            return Result.failure("No subtasks were created during breakdown")

        self._logger.info("Created %d subtasks", len(subtasks))
        return Result(success=True, data=subtasks)

    async def _delegate_subtasks(self, source_agent_id: str, subtasks: list) -> Result:
        """Delegate subtasks to appropriate agents.

        Args:
            source_agent_id: ID of the agent delegating the tasks.
            subtasks: List of subtasks to delegate.

        Returns:
            Result containing information about the delegated tasks.

        """
        # Delegate each subtask to an appropriate agent
        delegation_results = []
        for subtask in subtasks:
            # Determine the appropriate agent based on task complexity
            complexity = subtask.complexity.value.upper()

            self._logger.info(
                "Delegating subtask with complexity %s: %s",
                complexity,
                subtask.description[:DESCRIPTION_TRUNCATION_LENGTH]
                + ("..." if len(subtask.description) > DESCRIPTION_TRUNCATION_LENGTH else ""),
            )

            # Delegate the task using flexible delegation
            delegation_result = await self.delegate_task_flexible(
                source_agent_id=source_agent_id,
                task=subtask.description,
                complexity=complexity,
            )

            # Update the subtask with the delegation result
            if delegation_result.success:
                subtask.status = TaskStatus.IN_PROGRESS
                # Check if agent_id is in the result data dictionary
                if isinstance(delegation_result.data, dict) and "agent_id" in delegation_result.data:
                    subtask.assigned_agent_id = delegation_result.data["agent_id"]
            else:
                subtask.status = TaskStatus.FAILED
                subtask.error = delegation_result.error

            delegation_results.append(
                {
                    "task_id": str(subtask.task_id),
                    "success": delegation_result.success,
                    "assigned_agent_id": subtask.assigned_agent_id,
                    "error": delegation_result.error if not delegation_result.success else None,
                },
            )

        # Return the results
        return Result(
            success=True,
            data={
                "subtasks": [str(subtask.task_id) for subtask in subtasks],
                "delegation_results": delegation_results,
            },
        )

    def _find_agent_by_role(self, source_agent: Agent, target_role: str) -> str:
        """Find an agent by role.

        Args:
            source_agent: Source agent.
            target_role: Target agent role.

        Returns:
            Target agent ID.

        Raises:
            ValueError: If no suitable agent found.

        """
        target_agents = self._registry.find_agents_by_role(target_role)
        if not target_agents:
            msg = f"No agents found with role: {target_role}"
            raise ValueError(msg)

        # Prefer child agents of the source agent if available
        child_ids = source_agent.get_child_ids()
        for agent_info in target_agents:
            if agent_info.agent_id in child_ids:
                return agent_info.agent_id

        # If no child agent with the target role, use the first one found
        return target_agents[0].agent_id

    def _find_agent_by_complexity(self, source_agent_id: str, source_role: str | None, complexity: str) -> str:
        """Find an agent based on task complexity.

        Args:
            source_agent_id: ID of the source agent.
            source_role: Role of the source agent, if available.
            complexity: Task complexity.

        Returns:
            ID of the target agent.

        """
        # Determine the appropriate delegation based on source agent role and complexity
        if source_role == "architect":
            target_agent_id = self._architect_delegation_by_complexity(complexity)

            # Log the delegation decision
            log_delegation_decision(
                logger=self._logger,
                delegation_info=DelegationInfo(
                    source_agent_id=source_agent_id,
                    target_agent_id=target_agent_id,
                    task=f"Complexity-based task ({complexity})",
                    reason=f"Architect delegating based on {complexity} complexity",
                    additional_info={"complexity": complexity},
                ),
            )

            return target_agent_id

        if source_role == "planner":
            target_agent_id = self._planner_delegation_by_complexity(source_agent_id, complexity)

            # Log the delegation decision
            log_delegation_decision(
                logger=self._logger,
                delegation_info=DelegationInfo(
                    source_agent_id=source_agent_id,
                    target_agent_id=target_agent_id,
                    task=f"Complexity-based task ({complexity})",
                    reason=f"Planner delegating based on {complexity} complexity",
                    additional_info={"complexity": complexity},
                ),
            )

            return target_agent_id

        # Default case - find an executor
        executor_agents = self._registry.find_agents_by_role("executor")
        if executor_agents:
            target_agent_id = executor_agents[0].agent_id

            # Log the delegation decision
            log_delegation_decision(
                logger=self._logger,
                delegation_info=DelegationInfo(
                    source_agent_id=source_agent_id,
                    target_agent_id=target_agent_id,
                    task=f"Complexity-based task ({complexity})",
                    reason="Default delegation to executor",
                    additional_info={"complexity": complexity},
                ),
            )

            return target_agent_id

        # If no suitable agent found, return the source agent ID
        return source_agent_id

    def _architect_delegation_by_complexity(self, complexity: str) -> str:
        """Determine delegation path for Architect agent based on complexity.

        Args:
            complexity: Task complexity.

        Returns:
            Target agent ID.

        Raises:
            ValueError: If no suitable agent found.

        """
        if complexity == "SIMPLE":
            # Direct delegation from Architect to Executor for simple tasks
            executor_agents = self._registry.find_agents_by_role("EXECUTOR")
            if not executor_agents:
                msg = "No executor agents found for direct delegation"
                raise ValueError(msg)
            return executor_agents[0].agent_id
        # Standard delegation from Architect to Planner for complex tasks
        planner_agents = self._registry.find_agents_by_role("PLANNER")
        if not planner_agents:
            msg = "No planner agents found for delegation"
            raise ValueError(msg)
        return planner_agents[0].agent_id

    def _planner_delegation_by_complexity(self, source_agent_id: str, complexity: str) -> str:
        """Determine delegation path for Planner agent based on complexity.

        Args:
            source_agent_id: Source agent ID.
            complexity: Task complexity.

        Returns:
            Target agent ID.

        Raises:
            ValueError: If no suitable agent found.

        """
        if complexity == "COMPLEX":
            # Recursive delegation from Planner to another Planner for complex sub-components
            return self._find_or_create_planner(source_agent_id)
        # Standard delegation from Planner to Executor for implementable tasks
        executor_agents = self._registry.find_agents_by_role("EXECUTOR")
        if not executor_agents:
            msg = "No executor agents found for delegation"
            raise ValueError(msg)
        return executor_agents[0].agent_id

    def _find_or_create_planner(self, source_agent_id: str) -> str:
        """Find an existing planner agent or create a new one.

        Args:
            source_agent_id: Source agent ID.

        Returns:
            Planner agent ID.

        Raises:
            ValueError: If no planner agent found and no factory available.

        """
        planner_agents = self._registry.find_agents_by_role("PLANNER")
        if not planner_agents:
            msg = "No planner agents found for recursive delegation"
            raise ValueError(msg)

        # Find a different planner agent (not the source)
        for agent_info in planner_agents:
            if agent_info.agent_id != source_agent_id:
                return agent_info.agent_id

        # If no other planner available, create a new one
        if "PLANNER" in self._factories:
            new_planner = self._factories["PLANNER"](
                config={"parent_id": source_agent_id},
            )
            self._registry.register_agent(new_planner)
            return new_planner.get_agent_id()

        msg = "No planner factory available for creating new planner"
        raise ValueError(msg)

    def _find_agent_by_capability(self, source_agent: Agent, task: str) -> str:
        """Find an agent by capability or use existing child agents.

        Args:
            source_agent: Source agent.
            task: Task to delegate.

        Returns:
            Target agent ID.

        Raises:
            ValueError: If no suitable agent found.

        """
        # Check if source agent has child agents
        child_ids = source_agent.get_child_ids()
        if child_ids:
            # Use the first child agent
            return child_ids[0]

        # Extract task capabilities from the task description
        task_capabilities = self._extract_task_capabilities(task)

        # Find agents that can handle the task based on capabilities
        source_agent_id = source_agent.get_agent_id()
        candidate_agents = {}

        for agent_id, agent in self._registry.get_agents().items():
            if agent_id != source_agent_id:
                agent_capabilities = agent.get_capabilities()
                # Calculate capability match score
                match_score = self._calculate_capability_match_score(task_capabilities, agent_capabilities)
                if match_score > 0:
                    candidate_agents[agent_id] = match_score

        # Sort candidates by match score (highest first)
        if candidate_agents:
            sorted_candidates = sorted(candidate_agents.items(), key=lambda x: x[1], reverse=True)
            return sorted_candidates[0][0]

        msg = f"No suitable agent found for task: {task}"
        raise ValueError(msg)

    def _extract_task_capabilities(self, task: str) -> list[str]:
        """Extract capabilities required for a task from its description.

        This method analyzes the task description to identify key capabilities
        that would be required to complete it.

        Args:
            task: Task description.

        Returns:
            List of extracted capabilities.

        """
        # List of common capability keywords to look for
        capability_keywords = [
            "design",
            "architecture",
            "planning",
            "implementation",
            "coding",
            "testing",
            "debugging",
            "analysis",
            "research",
            "documentation",
            "review",
            "optimization",
            "refactoring",
            "integration",
            "deployment",
            "database",
            "frontend",
            "backend",
            "api",
            "ui",
            "ux",
            "security",
            "performance",
            "scalability",
            "monitoring",
            "maintenance",
        ]

        # Extract capabilities based on keyword presence
        task_lower = task.lower()

        # Use list comprehension instead of for loop
        extracted_capabilities = [keyword for keyword in capability_keywords if keyword.lower() in task_lower]

        # Add role-based capabilities based on task complexity indicators
        complexity_indicators = {
            "architect": ["system design", "high-level", "architecture", "overall structure"],
            "planner": ["break down", "plan", "organize", "coordinate", "schedule"],
            "executor": ["implement", "code", "write", "develop", "create", "build"],
        }

        for role, indicators in complexity_indicators.items():
            if any(indicator.lower() in task_lower for indicator in indicators):
                extracted_capabilities.append(role)

        return extracted_capabilities

    def _calculate_capability_match_score(self, task_capabilities: list[str], agent_capabilities: list[str]) -> float:
        """Calculate a match score between task capabilities and agent capabilities.

        Args:
            task_capabilities: Capabilities required for the task.
            agent_capabilities: Capabilities of the agent.

        Returns:
            Match score between 0.0 and 1.0, where higher is better.

        """
        if not task_capabilities or not agent_capabilities:
            return 0.0

        # Convert all capabilities to lowercase for case-insensitive matching
        task_caps_lower = [cap.lower() for cap in task_capabilities]
        agent_caps_lower = [cap.lower() for cap in agent_capabilities]

        # Count exact matches
        exact_matches = sum(1 for cap in task_caps_lower if cap in agent_caps_lower)

        # Count partial matches (substring matching)
        partial_matches = 0
        for task_cap in task_caps_lower:
            for agent_cap in agent_caps_lower:
                # Skip if already counted as exact match
                if task_cap == agent_cap:
                    continue
                # Check if one is substring of the other
                if task_cap in agent_cap or agent_cap in task_cap:
                    partial_matches += 0.5
                    break

        # Calculate final score
        total_task_capabilities = len(task_capabilities)
        match_score = (
            (exact_matches + partial_matches) / total_task_capabilities if total_task_capabilities > 0 else 0.0
        )

        return min(1.0, match_score)  # Cap at 1.0

    def discover_capabilities(self) -> dict[str, list[str]]:
        """Discover and categorize agent capabilities in the system.

        This method collects all capabilities from registered agents and
        organizes them into categories for easier discovery and usage.

        Returns:
            Dictionary mapping capability categories to lists of specific capabilities.

        """
        # Collect all unique capabilities from registered agents
        all_capabilities = set()
        for agent in self._registry.get_agents().values():
            all_capabilities.update(agent.get_capabilities())

        # Organize capabilities by category
        categorized_capabilities: dict[str, list[str]] = {category: [] for category in self._capability_categories}

        # Assign capabilities to categories
        for capability in all_capabilities:
            capability_lower = capability.lower()
            assigned = False

            # Check if capability belongs to a predefined category
            for category, keywords in self._capability_categories.items():
                if any(keyword in capability_lower for keyword in keywords):
                    if capability not in categorized_capabilities[category]:
                        categorized_capabilities[category].append(capability)
                    assigned = True
                    break

            # If not assigned to any category, put in specialized
            if not assigned and capability not in categorized_capabilities["specialized"]:
                categorized_capabilities["specialized"].append(capability)

        # Sort capabilities within each category
        for capabilities in categorized_capabilities.values():
            capabilities.sort()

        return categorized_capabilities

    def get_agents_with_capability(self, capability: str) -> list[str]:
        """Get IDs of agents that have a specific capability.

        Args:
            capability: The capability to search for.

        Returns:
            List of agent IDs that have the specified capability.

        """
        agent_ids = []
        for agent_id, agent in self._registry.get_agents().items():
            agent_capabilities = agent.get_capabilities()
            # Check for exact match or substring match
            if capability in agent_capabilities or any(
                capability.lower() in cap.lower() or cap.lower() in capability.lower() for cap in agent_capabilities
            ):
                agent_ids.append(agent_id)

        return agent_ids

    def get_capabilities_by_category(self, category: str) -> list[str]:
        """Get all capabilities in a specific category.

        Args:
            category: The category to get capabilities for.

        Returns:
            List of capabilities in the specified category.

        Raises:
            ValueError: If the category doesn't exist.

        """
        # First refresh the categorized capabilities
        categorized_capabilities = self.discover_capabilities()

        if category not in categorized_capabilities:
            valid_categories = list(categorized_capabilities.keys())
            msg = f"Invalid category: {category}. Valid categories are: {valid_categories}"
            raise ValueError(msg)

        return categorized_capabilities[category]

    def find_most_capable_agent_for_task(self, task: str, required_capabilities: list[str] | None = None) -> str | None:
        """Find the most capable agent for a given task.

        Args:
            task: Task description.
            required_capabilities: Optional list of required capabilities.

        Returns:
            Agent ID of the most capable agent, or None if no suitable agent found.

        """
        # Extract task capabilities if not provided
        task_capabilities = required_capabilities or self._extract_task_capabilities(task)
        if not task_capabilities:
            self._logger.warning("No capabilities extracted from task: %s", task)
            return None

        # Find agents that can handle the task based on capabilities
        candidate_agents = {}

        for agent_id, agent in self._registry.get_agents().items():
            agent_capabilities = agent.get_capabilities()
            # Calculate capability match score
            match_score = self._calculate_capability_match_score(task_capabilities, agent_capabilities)
            if match_score > 0:
                candidate_agents[agent_id] = match_score

        # Sort candidates by match score (highest first)
        if candidate_agents:
            sorted_candidates = sorted(candidate_agents.items(), key=lambda x: x[1], reverse=True)
            return sorted_candidates[0][0]

        return None

    async def route_task_by_capability(
        self,
        task: str,
        source_agent_id: str | None = None,
        required_capabilities: list[str] | None = None,
        task_complexity: TaskComplexity | None = None,
    ) -> Result:
        """Route a task to the most appropriate agent based on capabilities.

        This method analyzes the task, extracts required capabilities, and finds
        the most suitable agent to handle it. It considers both capability matching
        and task complexity to make optimal routing decisions.

        Args:
            task: Task description.
            source_agent_id: Optional ID of the agent initiating the routing.
            required_capabilities: Optional list of required capabilities.
            task_complexity: Optional task complexity for better agent matching.

        Returns:
            Result containing the target agent ID and routing information.

        """
        self._logger.debug("Routing task by capability: %s", task[:DESCRIPTION_TRUNCATION_LENGTH])

        # Extract task capabilities if not provided
        task_capabilities = required_capabilities or self._extract_task_capabilities(task)
        if not task_capabilities:
            return Result.failure("Could not extract capabilities from task")

        # Log the extracted capabilities
        self._logger.debug("Extracted capabilities: %s", task_capabilities)

        # Consider task complexity in routing decision
        complexity_str = task_complexity.value if task_complexity else "moderate"

        # Find the most suitable agent based on capabilities and complexity
        target_agent_id = None

        # If we have a source agent, avoid routing back to it
        excluded_agent_ids = [source_agent_id] if source_agent_id else []

        # Find candidate agents with matching capabilities
        candidate_agents = {}
        for agent_id, agent in self._registry.get_agents().items():
            if agent_id in excluded_agent_ids:
                continue

            agent_capabilities = agent.get_capabilities()
            match_score = self._calculate_capability_match_score(task_capabilities, agent_capabilities)

            # Only consider agents with a reasonable match score
            if match_score >= MIN_CAPABILITY_MATCH_THRESHOLD:
                # Get agent info to check role for complexity matching
                try:
                    agent_info = self._registry.get_agent_info(agent_id)
                    agent_role = getattr(agent_info, "role", None)

                    # Adjust score based on complexity-role alignment
                    if (
                        agent_role
                        and complexity_str
                        and (
                            (agent_role == "EXECUTOR" and complexity_str in ["simple", "moderate"])
                            or (agent_role == "PLANNER" and complexity_str in ["moderate", "complex"])
                            or (agent_role == "ARCHITECT" and complexity_str in ["complex", "very_complex"])
                        )
                    ):
                        match_score *= 1.2

                    candidate_agents[agent_id] = match_score
                except (AgentNotFoundError, AttributeError):
                    # If we can't get agent info, just use the base match score
                    candidate_agents[agent_id] = match_score

        # Sort candidates by adjusted match score (highest first)
        if candidate_agents:
            sorted_candidates = sorted(candidate_agents.items(), key=lambda x: x[1], reverse=True)
            target_agent_id = sorted_candidates[0][0]
            match_score = sorted_candidates[0][1]

            self._logger.info(
                "Selected agent %s for task with match score %.2f",
                target_agent_id,
                match_score,
            )

            # Create routing result with detailed information
            routing_info = {
                "target_agent_id": target_agent_id,
                "match_score": match_score,
                "capabilities": task_capabilities,
                "complexity": complexity_str,
                "candidates": dict(sorted_candidates[:3]),  # Include top 3 candidates
            }

            return Result(success=True, data=routing_info)

        return Result.failure("No suitable agent found for the task based on capabilities")


class AgentFactory:
    """Agent factory for creating agents."""

    def __init__(self, registry: InMemoryAgentRegistry) -> None:
        """Initialize agent factory.

        Args:
            registry: Agent registry.

        """
        self._registry = registry
        self._factories: dict[str, Callable[..., Agent]] = {}

    def register_factory(self, agent_type: str, factory: Callable[..., Agent]) -> None:
        """Register agent factory.

        Args:
            agent_type: Agent type.
            factory: Agent factory function.

        """
        self._factories[agent_type] = factory

    def create_agent(self, agent_type: str, **kwargs: dict[str, Any]) -> Agent:
        """Create agent.

        Args:
            agent_type: Agent type.
            **kwargs: Additional keyword arguments.

        Returns:
            Created agent.

        Raises:
            ValueError: If agent type not registered.

        """
        if agent_type not in self._factories:
            msg = f"Agent type not registered: {agent_type}"
            raise ValueError(msg)

        agent = self._factories[agent_type](**kwargs)
        self._registry.register_agent(agent, None)
        return agent


# Export symbols
__all__ = [
    "Agent",
    "AgentCoordinator",
    "AgentFactory",
    "AgentInfo",
    "AgentRegistry",
    "InMemoryAgentRegistry",
    "SimpleAgentCoordinator",
]

# For backward compatibility
SimpleAgentCoordinator = AgentCoordinator
