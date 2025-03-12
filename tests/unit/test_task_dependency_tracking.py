"""Tests for task dependency tracking functionality."""

from src.agent.state.base import AgentState
from src.common_types.task_types import Task, TaskDependency, TaskStatus


class TestTaskDependencyTracking:
    """Test task dependency tracking functionality."""

    def test_add_task(self) -> None:
        """Test adding a task to the state."""
        state = AgentState()
        task = Task(description="Test task")

        state.add_task(task)

        tasks = state.get_tasks()
        assert len(tasks) == 1
        assert tasks[0]["description"] == "Test task"
        assert tasks[0]["task_id"] == str(task.task_id)

    def test_get_task_by_id(self) -> None:
        """Test getting a task by ID."""
        state = AgentState()
        task = Task(description="Test task")

        state.add_task(task)

        retrieved_task = state.get_task_by_id(task.task_id)
        assert retrieved_task is not None
        assert retrieved_task.description == "Test task"
        assert retrieved_task.task_id == task.task_id

    def test_update_task(self) -> None:
        """Test updating a task."""
        state = AgentState()
        task = Task(description="Test task")

        state.add_task(task)

        # Update task
        task.description = "Updated task"
        state.update_task(task)

        retrieved_task = state.get_task_by_id(task.task_id)
        assert retrieved_task is not None
        assert retrieved_task.description == "Updated task"

    def test_is_task_blocked_by_dependencies(self) -> None:
        """Test checking if a task is blocked by dependencies."""
        state = AgentState()

        # Create dependency task
        dependency_task = Task(description="Dependency task")
        state.add_task(dependency_task)

        # Create dependent task
        dependent_task = Task(
            description="Dependent task",
            dependencies=[
                TaskDependency(
                    task_id=dependency_task.task_id,
                    description="Depends on dependency task",
                    is_blocking=True,
                ),
            ],
        )
        state.add_task(dependent_task)

        # Task should be blocked initially
        assert state.is_task_blocked_by_dependencies(dependent_task.task_id) is True

        # Complete dependency task
        dependency_task.status = TaskStatus.COMPLETED
        state.update_task(dependency_task)

        # Task should no longer be blocked
        assert state.is_task_blocked_by_dependencies(dependent_task.task_id) is False

    def test_update_task_status_based_on_dependencies(self) -> None:
        """Test updating task status based on dependencies."""
        state = AgentState()

        # Create dependency task
        dependency_task = Task(description="Dependency task")
        state.add_task(dependency_task)

        # Create dependent task
        dependent_task = Task(
            description="Dependent task",
            dependencies=[
                TaskDependency(
                    task_id=dependency_task.task_id,
                    description="Depends on dependency task",
                    is_blocking=True,
                ),
            ],
        )
        state.add_task(dependent_task)

        # Update status based on dependencies
        state.update_task_status_based_on_dependencies(dependent_task.task_id)

        # Task should be blocked
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.BLOCKED

        # Complete dependency task
        dependency_task.status = TaskStatus.COMPLETED
        state.update_task(dependency_task)

        # Update status based on dependencies
        state.update_task_status_based_on_dependencies(dependent_task.task_id)

        # Task should now be pending
        updated_task = state.get_task_by_id(dependent_task.task_id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.PENDING

    def test_update_dependent_tasks(self) -> None:
        """Test updating dependent tasks when a dependency is completed."""
        state = AgentState()

        # Create dependency task
        dependency_task = Task(description="Dependency task")
        state.add_task(dependency_task)

        # Create dependent tasks
        dependent_task1 = Task(
            description="Dependent task 1",
            dependencies=[
                TaskDependency(
                    task_id=dependency_task.task_id,
                    description="Depends on dependency task",
                    is_blocking=True,
                ),
            ],
        )
        dependent_task2 = Task(
            description="Dependent task 2",
            dependencies=[
                TaskDependency(
                    task_id=dependency_task.task_id,
                    description="Depends on dependency task",
                    is_blocking=True,
                ),
            ],
        )
        state.add_task(dependent_task1)
        state.add_task(dependent_task2)

        # Update status based on dependencies
        state.update_task_status_based_on_dependencies(dependent_task1.task_id)
        state.update_task_status_based_on_dependencies(dependent_task2.task_id)

        # Tasks should be blocked
        task1 = state.get_task_by_id(dependent_task1.task_id)
        task2 = state.get_task_by_id(dependent_task2.task_id)
        assert task1 is not None
        assert task1.status == TaskStatus.BLOCKED
        assert task2 is not None
        assert task2.status == TaskStatus.BLOCKED

        # Complete dependency task
        dependency_task.status = TaskStatus.COMPLETED
        state.update_task(dependency_task)

        # Update dependent tasks
        state.update_dependent_tasks(dependency_task.task_id)

        # Tasks should now be pending
        task1 = state.get_task_by_id(dependent_task1.task_id)
        task2 = state.get_task_by_id(dependent_task2.task_id)
        assert task1 is not None
        assert task1.status == TaskStatus.PENDING
        assert task2 is not None
        assert task2.status == TaskStatus.PENDING

    def test_complex_dependency_chain(self) -> None:
        """Test a complex chain of dependencies."""
        state = AgentState()

        # Create tasks in a dependency chain: A <- B <- C
        task_a = Task(description="Task A")
        task_b = Task(
            description="Task B",
            dependencies=[
                TaskDependency(
                    task_id=task_a.task_id,
                    description="Depends on Task A",
                    is_blocking=True,
                ),
            ],
        )
        task_c = Task(
            description="Task C",
            dependencies=[
                TaskDependency(
                    task_id=task_b.task_id,
                    description="Depends on Task B",
                    is_blocking=True,
                ),
            ],
        )

        state.add_task(task_a)
        state.add_task(task_b)
        state.add_task(task_c)

        # Update status based on dependencies
        state.update_task_status_based_on_dependencies(task_b.task_id)
        state.update_task_status_based_on_dependencies(task_c.task_id)

        # Tasks B and C should be blocked
        assert state.get_task_by_id(task_b.task_id).status == TaskStatus.BLOCKED
        assert state.get_task_by_id(task_c.task_id).status == TaskStatus.BLOCKED

        # Complete task A
        task_a.status = TaskStatus.COMPLETED
        state.update_task(task_a)
        state.update_dependent_tasks(task_a.task_id)

        # Task B should now be pending, but C should still be blocked
        assert state.get_task_by_id(task_b.task_id).status == TaskStatus.PENDING
        assert state.get_task_by_id(task_c.task_id).status == TaskStatus.BLOCKED

        # Complete task B
        task_b.status = TaskStatus.COMPLETED
        state.update_task(task_b)
        state.update_dependent_tasks(task_b.task_id)

        # Task C should now be pending
        assert state.get_task_by_id(task_c.task_id).status == TaskStatus.PENDING
