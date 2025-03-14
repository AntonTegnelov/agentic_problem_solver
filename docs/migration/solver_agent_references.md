# SolverAgent References in Codebase

This document tracks all references to the deprecated `SolverAgent` class in the codebase. These references need to be updated as part of the migration to the hierarchical agent system.

## Implementation

- `src/agent/solver.py`: Main implementation of the `SolverAgent` class

## Direct Usage

- `src/cli/main.py`:
  - Line ~122: `agent = SolverAgent(provider=provider, state_manager=state_manager, config=config)`
  - Line ~164: `agent = SolverAgent(provider=provider)`

## Tests

- `tests/unit/test_solver_agent.py`: Comprehensive unit tests for `SolverAgent`
- `tests/unit/test_cli_main.py`: Tests for CLI that use `SolverAgent`
- `tests/integration/test_cli_end_to_end.py`: Integration tests for CLI that use `SolverAgent`

## Documentation

- `docs/howto/migration.md`: Migration guide from `SolverAgent` to hierarchical agents
- `docs/explanation/architecture.md`: Architecture documentation mentioning `SolverAgent`
- `docs/reference/api.md`: API reference for `SolverAgent`
- `docs/tutorials/quickstart.md`: Quickstart guide mentioning `SolverAgent`
- `docs/howto/common_tasks.md`: Common tasks documentation using `SolverAgent`
- `README.md`: Project README with examples using `SolverAgent`

## Examples

- `docs/examples.py`: May contain examples using `SolverAgent`

## Migration Strategy

Each reference to `SolverAgent` needs to be addressed in the following ways:

1. **Implementation**: Gradually update `SolverAgent` internals to delegate to hierarchical agents
2. **Direct Usage**: Replace with appropriate hierarchical agent factory methods
3. **Tests**: Update tests to use hierarchical agents or mark as deprecated
4. **Documentation**: Update to focus on hierarchical agents and mark `SolverAgent` examples as deprecated

## Progress Tracking

- [ ] Update `SolverAgent` implementation to delegate to hierarchical agents
- [ ] Update CLI code to use hierarchical agents
- [ ] Update or deprecate tests for `SolverAgent`
- [ ] Update documentation to focus on hierarchical agents
- [ ] Remove all traces of `SolverAgent` from the codebase
