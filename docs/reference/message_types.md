# Message Types in APS

## Overview

This document provides guidance on the message types used throughout the Agentic Problem Solver (APS) codebase and the proper way to import them to maintain consistency.

## Standard Message Types

APS uses message types from `langchain_core.messages` but re-exports them through the `src.common_types.message_types` module to provide a consistent interface and allow for future extensions.

### Available Message Types

The following message types are available:

- `Message` (alias for `BaseMessage` from langchain_core)
- `AIMessage`
- `HumanMessage`
- `SystemMessage`
- `ToolMessage`

### Type Aliases

Additionally, the following type aliases are defined for use with messages:

- `MessageValue`: Union type for message content (strings, numbers, booleans, dictionaries, lists, or None)
- `CriteriaValue`: Union type for message filtering criteria values
- `CriteriaDict`: Dictionary type for message filtering criteria

## Importing Message Types

### Recommended Import Pattern

Always import message types from `src.common_types.message_types` rather than directly from `langchain_core.messages`:

```python
# Correct import pattern
from src.common_types.message_types import (
    AIMessage,
    CriteriaDict,
    CriteriaValue,
    HumanMessage,
    Message,
    MessageValue,
    SystemMessage,
    ToolMessage,
)
```

### Incorrect Import Pattern

Avoid importing directly from `langchain_core.messages` to prevent inconsistencies:

```python
# Avoid this import pattern
from langchain_core.messages import BaseMessage as Message  # Don't do this
from langchain_core.messages import AIMessage, HumanMessage  # Don't do this
```

## Type Checking

For type checking contexts, you can import the Message type from `src.common_types`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.common_types import Message
```

## Rationale

Centralizing message type imports through `src.common_types.message_types` provides several benefits:

1. **Consistency**: All code uses the same message types
2. **Extensibility**: We can extend or modify message types in one place
3. **Maintainability**: Easier to update dependencies or implementations
4. **Type Safety**: Consistent type annotations throughout the codebase

By following these guidelines, we ensure that the codebase remains consistent and maintainable as it evolves.
