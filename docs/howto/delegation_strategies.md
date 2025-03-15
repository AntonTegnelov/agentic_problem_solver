# Delegation Strategies in APS

This document outlines the delegation decision guidelines used in the APS hierarchical agent system. Understanding these strategies is essential for effectively working with the system and extending it with custom delegation logic.

## Overview

The APS hierarchical agent system uses a flexible delegation approach where each agent type (Architect, Planner, Executor) makes delegation decisions based on task complexity and other factors. This agent-driven delegation approach keeps decision-making close to the context where it's most relevant.

## Task Complexity Levels

Task complexity is a key factor in delegation decisions. The system defines four complexity levels:

- **SIMPLE**: Tasks that can be directly executed without further decomposition
- **MODERATE**: Tasks that may benefit from some planning but are relatively straightforward
- **COMPLEX**: Tasks that require significant planning and decomposition
- **VERY_COMPLEX**: Tasks that require multiple levels of planning and decomposition

## Delegation Paths

### ArchitectAgent Delegation

The ArchitectAgent sits at the top of the hierarchy and makes the following delegation decisions:

1. **Direct Delegation to ExecutorAgent**:

   - When task complexity is **SIMPLE** or **MODERATE**
   - Used for well-defined tasks that don't require further planning
   - Example: "Create a simple validation function"

2. **Delegation to PlannerAgent**:
   - When task complexity is **COMPLEX** or **VERY_COMPLEX**
   - Used for tasks that require further decomposition and planning
   - Example: "Design a user authentication system with multiple components"

### PlannerAgent Delegation

The PlannerAgent sits in the middle of the hierarchy and makes the following delegation decisions:

1. **Direct Delegation to ExecutorAgent**:

   - When subtask complexity is **SIMPLE** or **MODERATE**
   - Used for implementable tasks that don't require further planning
   - Example: "Implement a login form component"

2. **Recursive Delegation to Another PlannerAgent**:
   - When subtask complexity is **COMPLEX** or **VERY_COMPLEX**
   - Used for complex sub-components that require specialized planning
   - Example: "Design a database schema for user management"

## Complexity Analysis

### Rule-Based Complexity Analysis

Both ArchitectAgent and PlannerAgent use rule-based approaches to analyze task complexity:

1. **Keyword Matching**: The system looks for specific keywords and phrases that indicate complexity:

   - Simple indicators: "simple", "easy", "straightforward", "basic", etc.
   - Moderate indicators: "moderate", "multiple files", "component", "module", etc.
   - Complex indicators: "complex", "complicated", "system", "integration", etc.
   - Very complex indicators: "very complex", "highly complex", "architecture", etc.

2. **Additional Factors**:

   - Task description length (longer descriptions often indicate more complex tasks)
   - Number of requirements or steps
   - Technical complexity indicators (algorithms, optimization, security, etc.)

3. **Scoring System**:
   - Each indicator contributes to a weighted complexity score
   - Thresholds determine the final complexity level:
     - Score ≤ 3: SIMPLE
     - Score ≤ 6: MODERATE
     - Score ≤ 10: COMPLEX
     - Score > 10: VERY_COMPLEX

## Implementing Custom Delegation Logic

To implement custom delegation logic, you can:

1. **Extend Complexity Analysis**:

   - Modify the rule-based analysis methods in ArchitectAgent and PlannerAgent
   - Add new indicators or adjust weights for existing indicators
   - Implement LLM-based complexity analysis for more nuanced evaluation

2. **Customize Delegation Thresholds**:

   - Adjust when agents delegate to different types based on your specific needs
   - For example, you might want ArchitectAgent to delegate more tasks directly to ExecutorAgent

3. **Add New Delegation Factors**:
   - Consider factors beyond complexity, such as:
     - Task priority
     - Agent specialization
     - Resource availability
     - Execution time constraints

## Delegation Decision Logging

The system logs all delegation decisions for transparency and debugging:

```python
log_delegation_decision(
    logger=self._logger,
    delegation_info=DelegationInfo(
        source_agent_id=self._agent_id,
        target_agent_id=target_agent_id,
        task=task,
        reason=f"Direct delegation to executor due to {complexity.name} complexity",
        additional_info={"task_complexity": complexity.name},
    ),
)
```

This logging provides valuable insights into why specific delegation decisions were made.

## Best Practices

1. **Balance Delegation Depth**:

   - Too shallow: May not break down complex tasks sufficiently
   - Too deep: May create unnecessary overhead for simple tasks

2. **Consider Task Dependencies**:

   - Tasks with many dependencies may benefit from planner involvement
   - Independent tasks can often be delegated directly to executors

3. **Monitor Delegation Patterns**:

   - Review delegation logs to identify patterns and optimize strategies
   - Adjust complexity thresholds based on observed performance

4. **Test Delegation Changes**:
   - Always test changes to delegation logic with a variety of task types
   - Ensure that tasks are appropriately routed to the right agent types
