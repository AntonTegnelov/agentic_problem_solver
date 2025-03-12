## TODO

- create CI/CD that

  - [x] automatically runs all test on each commit, pytest and doctest
  - [x] checks for at least 75% test coverage
  - [x] checks that documentation has been written for all parts of the codebase
  - [x] run ruff with strict typing and code must pass without warnings

- enable the ai to break down the prompt into TODO lists following a standardized format

  - [ ] use langchain to do this
  - [ ] Define standardized TODO format schema
  - [ ] Create prompt templates for task breakdown
  - [ ] Implement validation for generated TODOs
  - [ ] Add priority system
  - [ ] Create task dependency tracking
  - [ ] verify that it works
  - [ ] make sure that there is enough unit tests
  - [ ] make sure that there is enough integration tests
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

- enable the top agent to create other agents by itself, as a tool

  - [ ] configure a framework for tools for the ai
  - [ ] Define agent creation interface
  - [ ] Implement agent lifecycle management
  - [ ] Create agent communication protocol
  - [ ] Add resource management for agents
  - [ ] Implement agent capability discovery
  - [ ] Create agent templates
  - [ ] verify that it works
  - [ ] make sure that there is enough unit tests
  - [ ] make sure that there is enough integration tests
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

- combine the two approaches and make the top agent break down the problem into TODO lists and delegate them to new agents

  - [ ] Create delegation strategy
  - [ ] Implement task assignment system
  - [ ] Add progress tracking
  - [ ] Create inter-agent communication system
  - [ ] Implement result aggregation
  - [ ] Add error handling for failed delegations
  - [ ] Create reporting mechanism
  - [ ] verify that it works
  - [ ] make sure that there is enough unit tests
  - [ ] make sure that there is enough integration tests
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

- create a vector database that stores the codebase for RAG

  - [ ] install postgress locally
  - [ ] setup postgress locally
  - [ ] setup postgress as a vector database
  - [ ] run postgress automatically when launching the cli
  - [ ] store code base in vector database
  - [ ] setup RAG
  - [ ] Create indexing system
  - [ ] Implement automatic updates
  - [ ] Add query optimization
  - [ ] Create backup system
  - [ ] Implement caching layer
  - [ ] verify that it works
  - [ ] make sure that there is enough unit tests
  - [ ] make sure that there is enough integration tests
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

- enable the new agents to systematically solve the TODO lists prompting themselves until the TODO is done

  - [ ] Create self-prompting mechanism
  - [ ] Implement progress tracking
  - [ ] Add completion verification
  - [ ] Create feedback loops
  - [ ] Implement retry mechanisms
  - [ ] Add success criteria validation
  - [ ] Create reporting system
  - [ ] verify that it works
  - [ ] make sure that there is enough unit tests
  - [ ] make sure that there is enough integration tests
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

- create a structure where the top agent is asked to break the problem down and delegate it to further agents, each new agent is also either tasked with further breaking down the problem or actually solving the problem

  - [ ] Design hierarchical agent structure
  - [ ] Implement role-based agent system
  - [ ] Create task distribution algorithm
  - [ ] Add coordination mechanism
  - [ ] Implement result aggregation
  - [ ] Create conflict resolution system
  - [ ] Add performance monitoring
  - [ ] verify that it works
  - [ ] make sure that there is enough unit tests
  - [ ] make sure that there is enough integration tests
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

- Implement consistent error handling:

  - [ ] Use existing error types from `common types`:
    - [ ] `APIKeyError` for provider key validation
    - [ ] `ConfigError` for configuration validation
    - [ ] `EmptyResponseError` for response validation
    - [ ] `InvalidModelError` for model validation
    - [ ] `RetryError` for retry handling
    - [ ] `TemperatureError` for parameter validation
  - [ ] Add error recovery mechanisms
  - [ ] Implement proper error logging

- Create feature branches for each major component
- Add PR templates
- Set up automated testing
- Implement version tagging

## Improve Code Structure and Organization

### Message System Improvements

- [x] Create dedicated messaging module
- [x] Implement proper message schemas
- [x] Add message validation
- [x] Extend existing message handlers in `src/messages.py`:
  - [x] Add support for structured message content using existing `MessageValue` type
  - [x] Implement message chain validation using existing `Message` class
  - [x] Add message history tracking with metadata using `additional_kwargs`
  - [x] Create message filtering and search utilities building on `get_message_metadata`
- [x] Improve message flow:
  - [x] Integrate with existing `Agent` protocol in `agent_types.py`
  - [x] Add message routing between agents using `process` and `process_stream` methods
  - [x] Implement message priority handling with `StepResult` type
  - [x] Add message retry and recovery using existing `RetryError` handling

### Provider System Enhancements

- [x] Reorganize provider directory structure
- [x] Remove method duplication
- [x] Standardize provider interfaces
- [x] Improve configuration handling
- [x] Add better error handling
- [x] Complete Provider Factory implementation:
  - [x] Extend existing `LLMProviderFactory` in `factory.py`:
    - [x] Add provider registration validation using `ConfigError`
    - [x] Implement provider configuration validation using `load_config_from_env`
    - [x] Add provider versioning support with `InvalidModelError` handling
    - [x] Create provider dependency resolution using existing config hierarchy
  - [x] Add provider lifecycle management:
    - [x] Implement provider state tracking using `Result[T]` type
    - [x] Add provider health checks with `EmptyResponseError` handling
    - [x] Create provider resource cleanup with proper error handling
  - [x] Improve provider selection:
    - [x] Add provider capability matching using existing config system
    - [x] Implement provider fallback chains with `RetryError` handling
    - [x] Add load balancing support with temperature control (`TemperatureError`)

### Agent System Improvements

- [x] Enhance agent state management:
  - [x] Extend `AgentState` class with more context tracking
  - [x] Add state validation using existing error types
  - [x] Implement state persistence
- [x] Improve agent step processing:
  - [x] Extend step prompts in `prompts.py` (UNDERSTAND, PLAN, EXECUTE, VERIFY)
  - [x] Add step validation using `AgentStep` enum
  - [x] Implement step retry mechanisms
- [x] Add agent communication:
  - [x] Create agent message protocol using `Message` class
  - [x] Implement agent discovery
  - [x] Add agent coordination

### Testing and Documentation

- [ ] Extend test coverage:
  - [x] Add unit tests for message system using existing types
  - [x] Add unit tests for agent state management
  - [x] Add unit tests for agent step processing
  - [x] Add unit tests for agent coordination
  - [ ] Create integration tests for provider factory
- [x] Improve documentation:
  - [x] Update architecture documentation with current components
  - [x] Create API reference for message system
  - [x] Create API reference for agent system
  - [ ] Document provider factory usage
  - [ ] Add examples for agent communication patterns
  - [ ] Document error handling using existing exception types
  - [ ] Add configuration examples using `load_env_var` and `load_config_from_env`

### Code Quality

- [ ] Implement consistent error handling:
  - [ ] Use existing error types from `common_types`:
    - [ ] `APIKeyError` for provider key validation
    - [ ] `ConfigError` for configuration validation
    - [ ] `EmptyResponseError` for response validation
    - [ ] `InvalidModelError` for model validation
    - [ ] `RetryError` for retry handling
    - [ ] `TemperatureError` for parameter validation
  - [ ] Add error recovery mechanisms
  - [ ] Implement proper error logging

### Git Workflow

- [ ] Create feature branches for each major component
- [ ] Add PR templates with testing requirements
- [ ] Set up automated testing with existing test structure
- [ ] Implement version tagging

### Additional testing

- add a few well chosen end-to-end tests
  - [ ] Design comprehensive test scenarios
  - [ ] Create test data and fixtures
  - [ ] Implement user journey tests
  - [ ] Implement reliability tests
  - [ ] Add integration points coverage
  - [ ] Create data persistence tests
  - [ ] Implement error recovery scenarios
  - [ ] Add concurrency tests
  - [ ] Create test reporting system
  - [ ] verify that it works
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

### Write Documentation

- [ ] Tutorials: Lessons that take the reader by the hand through a series of steps to complete a project (or meaningful exercise). Geared towards the user's learning.
- [ ] How-To Guides: Guides that take the reader through the steps required to solve a common problem (problem-oriented recipes).
- [x] References: Explanations that clarify and illuminate a particular topic. Geared towards understanding.
  - [x] Message System Reference
  - [x] Agent System Reference
  - [ ] Provider System Reference
  - [ ] Configuration System Reference
- [ ] Explanations: Technical descriptions of the machinery and how to operate it (key classes, functions, APIs, and so forth). Think Encyclopedia article.
- [ ] examples.py: A Python script file that gives simple examples of how to use the project.
- [ ] implement doctest.py so that code examples are correct
