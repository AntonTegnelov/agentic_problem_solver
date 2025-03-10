##TODO

- URGENT: Fix Code Duplication

  - [x] Consolidate State Management

    - [x] Keep single AgentState class in src/agent/state/base.py
    - [x] Remove duplicate from src/agent/agent_types/**init**.py
    - [x] Update all imports to use single source

  - [x] Unify Configuration System

    - [x] Create single configuration hierarchy
    - [x] Move provider-specific config to be extensions of base config
    - [x] Implement proper inheritance to avoid duplication

  - [x] Clean up Provider Implementation

    - [x] Move common functionality to base class
    - [x] Create proper abstractions for error handling and retries
    - [x] Standardize configuration handling

  - [x] Reorganize Configuration Files
    - [x] Move src/config.py contents to src/config/defaults.py
    - [x] Move enums (MessageRoles, AgentStep) to src/types/enums.py
    - [x] Update all imports to use new file locations
    - [x] Remove old config.py file

- Install langraph + langchain

  - [x] Install langraph
  - [x] Install langchain
  - [x] Verify installations work correctly
  - [x] Set up basic environment configuration

- setup git

  - [x] setup git
  - [x] push to github using their CLI
  - [x] update gitignore

- setup an easy way to switch llm provider

  - [x] use google gemini ai as it's the cheapest for now
  - [x] Create abstraction layer for LLM providers
  - [x] Implement configuration system for API keys
  - [x] Add support for environment variables
  - [x] Create provider switching mechanism
  - [x] Add fallback mechanisms
  - [x] verify that it works
  - [x] write unit tests
  - [x] write integration tests

- improve code structure and organization

  - [x] Unified Configuration Management

    - [x] Create src/config/ directory structure
    - [x] Move all configuration to dedicated modules
    - [x] Implement dataclass-based config objects
    - [x] Remove configuration duplication
    - [x] Standardize configuration parameters
    - [x] Add configuration validation
    - [x] Update all modules to use new config system

  - [x] Agent Architecture Improvements

    - [x] Create dedicated state management
    - [x] Implement flexible step processing
    - [x] Separate agent core from implementations
    - [x] Create proper abstraction layers
    - [x] Improve error handling
    - [x] Add better type safety
    - [x] Implement proper dependency injection

  - [ ] Provider System Enhancements

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

  - [ ] Message System Improvements

    - [x] Create dedicated messaging module
    - [x] Implement proper message schemas
    - [x] Add message validation
    - [ ] Extend existing message handlers in `src/messages.py`:
      - [ ] Add support for structured message content using existing `MessageValue` type
      - [ ] Implement message chain validation using existing `Message` class
      - [ ] Add message history tracking with metadata using `additional_kwargs`
      - [ ] Create message filtering and search utilities building on `get_message_metadata`
    - [ ] Improve message flow:
      - [ ] Integrate with existing `Agent` protocol in `agent_types.py`
      - [ ] Add message routing between agents using `process` and `process_stream` methods
      - [ ] Implement message priority handling with `StepResult` type
      - [ ] Add message retry and recovery using existing `RetryError` handling

  - [ ] Testing and Documentation

    - [ ] Add unit tests for new structure
      - [ ] Write tests for provider factory
      - [ ] Write tests for message handlers
      - [ ] Write tests for message flow
      - [ ] Write tests for configuration system
      - [ ] Write tests for error handling
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Create integration tests
      - [ ] Test provider factory with different providers
      - [ ] Test message flow with different handlers
      - [ ] Test configuration system with different configs
      - [ ] Test error handling with different scenarios
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add documentation for new architecture
      - [ ] Document provider factory
      - [ ] Document message handlers
      - [ ] Document message flow
      - [ ] Document configuration system
      - [ ] Document error handling
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Extend test coverage:
      - [ ] Add unit tests for message system using existing types
      - [ ] Create integration tests for provider factory
      - [ ] Add end-to-end agent communication tests
      - [ ] have a test coverage of at least 75%
    - [ ] Improve documentation:
      - [ ] Update architecture documentation with current components
      - [ ] Create API reference for message system
      - [ ] Document provider factory usage
      - [ ] Add examples for agent communication patterns
      - [ ] Document error handling using existing exception types
      - [ ] Add configuration examples using `load_env_var` and `load_config_from_env`

- create a top level agent that recieves a prompt

  - [x] create a CLI
  - [x] create an AI agent with langgraph
  - [x] Implement input validation
  - [x] Add error handling
  - [x] Create logging system
  - [x] Add configuration options for the agent
  - [x] Implement prompt templates
  - [ ] Enable use of the program by writing something in the terminal like: APS solve "Create a calculator app in Typescript and deploy it to amazon"
  - [ ] verify that it works
  - [ ] write unit tests
  - [ ] write integration tests
  - [ ] run ruff and fix all warnings

  write documentation

  - [ ] Tutorials: Lessons that take the reader by the hand through a series of steps to complete a project (or meaningful exercise). Geared towards the user's learning.
  - [ ] How-To Guides: Guides that take the reader through the steps required to solve a common problem (problem-oriented recipes).
  - [ ] References: Explanations that clarify and illuminate a particular topic. Geared towards understanding.
  - [ ] Explanations: Technical descriptions of the machinery and how to operate it (key classes, functions, APIs, and so forth). Think Encyclopedia article.
  - [ ] examples.py: A Python script file that gives simple examples of how to use the project.
  - [ ] implement doctest.py so that code examples are correct

- create CI/CD that

  - [ ] automatically runs all test on each commit, pytest and doctest
  - [ ] checks for at least 75% test coverage
  - [ ] checks that documentation has been written for all parts of the codebase
  - [ ] run ruff with strict typing and code must pass without warnings

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

- enable the new agents to systematically solve the TODO lists propmting themselves until the TODO is done

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

- add a few well chosen end-to-end tests

  - [ ] Design comprehensive test scenarios
  - [ ] Create test data and fixtures
  - [ ] Implement user journey tests
  - [ ] Add performance benchmarks
  - [ ] Create stress tests
  - [ ] Implement reliability tests
  - [ ] Add integration points coverage
  - [ ] Create data persistence tests
  - [ ] Implement error recovery scenarios
  - [ ] Add concurrency tests
  - [ ] Setup test environment automation
  - [ ] Create test reporting system
  - [ ] verify that it works
  - [ ] update documentation
  - [ ] verify the feature passes CI/CD

- Implement consistent error handling:

  - [ ] Use existing error types from `exceptions.py`:
    - [ ] `APIKeyError` for provider key validation
    - [ ] `ConfigError` for configuration validation
    - [ ] `EmptyResponseError` for response validation
    - [ ] `InvalidModelError` for model validation
    - [ ] `RetryError` for retry handling
    - [ ] `TemperatureError` for parameter validation
  - [ ] Add error recovery mechanisms
  - [ ] Implement proper error logging

- Improve type safety:

  - [ ] Extend existing generic types in `agent_types.py`:
    - [ ] `Agent[T, U]` for new agent types
    - [ ] `Result[T]` for operation results
    - [ ] `Message` for communication
  - [ ] Add runtime type validation
  - [ ] Implement stricter type bounds

- Add performance optimizations:

  - [ ] Implement message batching using existing message system
  - [ ] Add caching for provider factory
  - [ ] Optimize message routing using existing protocols

- Create feature branches for each major component
- Add PR templates
- Set up automated testing
- Implement version tagging

## Improve Code Structure and Organization

### Message System Improvements

- [x] Create dedicated messaging module
- [x] Implement proper message schemas
- [x] Add message validation
- [ ] Extend existing message handlers in `src/messages.py`:
  - [ ] Add support for structured message content using existing `MessageValue` type
  - [ ] Implement message chain validation using existing `Message` class
  - [ ] Add message history tracking with metadata using `additional_kwargs`
  - [ ] Create message filtering and search utilities building on `get_message_metadata`
- [ ] Improve message flow:
  - [ ] Integrate with existing `Agent` protocol in `agent_types.py`
  - [ ] Add message routing between agents using `process` and `process_stream` methods
  - [ ] Implement message priority handling with `StepResult` type
  - [ ] Add message retry and recovery using existing `RetryError` handling

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

- [ ] Enhance agent state management:
  - [ ] Extend `AgentState` class with more context tracking
  - [ ] Add state validation using existing error types
  - [ ] Implement state persistence
- [ ] Improve agent step processing:
  - [ ] Extend step prompts in `prompts.py` (UNDERSTAND, PLAN, EXECUTE, VERIFY)
  - [ ] Add step validation using `AgentStep` enum
  - [ ] Implement step retry mechanisms
- [ ] Add agent communication:
  - [ ] Create agent message protocol using `Message` class
  - [ ] Implement agent discovery
  - [ ] Add agent coordination

### Testing and Documentation

- [ ] Extend test coverage:
  - [ ] Add unit tests for message system using existing types
  - [ ] Create integration tests for provider factory
  - [ ] Add end-to-end agent communication tests
  - [ ] Implement performance benchmarks
- [ ] Improve documentation:
  - [ ] Update architecture documentation with current components
  - [ ] Create API reference for message system
  - [ ] Document provider factory usage
  - [ ] Add examples for agent communication patterns
  - [ ] Document error handling using existing exception types
  - [ ] Add configuration examples using `load_env_var` and `load_config_from_env`

### Code Quality

- [ ] Implement consistent error handling:
  - [ ] Use existing error types from `exceptions.py`:
    - [ ] `APIKeyError` for provider key validation
    - [ ] `ConfigError` for configuration validation
    - [ ] `EmptyResponseError` for response validation
    - [ ] `InvalidModelError` for model validation
    - [ ] `RetryError` for retry handling
    - [ ] `TemperatureError` for parameter validation
  - [ ] Add error recovery mechanisms
  - [ ] Implement proper error logging
- [ ] Improve type safety:
  - [ ] Extend existing generic types in `agent_types.py`:
    - [ ] `Agent[T, U]` for new agent types
    - [ ] `Result[T]` for operation results
    - [ ] `Message` for communication
  - [ ] Add runtime type validation
  - [ ] Implement stricter type bounds
- [ ] Add performance optimizations:
  - [ ] Implement message batching using existing message system
  - [ ] Add caching for provider factory
  - [ ] Optimize message routing using existing protocols

### Git Workflow

- [ ] Create feature branches for each major component
- [ ] Add PR templates with testing requirements
- [ ] Set up automated testing with existing test structure
- [ ] Implement version tagging
