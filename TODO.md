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
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Move all configuration to dedicated modules
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Implement dataclass-based config objects
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Remove configuration duplication
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Standardize configuration parameters
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Add configuration validation
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [ ] Update all modules to use new config system
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] add, commit and push git

  - [x] Agent Architecture Improvements

    - [x] Create dedicated state management
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Implement flexible step processing
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Separate agent core from implementations
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Create proper abstraction layers
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Improve error handling
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Add better type safety
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git
    - [x] Implement proper dependency injection
      - [x] Do the task
      - [x] ruff check .
      - [x] add, commit and push git

  - [ ] Provider System Enhancements

    - [ ] Reorganize provider directory structure
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Remove method duplication
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Standardize provider interfaces
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Improve configuration handling
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add better error handling
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Create proper provider factory
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Implement provider-specific validation
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git

  - [ ] Message System Improvements

    - [ ] Create dedicated messaging module
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Implement proper message schemas
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add message validation
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Create message handlers
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Improve message flow
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add message transformation utilities

  - [ ] Testing and Documentation

    - [ ] Add unit tests for new structure
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Create integration tests
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add documentation for new architecture
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Create architecture diagrams
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add code examples
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Update existing documentation
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git

  - [ ] Code Quality
    - [ ] Run type checking
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Fix all linter warnings
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add proper error messages
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Improve logging
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Add performance monitoring
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git
    - [ ] Create debug utilities
      - [ ] Do the task
      - [ ] ruff check .
      - [ ] try running APS solve "say hello"
      - [ ] add, commit and push git

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
