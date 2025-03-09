#Agentic Problem Solver
For every checked mark, please commit it to git
##TODO

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

- create a top level agent that recieves a prompt

  - [x] create a CLI
  - [x] create an AI agent with langgraph
  - [ ] Implement input validation
  - [ ] Add error handling
  - [ ] Create logging system
  - [ ] Add configuration options for the agent
  - [ ] Implement prompt templates
  - [ ] verify that it works
  - [ ] write unit tests
  - [ ] write integration tests

- create CI/CD that automatically runs all test

- enable the ai to break down the prompt into TODO lists following a standardized format

  - [ ] use langchain to do this
  - [ ] Define standardized TODO format schema
  - [ ] Create prompt templates for task breakdown
  - [ ] Implement validation for generated TODOs
  - [ ] Add priority system
  - [ ] Create task dependency tracking
  - [ ] verify that it works
  - [ ] write unit tests
  - [ ] write integration tests

- enable the top agent to create other agents by itself, as a tool

  - [ ] configure a framework for tools for the ai
  - [ ] Define agent creation interface
  - [ ] Implement agent lifecycle management
  - [ ] Create agent communication protocol
  - [ ] Add resource management for agents
  - [ ] Implement agent capability discovery
  - [ ] Create agent templates
  - [ ] verify that it works
  - [ ] write unit tests
  - [ ] write integration tests

- combine the two approaches and make the top agent break down the problem into TODO lists and delegate them to new agents

  - [ ] Create delegation strategy
  - [ ] Implement task assignment system
  - [ ] Add progress tracking
  - [ ] Create inter-agent communication system
  - [ ] Implement result aggregation
  - [ ] Add error handling for failed delegations
  - [ ] Create reporting mechanism
  - [ ] verify that it works
  - [ ] write unit tests
  - [ ] write integration tests

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
  - [ ] write unit tests
  - [ ] write integration tests

- enable the new agents to systematically solve the TODO lists propmting themselves until the TODO is done

  - [ ] Create self-prompting mechanism
  - [ ] Implement progress tracking
  - [ ] Add completion verification
  - [ ] Create feedback loops
  - [ ] Implement retry mechanisms
  - [ ] Add success criteria validation
  - [ ] Create reporting system
  - [ ] verify that it works
  - [ ] write unit tests
  - [ ] write integration tests

- create a structure where the top agent is asked to break the problem down and delegate it to further agents, each new agent is also either tasked with further breaking down the problem or actually solving the problem
  - [ ] Design hierarchical agent structure
  - [ ] Implement role-based agent system
  - [ ] Create task distribution algorithm
  - [ ] Add coordination mechanism
  - [ ] Implement result aggregation
  - [ ] Create conflict resolution system
  - [ ] Add performance monitoring
  - [ ] verify that it works
  - [ ] write unit tests
  - [ ] write integration tests
