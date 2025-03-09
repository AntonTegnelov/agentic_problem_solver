#Agentic Problem Solver

- Install langraph + langchain

  - [x] Install langraph
  - [x] Install langchain
  - [x] Verify installations work correctly
  - [x] Set up basic environment configuration

- setup an easy way to switch llm provider

  - [ ] use google gemini ai as it's the cheapest for now
  - [ ] Create abstraction layer for LLM providers
  - [ ] Implement configuration system for API keys
  - [ ] Add support for environment variables
  - [ ] Create provider switching mechanism
  - [ ] Add fallback mechanisms

- create a top level agent that recieves a prompt

  - [ ] create a CLI
  - [ ] create an AI agent with langgraph
  - [ ] Implement input validation
  - [ ] Add error handling
  - [ ] Create logging system
  - [ ] Add configuration options for the agent
  - [ ] Implement prompt templates

- enable the ai to break down the prompt into TODO lists following a standardized format

  - [ ] use langchain to do this
  - [ ] Define standardized TODO format schema
  - [ ] Create prompt templates for task breakdown
  - [ ] Implement validation for generated TODOs
  - [ ] Add priority system
  - [ ] Create task dependency tracking

- enable the top agent to create other agents by itself, as a tool

  - [ ] configure a framework for tools for the ai
  - [ ] Define agent creation interface
  - [ ] Implement agent lifecycle management
  - [ ] Create agent communication protocol
  - [ ] Add resource management for agents
  - [ ] Implement agent capability discovery
  - [ ] Create agent templates

- combine the two approaches and make the top agent break down the problem into TODO lists and delegate them to new agents

  - [ ] Create delegation strategy
  - [ ] Implement task assignment system
  - [ ] Add progress tracking
  - [ ] Create inter-agent communication system
  - [ ] Implement result aggregation
  - [ ] Add error handling for failed delegations
  - [ ] Create reporting mechanism

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

- enable the new agents to systematically solve the TODO lists propmting themselves until the TODO is done

  - [ ] Create self-prompting mechanism
  - [ ] Implement progress tracking
  - [ ] Add completion verification
  - [ ] Create feedback loops
  - [ ] Implement retry mechanisms
  - [ ] Add success criteria validation
  - [ ] Create reporting system

- create a structure where the top agent is asked to break the problem down and delegate it to further agents, each new agent is also either tasked with further breaking down the problem or actually solving the problem
  - [ ] Design hierarchical agent structure
  - [ ] Implement role-based agent system
  - [ ] Create task distribution algorithm
  - [ ] Add coordination mechanism
  - [ ] Implement result aggregation
  - [ ] Create conflict resolution system
  - [ ] Add performance monitoring
