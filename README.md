# Agno Code

My attempt to port Claude Code so I can use my free API keys instead of Claude. (Who doesn't love free?)

TS better be done within the next month becuase I need this bad.

## Deployment

- Web app
- Local Applications
- OAuth with DBs to protect API keys, save user configs, 
- Some good UI (I hope)
- Javascript + Python API stack (look into Bun, how to make applications)
- Examine the Agno OS UI and base it loosely on that
- Eventually (later) integrate with IDE extensions

## Claude Code Features

### File System 
- Access to your files: file search, reading 
- Code editing and file creation capabilities 

### Terminal & Execution
- Ability to run commands 
- Package management: npm, pip, etc.
- Testing and debugging 

### Integration & Configuration
- MCP config file for Model Context Protocol servers (npx add whatever)
- Skills directories with skill.md files
- Plugins with skills and/or agents and/or system prompts

### Context & Search
- Web search for documentation and current information (grounding)

---

# Agno Code Extensions (Beyond Claude Code)

### Shell Tools
- Package management: npm, pip, etc. (guided shell tools)
- Testing and debugging (guided shell tools, file operations: Pytest, Jest, JUnit, etc.)
- Access to your files: file search, reading, tree command, etc. (guided shell tools)
- General commands
- HITL / guardrails for potentially dangerous commands

### Github Integrations
- PyGithub tools (for reading, creating repos, cloning repos, etc.)

### Multi-Agent Architecture
- Ability to parallelize agent tasks (inherent)
- Options for detailed planning with or without HITL

### Enhanced Search
- Basic web search
- Exa search
- Git and GitHub repo searching (pyGithub)

### Writing Code
- Expect the agent to give plans and justify every change
- Optionality for full rewrites, diffs, suggestions


### Model Agnosticism
- Support for any OpenAI API-adhering models
- Rate limiting
- Multi API key cycling support (with limits tracking)
- Support for ALL hyperparameters (top p, temperature, max_tokens, freq_penalty, etc.)
- Config profiles for model presets and all options in general (ex. for zai-glm-4.6, temperature: 0.6 and max_completion_tokens: 5000)

### Dynamic Knowledge Management
- Dynamic knowledge graph for projects / directories using Graphiti + FalkorDB (handled by an ingestion agent and a query agent)
- Also have a standard indexed DB perhaps

### Plug and Play with existing Claude Code features
- Complete integration of agent skills (guided file reads)
- MCP Config file integration (dedicated MCP agent that routes to "MCP storage" agents that have sets of MCPs in context to minimize context bloat of main agent)
- Integration of claude code plugins (we will see about this one)