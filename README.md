# Agno Code

My attempt to port Claude Code so I can use my free API keys instead of Claude. (Who doesn't love free?)

TS better be done within the next month becuase I need this bad.

## Deployment

- Web app
- Local Application (Tauri or Electron)
- OAuth with DBs to protect API keys, save user configs, 
- Some good UI (I hope)
- Javascript + Python API stack (look into Bun, how to make applications)
- Examine the Agno OS UI and base it loosely on that
- Eventually (later) integrate with IDE extensions

## Claude Code Features

### Memory, Commands, Hooks
- Memory for system prompts (Claude.md)

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
- Subagents (better designed, more readily used vs Claude Code)
- Everything is in the name of making context LEAN
- Options for detailed planning with or without HITL (equivalent to Claude Code's planning mode)
- LATER: Build some pre-existing agent "profiles" (code reviewer, planner for UI, planner for Backend Infra, etc.) 

### Enhanced Search
- Basic web search
- Exa search
- Git and GitHub repo searching (pyGithub)
- Documentation Search: [Context7](https://github.com/upstash/context7) MCP for documentation (better yet, make it a custom agent skill) 
- Browserbase / Stagehand Search **(tentative)**

### Writing Code
- Expect the agent to give plans and justify every change
- Optionality for full rewrites, diffs, suggestions


### Model Agnosticism
- Support for any OpenAI API-adhering models
- Rate limiting
- Multi API key cycling support (with limits tracking)
- Support for ALL hyperparameters (top p, temperature, max_tokens, freq_penalty, etc.)
- Config profiles for model presets and all options in general (ex. for zai-glm-4.6, temperature: 0.6 and max_completion_tokens: 5000)

### Dynamic Knowledge 
- Dynamic knowledge graph for projects / directories using Graphiti + FalkorDB (handled by an ingestion agent and a query agent)
- Custom GUI for knowledge graph within the app
- Code Indexing for faster, better retrieval (investigate [KotaDB: Code Intelligence Engine](https://github.com/jayminwest/kotadb))
- Leverage LSP, other useful tools? [Serena: Coding Agent Toolkit](https://www.reddit.com/r/agno/s/oMaE1A5zqa)

### Memory, Commands, Hooks
- Memory: Agno.md system prompts (ex. never use Any for Typescript)
- [Commands](https://github.com/wshobson/commands/tree/main): Make command md files or scripts (commands directory? commands skill? we'll see)
- Essentially have a system for setting "rules" (could be same as Claude Code, Windsurf, or could be my own setup)

### Plug and Play with existing Claude Code features
- Complete integration of agent skills (guided file reads)
- MCP Config file integration (dedicated MCP agent that routes to "MCP storage" agents that have sets of MCPs in context to minimize context bloat of main agent)

- Integration of [claude code plugins](https://claude-plugins.dev/skills) 
- Very important as plugins contains Commands, Agent Sys Prompts, MCP config, Tools, and Skills
- Use [Edmund's setup](https://github.com/edmund-io/edmunds-claude-code) to test Agno ingestion

- Package Manager for skills and / or plugins **(tentative)**

### Plug and Play with Agno
- Port main AgnoOS features (knowledge, sessions, db, memory tabs)
- Support for Agno agents, teams, workflows (hybrid of this with Claude Code)