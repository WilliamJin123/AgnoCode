# Agno Code

My attempt to port Claude Code so I can use my free API keys instead of Claude. (Who doesn't love free?)

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

### Companion / Sidecar Agent
- Separate agent that follows along with Agno Code's outputs / operations from the user's perspective
- Answers conceptual questions / clarifications (ex. what does this code snippet do, what is this syntax exactly, etc.) without diluting workflow context
- Potentially: Redefine user prompts, intermediary between user and Agno Code

### Shell Tools
- Package management: npm, pip, etc. (guided shell tools)
- Testing and debugging (guided shell tools, file operations: Pytest, Jest, JUnit, etc.)
- Access to your files: file search, reading, tree command, etc. (guided shell tools)
- General commands
- HITL / guardrails for potentially dangerous commands

### Github Integrations
- PyGithub tools (for reading, creating repos, cloning repos, etc.)

### Multi-Agent Architecture
- Everything is in the name of making context LEAN
- Options for detailed planning with or without HITL (equivalent to Claude Code's planning mode)
- Experiment with Swarm, Evolutionary, Genetic, and other algorithms
- Create a memory framework that unifies tools / toolkits, behavioural profiles (system prompts, instructions), knowledge (additional context, RAG-related information), client preferences (what each user needs), sessions (session state, session summaries), temporal context, and adaptive / iterative learning mechanisms

### Enhanced Search
- Basic web search (for now, might even remove this just because its so naive)
- Exa search
- Git and GitHub repo searching (pyGithub)
- Documentation Search: [Context7](https://github.com/upstash/context7) MCP for documentation (better yet, make it a custom agent skill) 
- Browserbase / [Stagehand](https://github.com/browserbase/stagehand) Search 

### Writing Code
- Expect the agent to give plans and justify every change
- Optionality for full rewrites, diffs, suggestions

### Models
- Support for any OpenAI API-adhering models
- Multi API key cycling support (with limits tracking)
- Support for ALL hyperparameters (top p, temperature, max_tokens, freq_penalty, etc.)
- Config profiles for model presets and all options in general (ex. for zai-glm-4.6, temperature: 0.6 and max_completion_tokens: 5000)
- IN THE FUTURE: RL with smaller models for specific tasks to beat out frontier models using [Verifiers from PrimeIntellect](https://github.com/PrimeIntellect-ai/verifiers)

### Dynamic Knowledge 
- Hybrid RAG: (Trigram similarity + Graph + Agentic + Keyword search, etc.)
- Dynamic knowledge graph for projects / directories using Graphiti + FalkorDB (handled by an ingestion agent and a query agent)
- [Script to check for hallucinations](https://github.com/coleam00/mcp-crawl4ai-rag/tree/main/knowledge_graphs) using the knowledge graph
- Code Indexing for faster, better retrieval (investigate [KotaDB: Code Intelligence Engine](https://github.com/jayminwest/kotadb))
- Leverage LSP, other useful tools? [Serena: Coding Agent Toolkit](https://www.reddit.com/r/agno/s/oMaE1A5zqa)

### Memory, Commands, Hooks
- Memory: Agno.md system prompts (ex. never use Any for Typescript)
- [Commands](https://github.com/wshobson/commands/tree/main): Make command md files or scripts (commands directory? commands skill? we'll see)
- Essentially have a system for setting "rules" (could be same as Claude Code, Windsurf, or could be my own setup)

### Plug and Play with existing Claude Code features
- Complete integration of agent skills
- MCP Config file integration (nested MCP agent routers)
- Integration of [claude code plugins](https://claude-plugins.dev/skills) 
- Use [Edmund's setup](https://github.com/edmund-io/edmunds-claude-code) to test

- Package Manager for skills and / or plugins: make my own or use [craftdesk](https://github.com/mensfeld/craftdesk)

### Plug and Play with Agno
- Port main AgnoOS features (knowledge, sessions, db, memory tabs)
- Support for Agno agents, teams, workflows (hybrid of this with Claude Code)