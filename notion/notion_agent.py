from agno.agent import Agent
from agno.tools.notion import NotionTools
from agno.models.groq import Groq
from agno.run.agent import RunInput
from key_manager import MultiProviderWrapper
from agno.utils.log import log_debug, log_error

import os

ENV_FILE="../env/.env"
DB_FILE="../env/api_usage.db"
README_PATH="../README.md"

groq_wrapper = MultiProviderWrapper.from_env(
    provider='groq',
    model_class=Groq,
    model_id='llama-3.3-70b-versatile',
    env_file=ENV_FILE,
    db_path=DB_FILE
)

def get_project_readme(run_input: RunInput, file_path=README_PATH):
    log_debug("Adding project readme to context window")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            run_input.input_content += f"""
PROJECT README:
{content}
"""
        except Exception as e:
            log_error(f"Error reading README file: {e}")
    else:
        log_debug(f"README file not found at path: {file_path}")

notion_agent = Agent(
    name="Notion Project Manager",
    description="An agent that can manage Notion databases and pages for my project by referencing the README file.",
    instructions=[
        "You are an expert Technical Product Manager responsible for maintaining the 'Agno Code' Kanban board in Notion.",
        "Your goal is to parse the project README into structured database entries for tasks.",
        
        "**Content Parsing Strategy:**",
        "1. Treat main headers (e.g., 'Deployment', 'Claude Code Features', 'Agno Code Extensions') as 'Epics' or 'Categories'.",
        "2. Treat bullet points as individual 'Tasks' or 'User Stories'.",
        "3. Extract technical details (e.g., 'Tauri', 'Bun', 'FalkorDB') to populate a 'Tech Stack' or 'Context' property if available.",

        "**Priority & Status Logic:**",
        "1. Analyze text for urgency cues. Phrases like 'done within the next month' or 'need this bad' indicate 'High Priority'.",
        "2. Items marked 'Eventually' or 'Later' should be assigned to the 'Backlog'.",
        "3. Standard tasks should default to 'To Do' status.",

        "**Execution Rules:**",
        "1. **Search First:** Before creating any card, ALWAYS search the Notion database for existing tasks with similar names to prevent duplicates.",
        "2. **Atomic Updates:** If a task exists, update its content/description rather than creating a new entry.",
        "3. **Tagging:** Apply tags based on the section the item came from (e.g., tag 'Multi-Agent', 'Search', or 'Shell Tools').",
        
        "Be concise in your confirmation actions, summarizing how many tasks were created or updated per category.",
    ],
    tools=[NotionTools()],
    markdown=True,
    model=groq_wrapper.get_model(wait=False, id="moonshotai/kimi-k2-instruct-0905")[0],
    debug_level=2,
    debug_mode=True,
    pre_hooks=[get_project_readme],
    add_history_to_context=True,
    num_history_messages=6,
    read_chat_history=True,
    search_session_history=True,
    num_history_sessions=2
)

from agno.os import AgentOS



if __name__ == "__main__":
    
    agent_os = AgentOS(agents=[notion_agent],
        id="notion_agent_os",
        name="Notion Agent OS",  
        # run_hooks_in_background=True, 
    )

    app = agent_os.get_app()
    
    agent_os.serve(app="notion_agent:app", reload=True, port=8001, workers=3)