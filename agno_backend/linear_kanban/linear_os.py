

from agno.utils.log import log_debug, log_error
from agno.os import AgentOS
from agent import get_linear_agent, get_linear_team

import os

ENV_FILE="../env/.env"
DB_FILE="../env/api_usage.db"
README_PATH="../README.md"

linear_agent = get_linear_agent(debug_level=2, debug_mode=True)
linear_team = get_linear_team()

agent_os = AgentOS(
        agents=[linear_agent],
        teams=[linear_team],
        id="notion_agent_os",
        name="Notion Agent OS",  
        # run_hooks_in_background=True, 
    )

app = agent_os.get_app()

if __name__ == "__main__":
    
    agent_os.serve(app="linear_os:app", reload=True, port=8001, workers=3)