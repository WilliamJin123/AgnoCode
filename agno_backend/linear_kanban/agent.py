from agno.agent import Agent
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing
from agno.session import SessionSummaryManager
from agno.compression import CompressionManager
import os

from architect_tools import LinearTools
from key_manager import MultiProviderWrapper

ENV_FILE="../../api_management/.env"
# DB_FILE="../env/api_usage.db"
README_PATH="../../README.md"

LINEAR_API_KEY=os.getenv("LINEAR_API_KEY")

def read_readme(path: str = README_PATH) -> str:
    """
    Reads the content of the project README file.
    Args:
        path (str): This value defaults to a preset README_PATH environment variable. Only supply the path if explicitly instructed to by the user.

    Returns:
        str: The full text content of the README.md file.
    """
    try:
        with open(path, 'r') as f: 
            return f.read()
    except FileNotFoundError: 
        return "Error: README.md not found."

linear_tools = LinearTools(api_key=LINEAR_API_KEY)

groq_wrapper = MultiProviderWrapper.from_env(
    provider='groq',
    default_model_id='moonshotai/kimi-k2-instruct-0905',
    env_file=ENV_FILE,

)

cerebras_wrapper = MultiProviderWrapper.from_env(
    provider='cerebras',
    default_model_id='llama-3.3-70b',
    env_file=ENV_FILE,
)

s_sum_manager = SessionSummaryManager(
    model=groq_wrapper.get_model(
        id="llama-3.3-70b-versatile", estimated_tokens=500
    ),
    summary_request_message="Provide the summary of the conversation. Include all relevant details",
    
)

tool_compression_manager = CompressionManager(
    model=groq_wrapper.get_model(
        id="llama-3.3-70b-versatile", estimated_tokens=2000
    ),
    compress_tool_results_limit=4
)

desc="You sync a README to Linear, creating labels for categories."

team_prompt=[
    "You are the Team Lead for Linear synchronization.",
    "You have direct access to the README via read_readme tool.",
    "",
    "--- YOUR WORKFLOW ---",
    "1. Read the README to understand all tasks and features",
    "2. Analyze each item and determine its State, Priority, and Labels",
    "3. Delegate specific Linear operations to your Linear Architect member",
    "4. Synthesize results and report to user",
    "",
    "--- SEMANTIC ANALYSIS (Your Responsibility) ---",
    "For each task in the README, YOU decide:",
    "",
    "STATE:",
    "  • [x] checkbox → Done",
    "  • [/] checkbox or '(WIP)' → In Progress",
    "  • Keywords 'Core', 'MVP', 'Must have', 'Initial', 'Basic' → Todo",
    "  • Keywords 'Eventually', 'Later', 'Future', 'Maybe' → Backlog",
    "  • Default (no special markers) → Backlog",
    "",
    "PRIORITY:",
    "  • 'Critical', 'Blocker', 'Fix' → Urgent (1)",
    "  • 'Core', 'MVP', 'Security', 'OAuth' → High (2)",
    "  • Standard features → Medium (3)",
    "  • 'Eventually', 'Later', 'Polish', 'Experiment' → Low (4)",
    "",
    "LABELS:",
    "  • Extract from section headers (e.g., '## Deployment' → label: 'Deployment')",
    "  • Use the deepest/most specific header above each task",
    "  • For '### Shell Tools' under '## Features' → label: 'Shell Tools'",
    "",
    "--- DELEGATION STYLE ---",
    "First ask: 'Get current teams and issues from Linear'",
    "Then give SPECIFIC batched instructions:",
    "  ✓ 'Create these issues: [{title: \"Web app\", state: \"Todo\", priority: 2, labels: [\"Deployment\"]}, ...]'",
    "  ✓ 'Update issue X to state InProgress, priority High'",
    "  ✗ 'Sync everything' (too vague)",
    "",
    "--- DIFF LOGIC ---",
    "Compare README (desired) vs Linear (current):",
    "  • TO_CREATE: In README, not in Linear (fuzzy match titles)",
    "  • TO_UPDATE: In both, but state/priority differs from your analysis",
    "  • TO_DELETE: In Linear, completely absent from README (be conservative)",
    "",
    "You are the strategic brain. Linear Architect is your hands."
]

agent_prompt=[
    "You are the Linear Architect, a specialized Linear API executor.",
    "You handle ALL interactions with Linear's GraphQL API.",
    "",
    "--- YOUR ROLE ---",
    "Execute precise operations as instructed by Team Lead:",
    "  • Fetch teams, states, and current issues",
    "  • Create/update/delete issues in batches",
    "  • Ensure labels exist before using them",
    "  • Report results concisely",
    "",
    "--- MEMORY RULE ---",
    "Call get_teams and get_issues ONCE when first requested.",
    "YOU REMEMBER THE RESULTS for the rest of this conversation.",
    "Do NOT re-call unless Team Lead explicitly says 'refresh' or 'get latest'.",
    "",
    "--- EXECUTION PATTERN ---",
    "Team Lead will give you structured data like:",
    "  'Create: [{title: \"X\", state_id: \"abc\", priority: 2}, ...]'",
    "",
    "You:",
    "  1. Call ensure_labels for any new labels",
    "  2. Call batch_create_issues with the exact data provided",
    "  3. Report: 'Created 5 issues: Web app, OAuth, ...'",
    "",
    "--- RESPONSE STYLE ---",
    "Be concise. Team Lead needs summaries, not raw JSON:",
    "  ✓ 'Fetched 12 current issues across 3 states'",
    "  ✓ 'Created 8 issues, updated 3, no deletions'",
    "  ✗ [500 lines of JSON dump]",
    "",
    "You are a clean, efficient execution layer.",
    "No strategic thinking. Just precise tool execution."
]

db = SqliteDb(db_file="./db_linear.db", session_table="Linear Sessions", traces_table="Linear Traces")
setup_tracing(
    db=db,
    batch_processing=True,
    max_queue_size=2048,
    max_export_batch_size=512,
    schedule_delay_millis=5000,
)


def get_linear_agent( debug_mode = True, debug_level = 2, **kwargs):
    return Agent(
    name="Linear Architect",
    id="linear_architect",
    # model=groq_wrapper.get_model(),
    # model=cerebras_wrapper.get_model(id="zai-glm-4.6"),
    # model=cerebras_wrapper.get_model(id="qwen-3-235b-a22b-instruct-2507"),
    model=cerebras_wrapper.get_model(
        id="gpt-oss-120b", estimated_tokens=3000,
    ),
    tools=[linear_tools],
    tool_call_limit=50,
    description=desc,
      
    # instructions=agent_prompt,
    markdown=True,
    
    db=db,
    retries=3,
    delay_between_retries=1,

    compress_tool_results=True,
    compression_manager=tool_compression_manager,

    add_history_to_context=True,
    num_history_runs=3,
    max_tool_calls_from_history=2,

    read_chat_history=True,
    read_tool_call_history=True,

    add_session_summary_to_context=True,    
    session_summary_manager=s_sum_manager,

    search_session_history=True,
    num_history_sessions=5,
    
    debug_mode=debug_mode,
    debug_level=debug_level,

    **kwargs
)

# TEAM
def get_linear_team( debug_mode = True, debug_level = 2, **kwargs):
    return Team(
        name="Linear Team",
        id="linear_team",
        
        model=groq_wrapper.get_model(estimated_tokens=3000),
        # model=cerebras_wrapper.get_model(id="zai-glm-4.6"),
        # model=cerebras_wrapper.get_model(id="qwen-3-235b-a22b-instruct-2507"),
        # model=cerebras_wrapper.get_model(id="gpt-oss-120b"),
        # model=cerebras_wrapper.get_model(id="zai-glm-4.6"),
        
        tools=[read_readme],
        members=[get_linear_agent(debug_mode=debug_mode, debug_level=debug_level, **kwargs)],
        description="Team for managing Linear boards and issues.",
        instructions=team_prompt,
        show_members_responses=True,
        markdown=True,
        
        db=db,
        retries=3,
        delay_between_retries=1,

        compress_tool_results=True,
        compression_manager=tool_compression_manager,

        add_history_to_context=True,
        num_history_runs=3,
        max_tool_calls_from_history=2,
        # read_chat_history=True,

        add_session_summary_to_context=True,    
        session_summary_manager=s_sum_manager,

        # search_session_history=True,
        # num_history_sessions=5,
        
        debug_mode=debug_mode,
        debug_level=debug_level,
        
        **kwargs
    )


if __name__ == "__main__":
    agent = get_linear_agent()
    agent.print_response("Sync my README to the Linear board.")

