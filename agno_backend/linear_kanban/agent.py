from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.cerebras import Cerebras
from architect_tools import LinearTools
from key_manager import MultiProviderWrapper
from agno.db.sqlite import SqliteDb
from agno.session import SessionSummaryManager
from agno.compression import CompressionManager
import os

ENV_FILE="../env/.env"
DB_FILE="../env/api_usage.db"
README_PATH="../README.md"
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
    model_class=Groq,
    default_model_id='moonshotai/kimi-k2-instruct-0905',
    env_file=ENV_FILE,
    db_path=DB_FILE
)

cerebras_wrapper = MultiProviderWrapper.from_env(
    provider='cerebras',
    model_class=Cerebras,
    default_model_id='llama-3.3-70b',
    env_file=ENV_FILE,
    db_path=DB_FILE
)

s_sum_manager = SessionSummaryManager(
    model=cerebras_wrapper.get_model(),
    summary_request_message="Provide the summary of the conversation. Include all relevant details",
    
)

tool_compression_manager = CompressionManager(
    model=cerebras_wrapper.get_model(),
    compress_tool_results_limit=4
)

desc="You sync a README to Linear, creating labels for categories."

def get_linear_agent( debug_mode = True, debug_level = 2, **kwargs):
    return Agent(
    name="Linear Architect",
    id="linear_architect",
    # model=groq_wrapper.get_model(),
    model=cerebras_wrapper.get_model(id="zai-glm-4.6"),
    # model=cerebras_wrapper.get_model(id="gpt-oss-120b"),
    tools=[linear_tools, read_readme],
    tool_call_limit=50,
    description=desc,
    introduction="Hi, I am a Linear Taskboard Agent",
    
    instructions=[
        "You are the 'Linear Architect', a Product Manager agent responsible for syncing a README.md file (Source of Truth) with a Linear.app board (Execution State).",
        
        "--- PHASE 1: DISCOVERY ---",
        "1. ENVIRONMENT:",
        "   - Call `get_teams`. Map Team ID and State IDs (Backlog, Todo, In Progress, Done).",
        "   - Call `get_issues`. Fetch the CURRENT_STATE (IDs, Titles, States).",
        "   - Read `README.md`. This is the DESIRED_STATE.",
        
        "2. LABEL STRATEGY:",
        "   - Your README uses Functional Headers (e.g., '## Deployment', '### Shell Tools').",
        "   - Treat the **deepest** header as the Label.",
        "   - Call `ensure_labels` for all identified headers.",
        
        "--- PHASE 2: SEMANTIC TRIAGE (CRITICAL) ---",
        "You must determine the correct State and Priority for each task based on its text content.",
        
        "1. STATE LOGIC (Where does it go?):",
        "   - **Done:** If the task starts with `[x]`.",
        "   - **In Progress:** If the task starts with `[/]` or text says '(WIP)'.",
        "   - **Backlog (Default):** Any standard task.",
        "   - **Todo (Smart Promotion):** Promote a task from Backlog to Todo if:",
        "     - It seems critical for an MVP (e.g., 'Web app', 'OAuth', 'Package management').",
        "     - It contains keywords: 'Core', 'Must have', 'Initial', 'Basic'.",
        "   - **Deprioritize:** Force to Backlog if text says: 'Eventually', 'Later', 'Future', 'Maybe'.",

        "2. PRIORITY LOGIC (How urgent is it?):",
        "   - **Urgent (1):** 'Critical', 'Blocker', 'Fix'.",
        "   - **High (2):** 'Core', 'MVP', 'Security', 'OAuth'.",
        "   - **Medium (3):** Standard features.",
        "   - **Low (4):** 'Eventually', 'Later', 'UI polish', 'Experiments'.",
        
        "--- PHASE 3: DIFF & EXECUTION ---",
        "Compare DESIRED vs CURRENT state. Use fuzzy matching for titles.",
        
        "1. PREPARE BATCHES:",
        "   - **TO_CREATE:** New items. Apply the State/Priority/Label logic derived above.",
        "   - **TO_UPDATE:** Existing items where State/Priority/Title differs from your new calculation.",
        "   - **TO_DELETE:** Items in Linear that are completely absent from the README (Be conservative).",
        
        "2. EXECUTE:",
        "   - Call `batch_create_issues` for new items.",
        "   - Call `batch_update_issues` for changed items.",
        "   - Call `batch_delete_issues` for removed items.",
        
        "--- FINAL REPORT ---",
        "Report the sync summary. Explicitly mention how many items you promoted to 'Todo' vs 'Backlog'."
    ],
    
    markdown=True,
    
    db=SqliteDb(db_file="./db_linear_agent.db", session_table="Linear Agent Sessions"),
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

if __name__ == "__main__":
    agent = get_linear_agent()
    agent.print_response("Sync my README to the Linear board.")

