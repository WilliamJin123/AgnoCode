import asyncio
import sqlite3
import time
from agno.agent import Agent
from key_manager import MultiProviderWrapper
from companion import WithCompanionAgent
from agno.db.sqlite import SqliteDb

ENV_FILE = "../../api_management/.env"
DB_FILE = "companion_test.db"

wrapper = MultiProviderWrapper.from_env(
    provider="cerebras", default_model_id="llama3.1-8b", env_file=ENV_FILE
)

data_worker = Agent(
    model=wrapper.get_model(),
    name="Data Engineer",
    instructions=["You are a python data engineer. Output concise python code only. Sign off all message with 'Signed by the Data Engineer'."],
    db=SqliteDb(db_file=DB_FILE, session_table="data_worker-sessions")
)

print("--- Initializing Companion Agent ---")
companion_system = WithCompanionAgent(
    worker=data_worker,
    db_path=DB_FILE,
    full_context_window=1, 
    instructions="Sign all message at the end with 'Signed by the Companion'",
    router_wrapper=wrapper
    # model_settings={"id": "llama-3.3-70b", "provider": "cerebras"},
    # model_settings={"id": "llama-3.3-70b-versatile", "provider": "groq"},
    # router_settings={"id": "llama-3.1-8b-instant", "provider": "groq"}
)

if __name__ == "__main__":
    print(f"\n--- Starting Multi-Mode Test Simulation (DB: {DB_FILE}) ---")

    print("\n--- TEST 1: Sync Streaming ---")
    user_input_1 = "Create a dictionary of 5 users with random ages."
    data_worker.print_response(user_input_1, markdown=True, stream=True)
    time.sleep(1)

    print("\n--- TEST 2: Sync Non-Streaming ---")
    user_input_2 = "Write a function to filter users over 30."
    # We call .run directly to test the non-streaming synchronous path
    data_worker.print_response(user_input_2, stream=False, markdown=True)
    time.sleep(1)

    # 3. Async Non-Streaming (.arun call)
    # Testing: arun_dispatch -> _arun_standard -> _aupdate_worker_state
    print("\n--- TEST 3: Async Non-Streaming ---")
    user_input_3 = "Convert this data to a Pandas DataFrame."
    asyncio.run(data_worker.aprint_response(user_input_3, stream=False, markdown=True))
    time.sleep(1)

    # 4. Final Handoff: Companion Verification
    # Testing: The router logic and summary retrieval from the 3 previous modes
    print("\n--- TEST 4: Companion Routing + Async Streaming ---")
    user_input_4 = "Explain what we have built so far and why we used a DataFrame."
    asyncio.run(data_worker.aprint_response(user_input_4, markdown=True, stream=True))

    # --- Verifying SQLite Persistence ---
    print("\n--- Verifying SQLite Persistence ---")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("SELECT * FROM session_summaries ORDER BY id ASC")
        rows = cursor.fetchall()
        print(f"Found {len(rows)} summarized records in DB:")
        for row in rows:
            print(f"[{row[2]}] ID {row[0]}: {row[1]}")

    print("\n--- Test Complete ---")
