import os
import time
from agno.agent import Agent
from key_manager import MultiProviderWrapper
from companion import WithCompanionAgent

ENV_FILE = "../../api_management/.env"
DB_FILE = "test_project.db"

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)


wrapper = MultiProviderWrapper.from_env(
    provider="cerebras", default_model_id="llama3.1-8b", env_file=ENV_FILE
)

data_worker = Agent(
    model=wrapper.get_model(),
    name="Data Engineer",
    instructions=["You are a python data engineer. Output concise python code only."]
)

print("--- Initializing Companion Agent ---")
companion_system = WithCompanionAgent(
    worker=data_worker,
    db_path=DB_FILE,
    full_context_window=1, 
    instructions="Sign all message at the end with 'Signed, William Jin'"
    # model_settings={"id": "llama-3.3-70b", "provider": "cerebras"},
    # router_settings={"id": "llama-3.1-8b-instant", "provider": "groq"}
)

if __name__ == "__main__":

    test_script = [
        # Turn 1: Worker Task
        ("Create a dictionary of 5 users with random ages.", "WORKER"),
        
        # Turn 2: Worker Task (This will push Turn 1 into SQLite summary)
        ("Write a function to filter users over 30.", "WORKER"),
        
        # Turn 3: Worker Task (This will push Turn 2 into SQLite summary)
        ("Convert this data to a Pandas DataFrame.", "WORKER"),
        
        # Turn 4: Companion Query (Should access SQLite summaries to answer)
        ("Explain what we have built so far and why we used a DataFrame.", "COMPANION_ROUTED")
    ]

    print(f"\n--- Starting Test Simulation (DB: {DB_FILE}) ---")

    for user_input, expected_route in test_script:
        print(f"\n> USER: {user_input}")
        
        # We use .run() which is now the patched version
        response = data_worker.run(user_input)
        
        print(f"> RESPONSE: {response.content}\n")
        print("-" * 50)
        time.sleep(1) # Pause for readability

    import sqlite3
    print("\n--- Verifying SQLite Persistence ---")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("SELECT * FROM project_summaries")
        rows = cursor.fetchall()
        print(f"Found {len(rows)} summarized records in DB:")
        for row in rows:
            print(f"ID: {row[0]} | Summary: {row[1]}")

    print("\n--- Test Complete ---")

