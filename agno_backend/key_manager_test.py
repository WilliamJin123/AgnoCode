import asyncio
from key_manager import MultiProviderWrapper
from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini

ENV_FILE = "../env/.env"
DB_FILE = "../env/api_usage.db"

async def main():
    print("--- STARTING SIMPLE KEY MANAGER TEST ---\n")

    # 1. Initialize Wrappers
    print("Initializing Wrappers...")
    cerebras = MultiProviderWrapper.from_env("cerebras", Cerebras, 'llama3.1-8b', env_file=ENV_FILE, db_path=DB_FILE)
    groq = MultiProviderWrapper.from_env("groq", Groq, 'llama-3.3-70b-versatile', env_file=ENV_FILE, db_path=DB_FILE)
    gemini = MultiProviderWrapper.from_env("gemini", Gemini, 'gemini-2.5-flash', env_file=ENV_FILE, db_path=DB_FILE)

    # 2. Test Basic Rotation (Cerebras)
    print("\n[CEREBRAS] Testing Key Rotation (3 Requests)")
    for i in range(3):
        try:
            model = cerebras.get_model()
            print(f"  Req {i+1}: Key ...{model.api_key[-8:]} -> ", end="", flush=True)
            
            agent = Agent(model=model)
            agent.print_response("Say 'Confirmed'", stream=False)
        except Exception as e:
            print(f"Failed: {e}")

    # 3. Test Capacity Check (Groq)
    print("\n[GROQ] Testing High-Load Token Estimation (2000 tokens)")
    try:
        # Should verify if any key has 2000 tokens of capacity available
        model = groq.get_model(estimated_tokens=2000) 
        print(f"  Success: Key ...{model.api_key[-8:]} accepted load.")
        
        agent = Agent(model=model)
        await agent.aprint_response("Explain quantum entanglement in one sentence.", stream=True)
    except RuntimeError as e:
        print(f"  Expected Limit Reached: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. Test Async Streaming (Gemini)
    print("\n[GEMINI] Testing Async Streaming")
    last_key = None
    try:
        model = gemini.get_model()
        last_key = model.api_key
        print(f"  Using Key ...{last_key[-8:]}")
        
        agent = Agent(model=model)
        await agent.aprint_response("List 3 fruits.", stream=True)
    except Exception as e:
        print(f"  Error: {e}")

    # 5. Test Statistics
    print("\n" + "="*20 + " STATS AUDIT " + "="*20)
    
    print("\n1. Global Stats (Cerebras):")
    cerebras.print_global_stats()

    print("\n2. Model Stats (Groq - llama-3.3-70b-versatile):")
    groq.print_model_stats('llama-3.3-70b-versatile')

    if last_key:
        print(f"\n3. Key Stats (Gemini - {last_key[-8:]}):")
        gemini.print_key_stats(last_key)

        print(f"\n4. Granular Stats (Gemini Key + gemini-2.5-flash):")
        gemini.print_granular_stats(last_key, 'gemini-2.5-flash')

    print("\n--- TEST COMPLETE ---")

if __name__ == "__main__":
    asyncio.run(main())