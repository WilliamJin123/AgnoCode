import asyncio
from key_manager import MultiProviderWrapper
from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.models.openrouter import OpenRouter

ENV_FILE = ".env"
DB_FILE = "api_usage.db"

async def main():
    print("--- STARTING SIMPLE KEY MANAGER TEST ---\n")

    # 1. Initialize Wrappers
    print("Initializing Wrappers...")
    cerebras = MultiProviderWrapper.from_env("cerebras", Cerebras, 'llama3.1-8b', env_file=ENV_FILE, db_path=DB_FILE)
    groq = MultiProviderWrapper.from_env("groq", Groq, 'llama-3.3-70b-versatile', env_file=ENV_FILE, db_path=DB_FILE)
    gemini = MultiProviderWrapper.from_env("gemini", Gemini, 'gemini-2.5-flash', env_file=ENV_FILE, db_path=DB_FILE)
    openrouter = MultiProviderWrapper.from_env("openrouter", OpenRouter, 'qwen/qwen3-coder:free', env_file=ENV_FILE, db_path=DB_FILE)
    # 2. Test Basic Rotation (Cerebras)
    print("\n[CEREBRAS] Testing Key Rotation (3 Requests)")
    for i in range(3):
        try:
            model = cerebras.get_model()
            print(f"  Req {i+1}: Key ...{model.api_key[-8:]} -> ", end="", flush=True)
            
            agent = Agent(model=model)
            agent.print_response("Say 'Confirmed'", stream=False)
            agent.print_response("Say 'Confirmed (with Stream)'", stream=True)
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
        await agent.aprint_response("List 3 fruits.", stream=False)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[OPENROUTER] Testing Free Model with Key Rotation")
    openrouter_last_key = None
    try:
        model = openrouter.get_model()
        openrouter_last_key = model.api_key
        print(f"  Req {i+1}: Key ...{model.api_key[-8:]} -> ", end="", flush=True)
        
        agent = Agent(model=model)
        agent.print_response("Write a Python function to add two numbers.", stream=False)
    except Exception as e:
        print(f"Failed: {e}")

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

    if openrouter_last_key:
        print(f"\n7. Key Stats (OpenRouter - {openrouter_last_key[-8:]}):")
        openrouter.print_key_stats(openrouter_last_key)

        print(f"\n8. Granular Stats (OpenRouter Key + qwen/qwen3-coder:free):")
        openrouter.print_granular_stats(openrouter_last_key, 'qwen/qwen3-coder:free')

    print("\n--- TEST COMPLETE ---")
    
if __name__ == "__main__":
    asyncio.run(main())