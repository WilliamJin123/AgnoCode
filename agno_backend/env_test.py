import asyncio
import time
from pathlib import Path

from key_manager import MultiProviderWrapper
from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini

ENV_FILE = "../env/.env"
DB_FILE = "../env/api_usage.db"

async def main():
    
    print(f" STARTING COMPREHENSIVE KEY MANAGER TEST\n")
    
    # Cerebras: Will use for Rapid Rotation Testing
    cerebras = MultiProviderWrapper.from_env(
        provider="cerebras",
        model_class=Cerebras, 
        default_model_id='zai-glm-4.6', 
        env_file=ENV_FILE,
        db_path=DB_FILE
    )

    # Groq: Will use for Rate Limit / Token Estimation Testing
    groq = MultiProviderWrapper.from_env(
        provider="groq",
        model_class=Groq,
        default_model_id='moonshotai/kimi-k2-instruct-0905',
        env_file=ENV_FILE,
        db_path=DB_FILE
    )

    # Gemini: Will use for Streaming Integration
    gemini = MultiProviderWrapper.from_env(
        provider="gemini",
        model_class=Gemini,
        default_model_id='gemini-2.5-flash',
        env_file=ENV_FILE,
        db_path=DB_FILE
    )
    
    print(f"\n[CEREBRAS] TEST: Sticky Rotation")
    print("Goal: Fire 3 sequential requests. Should see the same keys as long as they aren't overloaded ")
    
    for i in range(1, 4):
        try:
            # We don't wait long, fail fast if keys are exhausted
            model, usage = cerebras.get_model(estimated_tokens=50, wait=True, timeout=5)
            suffix = usage.api_key[-8:] if len(usage.api_key) > 8 else usage.api_key
            
            print(f"   -> Request {i}: Acquired Key ...{suffix}")
            
            agent = Agent(model=model)
            response = agent.run(f"Say 'Hello {i}' and nothing else.")
            print(f"      Output: {response.content}\n")
            
        except Exception as e:
            print(f"   -> Request {i} Failed: {e}")
            

    print(f"\n[GROQ] TEST: Token Estimation Logic")
    print("Goal: Request a model with high estimated usage to test rate-limit checker.")
    
    try:
        # We estimate 2000 tokens. The manager should check if any key has 
        # room for 2000 tokens in its TPM/TPD buckets.
        est_tokens = 2000
        print(f"   -> Requesting key with {est_tokens} estimated tokens buffer...")
        
        model, usage = groq.get_model(estimated_tokens=est_tokens, wait=True)
        print(f"   -> Success! Key ...{usage.api_key[-8:]} had sufficient capacity.")
        
        agent = Agent(model=model)
        await agent.aprint_response("Explain quantum entanglement in 1 sentence.", stream=True)
        
    except RuntimeError as e:
        print(f"   -> Expected Behavior (if limits low): {e}")
    except Exception as e:
        print(f"   -> Unexpected Error: {e}")

    print(f"\n[GEMINI] TEST: Misc.")
    
    try:
        model, usage = gemini.get_model(wait=True)
        print(f"   -> Streaming with Key ...{usage.api_key[-8:]}")
        
        agent = Agent(model=model)
        # Testing a different call method
        response = await agent.arun("List 3 fruits.", stream=False)
        print(f"   -> Response: {response.content}")
        
    except Exception as e:
        print(f"   ->  Gemini Test Failed: {e}")

    print(f"\n{'='*20} STATS AUDIT {'='*20}")
      
    # # A. Global Stats (Cerebras)
    # print(f"\n--- Cerebras Global Stats ---")
    # cerebras.print_global_stats()

     # B. Model Specific Stats (Cerebras)
    print(f"\n--- Cerebras Model Stats (zai-glm-4.6) ---")
    groq.print_model_stats('zai-glm-4.6')
    
    
    # B. Model Specific Stats (Groq)
    print(f"\n--- Groq Model Stats (moonshotai/kimi-k2-instruct-0905) ---")
    groq.print_model_stats('moonshotai/kimi-k2-instruct-0905')

    # C. Specific Key Stats (Gemini - utilizing the last used key)
    # We grab the key from the loop above (usage.api_key)
    if 'usage' in locals():
        print(f"\n--- Gemini Specific Key Stats ---")
        gemini.print_key_stats(usage.api_key)
        
        print(f"\n--- Gemini Granular Stats (Key + Model) ---")
        gemini.print_granular_stats(usage.api_key, 'gemini-2.5-flash')

    print(f"\n{'='*50}\n TEST SUITE COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())