import asyncio
from key_manager import MultiProviderWrapper
from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.models.openrouter import OpenRouter

ENV_FILE = ".env"
DB_FILE = "api_usage.db"

async def run_stress_test(provider_name, wrapper, model_id, limit_attribute_name):
    print(f"\n{'#'*60}")
    print(f"STRESS TEST: {provider_name.upper()} ({model_id})")
    print(f"Targeting Limit: {limit_attribute_name}")
    print(f"{'#'*60}\n")

    # 1. ROBUST FETCH: Try specific model ID first, then fallback to 'default'
    provider_limits = wrapper.MODEL_LIMITS.get(provider_name, {})
    limits_config = provider_limits.get(model_id)
    
    if not limits_config:
        print(f" Specific limit for '{model_id}' not found. Using 'default'.")
        limits_config = provider_limits.get('default')

    if not limits_config:
        print(f" No limits configuration found for {provider_name}. Cannot stress test.")
        return

    # 2. Get the actual integer value
    limit_value = getattr(limits_config, limit_attribute_name, None)

    if limit_value is None:
        print(f" Limit '{limit_attribute_name}' is None (unlimited). Cannot stress test.")
        return

    # 3. Set target loops (Limit + 2 to force a switch)
    target_requests = limit_value + 2
    print(f"-> Limit is {limit_value}. Running {target_requests} requests to force rotation.\n")

    for i in range(1, target_requests + 1):
        try:
            # Get model (this counts against our local limit logic)
            # wait=False ensures we fail fast if our local logic says "Stop"
            model = wrapper.get_model(id=model_id, estimated_tokens=10, wait=False)
            
            # Safe key printing
            current_key = "UNKNOWN"
            if hasattr(model, 'api_key') and model.api_key:
                current_key = model.api_key[-8:]
            
            print(f"[{i}/{target_requests}] Key ...{current_key} | ", end="", flush=True)

            # Fire a cheap request
            agent = Agent(model=model)
            response = await agent.arun("hi", stream=False)
            
            print(" Success")
            
            # 4. CRITICAL: Small sleep to avoid 'burst' limits (helps with Gemini)
            await asyncio.sleep(0.5)

        except RuntimeError as e:
            print(f" BLOCKED (Local): {e}")
            print(f"   (This confirms the Local Rate Limiter is working!)")
            break
        except Exception as e:
            # This catches 429s from the provider that slipped past our local limiter
            print(f" API Error (Provider): {e}")
            # If we hit a real provider error, we should probably stop or slow down
            await asyncio.sleep(2)
async def main():
    # 1. Initialize Wrappers
    cerebras = MultiProviderWrapper.from_env("cerebras", env_file=ENV_FILE)
    groq = MultiProviderWrapper.from_env("groq", 'groq/compound-mini', env_file=ENV_FILE)
    gemini = MultiProviderWrapper.from_env("gemini", 'gemini-2.5-flash', env_file=ENV_FILE)
    openrouter = MultiProviderWrapper.from_env("openrouter", 'tngtech/deepseek-r1t2-chimera:free', env_file=ENV_FILE)
    
    # 2. Run Stress Tests
    
    # Test A: Cerebras (Hourly Requests)
    # await run_stress_test("cerebras", cerebras, 'zai-glm-4.6', 'requests_per_minute')

    # Test B: Groq (Daily Requests)
    # await run_stress_test("groq", groq, 'groq/compound-mini', 'requests_per_day')

    # Test C: Gemini (Daily Requests)
    await run_stress_test("gemini", gemini, 'gemini-2.5-flash', 'requests_per_minute')

    # Test D: OpenRouter (Requests Per Minute)
    await run_stress_test( "openrouter",  openrouter,  'tngtech/deepseek-r1t2-chimera:free',  'requests_per_minute')

    print("\n✅ ALL STRESS TESTS COMPLETED")

if __name__ == "__main__":
    asyncio.run(main())