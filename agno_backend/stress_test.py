import asyncio
from key_manager import MultiProviderWrapper
from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini

ENV_FILE = "../env/.env"
DB_FILE = "../env/api_usage.db"

async def run_stress_test(provider_name, wrapper, model_id, limit_attribute_name):
    print(f"\n{'#'*60}")
    print(f"STRESS TEST: {provider_name.upper()} ({model_id})")
    print(f"Targeting Limit: {limit_attribute_name}")
    print(f"{'#'*60}\n")

    # 1. Dynamically fetch the limit value from the wrapper's config
    try:
        limits_config = wrapper.MODEL_LIMITS[provider_name][model_id]
        limit_value = getattr(limits_config, limit_attribute_name)
    except (KeyError, AttributeError):
        print(f"❌ Could not find limit '{limit_attribute_name}' for {model_id}")
        return

    if limit_value is None:
        print("❌ Limit value is None (unlimited). Cannot stress test.")
        return

    # 2. Set target loops (Limit + 2 to force a switch)
    target_requests = limit_value + 2
    print(f"-> Limit is {limit_value}. Running {target_requests} requests to force rotation.\n")

    for i in range(1, target_requests + 1):
        try:
            # Get model (this counts against our local limit logic)
            model = wrapper.get_model(estimated_tokens=10, wait=False)
            current_key = model.api_key[-8:]
            
            print(f"[{i}/{target_requests}] Key ...{current_key} | ", end="", flush=True)

            # Fire a cheap request
            agent = Agent(model=model)
            # We use a tiny prompt to save tokens and time
            response = await agent.arun("hi", stream=False)
            
            print("✅ Success")
            
            # Tiny sleep to prevent local OS networking issues
            await asyncio.sleep(0.1)

        except RuntimeError as e:
            print(f"❌ BLOCKED: {e}")
            print(f"   (This usually means ALL keys are exhausted or rate-limited)")
            break
        except Exception as e:
            print(f"⚠️ API Error: {e}")

async def main():
    # 1. Initialize Wrappers
    cerebras = MultiProviderWrapper.from_env("cerebras", Cerebras, 'zai-glm-4.6', env_file=ENV_FILE, db_path=DB_FILE)
    groq = MultiProviderWrapper.from_env("groq", Groq, 'groq/compound-mini', env_file=ENV_FILE, db_path=DB_FILE)
    gemini = MultiProviderWrapper.from_env("gemini", Gemini, 'gemini-2.5-flash', env_file=ENV_FILE, db_path=DB_FILE)

    # 2. Run Stress Tests
    
    # Test A: Cerebras (Hourly Requests)
    # This will likely be 100+ requests. Ensure you have enough keys or patience.
    await run_stress_test("cerebras", cerebras, 'zai-glm-4.6', 'requests_per_minute')

    # Test B: Groq (Daily Requests)
    # 'groq/compound-mini' usually has a low daily limit (e.g. 250)
    # await run_stress_test("groq", groq, 'groq/compound-mini', 'requests_per_day')

    # Test C: Gemini (Daily Requests)
    # 'gemini-2.5-flash' usually has a very low limit (e.g. 20), making this the fastest test
    # await run_stress_test("gemini", gemini, 'gemini-2.5-flash', 'requests_per_day')

    print("\n✅ ALL STRESS TESTS COMPLETED")

if __name__ == "__main__":
    asyncio.run(main())