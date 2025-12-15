import asyncio
from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini

# Adjust import path as needed
from env_manager import MultiProviderWrapper

# 1. Setup Wrappers
cerebras_wrapper = MultiProviderWrapper.from_env(
    provider='cerebras',
    model_class=Cerebras, 
    default_model_id='llama3.1-8b', 
    temperature=0.7,
    env_file="./local.env"
)

groq_wrapper = MultiProviderWrapper.from_env(
    provider='groq',
    model_class=Groq,
    default_model_id='llama-3.3-70b-versatile',
    env_file="./local.env"
)

gemini_wrapper = MultiProviderWrapper.from_env(
    provider='gemini',
    model_class=Gemini,
    default_model_id='gemini-2.5-flash',
    env_file="./local.env"
)

wrappers = {
    "Cerebras": cerebras_wrapper, 
    "Groq": groq_wrapper, 
    "Gemini": gemini_wrapper
}

request_prompt = "Write a 1-sentence interesting fact about space."

async def main():
    # Store usage data to demonstrate specific reports later
    usage_history = {} 

    print(f"\n{'='*20} STARTING REQUESTS {'='*20}")

    for provider_name, wrapper in wrappers.items():
        print(f"\n--- Running {provider_name} ---")
        
        try:
            # 1. Get Model
            model = wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
            
            # 2. Capture Identifier for Reports
            # We use the key suffix to query stats later
            used_key = model.api_key 
            used_model_id = wrapper.default_model_id
            
            usage_history[provider_name] = {
                "key": used_key,
                "model_id": used_model_id
            }
            
            agent = Agent(model=model, markdown=True)
            
            # 3. Execute
            print(f"-> Using Key: ...{used_key[-8:]}")
            await agent.aprint_response(request_prompt, stream=True)
            
        except Exception as e:
            print(f"-> FAILED: {e}")
            usage_history[provider_name] = None

        await asyncio.sleep(0.5)

    print(f"\n\n{'='*20} GENERATING REPORTS {'='*20}")

    for provider_name, wrapper in wrappers.items():
        history = usage_history.get(provider_name)
        if not history:
            continue

        print(f"\n\n>>> REPORTS FOR {provider_name.upper()} <<<\n")

        # 1. Global Stats (All keys, all models)
        # Useful for: High-level dashboard
        wrapper.print_global_stats()
        
        # 2. Key Stats (Specific Key)
        # Useful for: Checking if a specific key is hitting limits
        wrapper.print_key_stats(identifier=history['key'])

        # 3. Model Stats (Specific Model)
        # Useful for: Checking which keys are contributing to a heavy model
        wrapper.print_model_stats(model_id=history['model_id'])

        # 4. Granular Stats (Specific Key + Specific Model)
        # Useful for: Debugging specific key/model pairings
        wrapper.print_granular_stats(
            identifier=history['key'], 
            model_id=history['model_id']
        )

if __name__ == "__main__":
    asyncio.run(main())