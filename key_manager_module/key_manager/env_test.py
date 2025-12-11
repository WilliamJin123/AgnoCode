from key_manager_module.key_manager.env_manager import MultiProviderWrapper
import time

from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.utils.pprint import pprint_run_response


cerebras_wrapper = MultiProviderWrapper.from_env(
    provider='cerebras',
    model_class=Cerebras, 
    model_id='llama3.1-8b',
    # Pass model-specific kwargs
    temperature=0.7,
    max_completion_tokens=512,
    env_file="./local.env"
)

groq_wrapper = MultiProviderWrapper.from_env(
    provider='groq',
    model_class=Groq,
    model_id='llama-3.3-70b-versatile',
    env_file="./local.env"
)

gemini_wrapper = MultiProviderWrapper.from_env(
    provider='gemini',
    model_class=Gemini,
    model_id='gemini-2.5-flash',
    env_file="./local.env"
)


wrappers = {
    "Cerebras": cerebras_wrapper, 
    "Groq": groq_wrapper, 
    "Gemini": gemini_wrapper
}

request_prompt = "Write a 5-sentence story about a historical object being found in the future."
async def main():
    for provider_name, wrapper in wrappers.items():
        print(f"\n{'='*20} RUNNING {provider_name.upper()} {'='*20}")
        
        try:
            model, key_usage = wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
            
            agent = Agent(model=model, markdown=True)
            
            print(f"-> Key **...{key_usage.api_key[-8:]}** selected. Sending request...")
            
            start_time = time.time()
            response = await agent.aprint_response(request_prompt, stream=True)
            duration = time.time() - start_time

        except RuntimeError as e:
            print(f"-> FAILED: Could not get a key or request failed: {e}")
        except Exception as e:
            print(f"-> FAILED: An API or network error occurred: {e}")

    for wrapper in wrappers.values():
        print("Printing stats")
        wrapper.print_stats(start=1, end=1)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())