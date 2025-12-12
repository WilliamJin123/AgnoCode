from key_manager_module.key_manager.env_manager import MultiProviderWrapper
import time

from agno.agent import Agent
from agno.models.cerebras import Cerebras
from agno.models.groq import Groq
from agno.models.google import Gemini

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
            model = wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
            
            agent = Agent(model=model, markdown=True)
            
            print(f"-> Key **...{model.api_key[-8:]}** selected. Sending request...")
            
            start_time = time.time()
            response = await agent.aprint_response(request_prompt, stream=True)
            duration = time.time() - start_time

        except RuntimeError as e:
            print(f"-> FAILED: Could not get a key or request failed: {e}")
        except Exception as e:
            print(f"-> FAILED: An API or network error occurred: {e}")

    for provider_name, wrapper in wrappers.items():
        print(f"Printing stats for {provider_name}")
        wrapper.print_global_stats()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())